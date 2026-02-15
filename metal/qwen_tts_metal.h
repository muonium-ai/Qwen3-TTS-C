/*
 * qwen_tts_metal.h - Metal GPU Backend for Qwen3-TTS
 *
 * Provides GPU-accelerated compute kernels via Apple Metal.
 * All functions are C-callable (implemented in Objective-C .m file).
 */

#ifndef QWEN_TTS_METAL_H
#define QWEN_TTS_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Initialization & Lifecycle
 * ======================================================================== */

/* Initialize Metal backend. Returns 0 on success, -1 on failure. */
int metal_init(void);

/* Check if Metal is available and initialized. */
int metal_is_available(void);

/* Shutdown Metal backend, release all resources. */
void metal_shutdown(void);

/* Print Metal device info to stderr. */
void metal_print_info(void);

/* ========================================================================
 * Buffer Management
 *
 * Metal buffers are identified by opaque IDs. The backend manages a pool
 * of buffers internally. Buffers can be:
 *   - Uploaded from CPU (weights)
 *   - Created empty (scratch/intermediates)
 *   - Shared (CPU+GPU accessible for results)
 * ======================================================================== */

typedef int metal_buf_t;  /* opaque buffer ID, -1 = invalid */

#define METAL_BUF_INVALID (-1)

/* Create a GPU buffer from CPU data (copies data). Returns buffer ID. */
metal_buf_t metal_buf_create(const void *data, size_t size);

/* Wrap existing CPU pointer as a shared Metal buffer (zero-copy on unified memory).
 * The pointer must remain valid for the lifetime of the buffer.
 * For page-aligned pointers (e.g., mmap'd), uses newBufferWithBytesNoCopy.
 * For unaligned pointers, copies data. */
metal_buf_t metal_buf_from_ptr(void *ptr, size_t size);

/* Create an empty GPU buffer of given size. */
metal_buf_t metal_buf_create_empty(size_t size);

/* Resize buffer if current size < needed. Returns same or new ID. */
metal_buf_t metal_buf_ensure(metal_buf_t buf, size_t size);

/* Release a buffer. */
void metal_buf_release(metal_buf_t buf);

/* Get CPU-accessible pointer to buffer contents (shared memory on Apple Silicon). */
void *metal_buf_contents(metal_buf_t buf);

/* Get buffer size in bytes. */
size_t metal_buf_size(metal_buf_t buf);

/* Copy data from CPU to buffer. */
void metal_buf_write(metal_buf_t buf, const void *data, size_t size);

/* Copy data from buffer to CPU. */
void metal_buf_read(metal_buf_t buf, void *dst, size_t size);

/* ========================================================================
 * GPU Compute Kernels
 *
 * These operate on Metal buffers. Results are available after metal_sync().
 * Multiple kernel calls are batched into a single command buffer for
 * efficiency. Call metal_sync() when you need results on CPU.
 * ======================================================================== */

/* Synchronize: wait for all enqueued GPU work to complete. */
void metal_sync(void);

/* Begin a new command buffer (for batching multiple kernels). */
void metal_begin(void);

/* Commit the current command buffer (submit to GPU). */
void metal_commit(void);

/* --- Matrix Operations --- */

/* out[rows] = A_bf16[rows, cols] @ x[cols]  (BF16 weights, F32 in/out) */
void metal_matvec_bf16(metal_buf_t out, metal_buf_t A_bf16, metal_buf_t x,
                       int rows, int cols);

/* Fused QKV projection: computes Q, K, V matvecs in one dispatch */
void metal_qkv_matvec_bf16(metal_buf_t q, metal_buf_t k, metal_buf_t v,
                           metal_buf_t wq_bf16, metal_buf_t wk_bf16, metal_buf_t wv_bf16,
                           metal_buf_t x, int num_heads, int kv_heads, int head_dim, int hidden);

/* out[rows] = A[rows, cols] @ x[cols]  (all F32) */
void metal_matvec_f32(metal_buf_t out, metal_buf_t A, metal_buf_t x,
                      int rows, int cols);

/* C[M,N] = A[M,K] @ B[N,K]^T  (all F32) */
void metal_matmul_f32(metal_buf_t C, metal_buf_t A, metal_buf_t B,
                      int M, int N, int K);

/* C[M,N] = A[M,K] @ B_bf16[N,K]^T  (A is F32, B is BF16) */
void metal_matmul_bf16(metal_buf_t C, metal_buf_t A, metal_buf_t B_bf16,
                       int M, int N, int K);

/* --- Normalization --- */

/* RMSNorm: out[i] = weight[i] * (x[i] / sqrt(mean(x^2) + eps)) */
void metal_rms_norm(metal_buf_t out, metal_buf_t x, metal_buf_t weight,
                    int dim, float eps);

/* RMSNorm in-place on x */
void metal_rms_norm_inplace(metal_buf_t x, metal_buf_t weight,
                            int dim, float eps);

/* LayerNorm: out[i] = weight[i] * (x[i] - mean) / sqrt(var + eps) + bias[i] */
void metal_layer_norm(metal_buf_t out, metal_buf_t x, metal_buf_t weight,
                      metal_buf_t bias, int dim, float eps);

/* --- Attention --- */

/* Softmax over n elements (in-place) */
void metal_softmax(metal_buf_t x, int n);

/* Fused Q/K RMSNorm + RoPE for single-token attention */
void metal_qk_norm_rope(metal_buf_t q, metal_buf_t k,
                        metal_buf_t q_norm_weight, metal_buf_t k_norm_weight,
                        metal_buf_t cos_buf, metal_buf_t sin_buf,
                        int num_heads, int kv_heads, int head_dim, float eps);

/* KV cache write: store current k,v into [layer_idx, pos] */
void metal_kv_cache_store(metal_buf_t kv_k, metal_buf_t kv_v,
                          metal_buf_t k, metal_buf_t v,
                          int layer_idx, int pos, int kv_dim, int kv_max);

/* Single-token attention kernels over KV cache */
void metal_attn_scores(metal_buf_t scores, metal_buf_t q, metal_buf_t kv_k,
                       int layer_idx, int seq_len, int kv_max,
                       int num_heads, int kv_heads, int head_dim, float scale);
void metal_attn_softmax_rows(metal_buf_t scores, int num_heads, int seq_len, int kv_max);
void metal_attn_weighted_sum(metal_buf_t out, metal_buf_t scores, metal_buf_t kv_v,
                             int layer_idx, int seq_len, int kv_max,
                             int num_heads, int kv_heads, int head_dim);
void metal_attn_fused(metal_buf_t out, metal_buf_t scores, metal_buf_t q,
                      metal_buf_t kv_k, metal_buf_t kv_v,
                      int layer_idx, int seq_len, int kv_max,
                      int num_heads, int kv_heads, int head_dim, float scale);

/* --- Activations --- */

/* SiLU in-place: x[i] = x[i] / (1 + exp(-x[i])) */
void metal_silu_inplace(metal_buf_t x, int n);

/* GELU in-place */
void metal_gelu_inplace(metal_buf_t x, int n);

/* SnakeBeta: out = x + inv_beta * sin^2(alpha * x) */
void metal_snake_beta(metal_buf_t out, metal_buf_t x, metal_buf_t alpha,
                      metal_buf_t beta, int channels, int length);

/* --- Element-wise Operations --- */

void metal_add_inplace(metal_buf_t a, metal_buf_t b, int n);
void metal_mul_inplace(metal_buf_t a, metal_buf_t b, int n);
void metal_copy(metal_buf_t dst, metal_buf_t src, int n);
void metal_scale_inplace(metal_buf_t x, float scale, int n);
void metal_clamp(metal_buf_t x, int n, float min_val, float max_val);

/* --- RoPE --- */

void metal_rope_apply(metal_buf_t q, metal_buf_t k, metal_buf_t cos_buf,
                      metal_buf_t sin_buf, int num_heads, int head_dim);

/* --- Convolution --- */

void metal_causal_conv1d(metal_buf_t out, metal_buf_t input, metal_buf_t weight,
                         metal_buf_t bias, int in_channels, int out_channels,
                         int kernel_size, int length, int dilation, int groups);

void metal_transposed_conv1d(metal_buf_t out, metal_buf_t input, metal_buf_t weight,
                             metal_buf_t bias, int in_channels, int out_channels,
                             int kernel_size, int stride, int length, int *out_length);

/* --- Conversion --- */

void metal_bf16_to_f32(metal_buf_t out, metal_buf_t in_buf, int n);
void metal_bf16_row_to_f32(metal_buf_t out, metal_buf_t in_buf, int row_idx, int row_size);
void metal_argmax_i32(metal_buf_t out_idx, metal_buf_t x, int n);

/* --- SwiGLU fused --- */

/* gate_out = silu(A_gate @ x) * (A_up @ x) where A is [2*intermediate, hidden] BF16 */
void metal_swiglu_matvec_bf16(metal_buf_t out, metal_buf_t gate_up_bf16,
                              metal_buf_t x, int intermediate, int hidden);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_METAL_H */
