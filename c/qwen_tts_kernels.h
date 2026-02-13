/*
 * qwen_tts_kernels.h - Math kernel declarations for Qwen3-TTS
 */

#ifndef QWEN_TTS_KERNELS_H
#define QWEN_TTS_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * BLAS configuration
 * ======================================================================== */

#ifdef USE_BLAS
  #ifdef ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #elif defined(USE_OPENBLAS)
    #include <cblas.h>
  #endif
#endif

/* ========================================================================
 * Core math operations
 * ======================================================================== */

/* RMSNorm: out[i] = weight[i] * (x[i] / sqrt(mean(x^2) + eps)) */
void kernel_rms_norm(float *out, const float *x, const float *weight, int dim, float eps);

/* RMSNorm in-place */
void kernel_rms_norm_inplace(float *x, const float *weight, int dim, float eps);

/* LayerNorm: out[i] = weight[i] * (x[i] - mean) / sqrt(var + eps) + bias[i] */
void kernel_layer_norm(float *out, const float *x, const float *weight, const float *bias, int dim, float eps);

/* Matrix-vector multiply: out = A @ x, A is [rows, cols], x is [cols], out is [rows]
 * A stored as BF16 */
void kernel_matvec_bf16(float *out, const uint16_t *A_bf16, const float *x, int rows, int cols);

/* Matrix-vector multiply: out = A @ x, A is [rows, cols] F32 */
void kernel_matvec_f32(float *out, const float *A, const float *x, int rows, int cols);

/* Matrix-matrix multiply: C = A @ B^T
 * A is [M, K], B is [N, K], C is [M, N] - all F32 */
void kernel_matmul_f32(float *C, const float *A, const float *B, int M, int N, int K);

/* Matrix-matrix multiply with BF16 weight:
 * C = A @ B^T, A is [M, K] F32, B is [N, K] BF16, C is [M, N] F32 */
void kernel_matmul_bf16(float *C, const float *A, const uint16_t *B_bf16, int M, int N, int K);

/* Fused gate+up matvec for SwiGLU:
 * gate = A_gate @ x, up = A_up @ x (both BF16)
 * out[i] = silu(gate[i]) * up[i] */
void kernel_swiglu_matvec_bf16(float *out, const uint16_t *gate_up_fused_bf16,
                                const float *x, int intermediate, int hidden);

/* SiLU activation: x * sigmoid(x) */
void kernel_silu_inplace(float *x, int n);

/* GELU activation */
void kernel_gelu_inplace(float *x, int n);

/* SnakeBeta activation: x + inv_beta * sin^2(alpha * x)
 * alpha/inv_beta are preprocessed at model load time. */
void kernel_snake_beta(float *out, const float *x, const float *alpha,
                       const float *beta, int channels, int length);

/* Element-wise add: out[i] = a[i] + b[i] */
void kernel_add(float *out, const float *a, const float *b, int n);

/* Element-wise add in-place: a[i] += b[i] */
void kernel_add_inplace(float *a, const float *b, int n);

/* Element-wise multiply in-place: a[i] *= b[i] */
void kernel_mul_inplace(float *a, const float *b, int n);

/* Scale in-place: x[i] *= scale */
void kernel_scale_inplace(float *x, float scale, int n);

/* Softmax over n elements */
void kernel_softmax(float *x, int n);

/* Top-K sampling: returns sampled index */
int kernel_sample_top_k(const float *logits, int vocab_size, int top_k,
                        float top_p, float temperature, float *rng_state);

/* Apply repetition penalty to logits */
void kernel_apply_repetition_penalty(float *logits, const int *token_ids,
                                     int n_tokens, int vocab_size, float penalty);

/* RoPE: apply rotary position embedding to q/k vectors
 * q/k shape: [num_heads, head_dim] */
void kernel_rope_apply(float *q, float *k, const float *cos, const float *sin,
                       int num_heads, int head_dim);

/* M-RoPE: multimodal rotary position embedding
 * Uses 3 position streams (temporal, height, width) */
void kernel_mrope_apply(float *q, float *k, const float *cos, const float *sin,
                        int num_heads, int head_dim, const int *mrope_section);

/* Causal Conv1d forward pass */
void kernel_causal_conv1d(float *out, const float *input, const float *weight,
                          const float *bias, int in_channels, int out_channels,
                          int kernel_size, int length, int dilation, int groups);

/* Transposed Conv1d (for upsampling) */
void kernel_transposed_conv1d(float *out, const float *input, const float *weight,
                              const float *bias, int in_channels, int out_channels,
                              int kernel_size, int stride, int length, int *out_length);

/* Clamp values to [-1, 1] */
void kernel_clamp(float *x, int n, float min_val, float max_val);

/* Dot product */
float kernel_dot(const float *a, const float *b, int n);

/* Sum of squares */
float kernel_sum_sq(const float *x, int n);

/* Copy BF16 array to F32 */
void kernel_bf16_to_f32(float *out, const uint16_t *in, int n);

/* Zero buffer */
void kernel_zero(float *x, int n);

/* ========================================================================
 * Platform-specific dispatch
 * ======================================================================== */

/* Initialize kernel dispatch based on detected CPU features */
void kernel_init(void);

#endif /* QWEN_TTS_KERNELS_H */
