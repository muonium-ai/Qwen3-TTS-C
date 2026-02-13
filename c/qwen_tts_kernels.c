/*
 * qwen_tts_kernels.c - Math kernel implementations for Qwen3-TTS
 *
 * Uses BLAS (Accelerate on macOS, OpenBLAS on Linux) for matrix ops.
 * Falls back to scalar implementations when BLAS is not available.
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ======================================================================== */
/* RMSNorm                                                                   */
/* ======================================================================== */

void kernel_rms_norm(float *out, const float *x, const float *weight, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv * weight[i];
}

void kernel_rms_norm_inplace(float *x, const float *weight, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) x[i] = x[i] * inv * weight[i];
}

/* ======================================================================== */
/* LayerNorm                                                                 */
/* ======================================================================== */

void kernel_layer_norm(float *out, const float *x, const float *weight, const float *bias, int dim, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
    var /= (float)dim;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv;
        if (weight) out[i] *= weight[i];
        if (bias) out[i] += bias[i];
    }
}

/* ======================================================================== */
/* Matrix-Vector Multiply                                                    */
/* ======================================================================== */

#if defined(USE_BLAS) && !defined(__ARM_NEON) && !defined(__aarch64__)
/* Persistent scratch buffer for BF16→F32 conversion (non-NEON BLAS path) */
static float *_bf16_scratch = NULL;
static size_t _bf16_scratch_cap = 0;

static float *bf16_scratch_get(size_t n) {
    if (n > _bf16_scratch_cap) {
        free(_bf16_scratch);
        _bf16_scratch = (float *)malloc(n * sizeof(float));
        _bf16_scratch_cap = n;
    }
    return _bf16_scratch;
}
#endif

void kernel_matvec_bf16(float *out, const uint16_t *A_bf16, const float *x, int rows, int cols) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    /* Fused BF16→F32 + dot product using NEON — no intermediate buffer needed */
    for (int r = 0; r < rows; r++) {
        const uint16_t *row = A_bf16 + (size_t)r * cols;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            /* Load 8 BF16 values */
            uint16x8_t bf = vld1q_u16(row + c);
            /* Split into two halves and shift to F32 */
            uint32x4_t lo = vshll_n_u16(vget_low_u16(bf), 16);
            uint32x4_t hi = vshll_n_u16(vget_high_u16(bf), 16);
            float32x4_t f0 = vreinterpretq_f32_u32(lo);
            float32x4_t f1 = vreinterpretq_f32_u32(hi);
            /* Load 8 F32 x values */
            float32x4_t x0 = vld1q_f32(x + c);
            float32x4_t x1 = vld1q_f32(x + c + 4);
            /* Fused multiply-accumulate */
            acc0 = vfmaq_f32(acc0, f0, x0);
            acc1 = vfmaq_f32(acc1, f1, x1);
        }
        /* Reduce accumulators */
        acc0 = vaddq_f32(acc0, acc1);
        float sum = vaddvq_f32(acc0);
        /* Handle remaining elements */
        for (; c < cols; c++) {
            uint32_t bits = ((uint32_t)row[c]) << 16;
            float val;
            memcpy(&val, &bits, sizeof(float));
            sum += val * x[c];
        }
        out[r] = sum;
    }
#elif defined(USE_BLAS)
    /* Convert BF16 matrix to F32 using persistent scratch, then use BLAS sgemv */
    size_t total = (size_t)rows * cols;
    float *A_f32 = bf16_scratch_get(total);
    for (size_t i = 0; i < total; i++) {
        uint32_t bits = ((uint32_t)A_bf16[i]) << 16;
        memcpy(&A_f32[i], &bits, sizeof(float));
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, A_f32, cols, x, 1, 0.0f, out, 1);
#else
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const uint16_t *row = A_bf16 + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            sum += bf16_to_f32(row[c]) * x[c];
        }
        out[r] = sum;
    }
#endif
}

void kernel_matvec_f32(float *out, const float *A, const float *x, int rows, int cols) {
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, A, cols, x, 1, 0.0f, out, 1);
#else
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *row = A + (size_t)r * cols;
        for (int c = 0; c < cols; c++) sum += row[c] * x[c];
        out[r] = sum;
    }
#endif
}

/* ======================================================================== */
/* Matrix-Matrix Multiply                                                    */
/* ======================================================================== */

void kernel_matmul_f32(float *C, const float *A, const float *B, int M, int N, int K) {
    /* C[M,N] = A[M,K] @ B[N,K]^T  →  C = A * B^T */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                1.0f, A, K, B, K, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * B[n * K + k];
            C[m * N + n] = sum;
        }
    }
#endif
}

void kernel_matmul_bf16(float *C, const float *A, const uint16_t *B_bf16, int M, int N, int K) {
    /* C[M,N] = A[M,K] @ B[N,K]^T  where B is BF16 */
    /* Convert B to F32 first for BLAS */
#ifdef USE_BLAS
    float *B_f32 = (float *)malloc((size_t)N * K * sizeof(float));
    for (size_t i = 0; i < (size_t)N * K; i++) {
        uint32_t bits = ((uint32_t)B_bf16[i]) << 16;
        memcpy(&B_f32[i], &bits, sizeof(float));
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                1.0f, A, K, B_f32, K, 0.0f, C, N);
    free(B_f32);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * bf16_to_f32(B_bf16[n * K + k]);
            C[m * N + n] = sum;
        }
    }
#endif
}

/* ======================================================================== */
/* SwiGLU                                                                    */
/* ======================================================================== */

void kernel_swiglu_matvec_bf16(float *out, const uint16_t *gate_up_fused_bf16,
                                const float *x, int intermediate, int hidden) {
    /* gate_up_fused is [2*intermediate, hidden], first half is gate, second is up */
    float *gate = (float *)malloc(intermediate * sizeof(float));
    float *up = (float *)malloc(intermediate * sizeof(float));

    kernel_matvec_bf16(gate, gate_up_fused_bf16, x, intermediate, hidden);
    kernel_matvec_bf16(up, gate_up_fused_bf16 + (size_t)intermediate * hidden, x, intermediate, hidden);

    for (int i = 0; i < intermediate; i++) {
        /* SiLU(gate) * up */
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }

    free(gate);
    free(up);
}

/* ======================================================================== */
/* Activation functions                                                      */
/* ======================================================================== */

void kernel_silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

void kernel_gelu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

void kernel_snake_beta(float *out, const float *x, const float *alpha,
                       const float *beta, int channels, int length) {
    /* SnakeBeta: out = x + (1/beta) * sin^2(alpha * x) */
    /* alpha and beta are stored as log values (need exp) */
    const float eps = 1e-9f;
    for (int c = 0; c < channels; c++) {
        float a = expf(alpha[c]);
        float b = expf(beta[c]);
        float inv_b = 1.0f / (b + eps);
        for (int t = 0; t < length; t++) {
            int idx = c * length + t;
            float s = sinf(x[idx] * a);
            out[idx] = x[idx] + inv_b * s * s;
        }
    }
}

/* ======================================================================== */
/* Element-wise operations                                                   */
/* ======================================================================== */

void kernel_add(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void kernel_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

void kernel_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

void kernel_scale_inplace(float *x, float scale, int n) {
    for (int i = 0; i < n; i++) x[i] *= scale;
}

void kernel_zero(float *x, int n) {
    memset(x, 0, n * sizeof(float));
}

void kernel_clamp(float *x, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        if (x[i] < min_val) x[i] = min_val;
        if (x[i] > max_val) x[i] = max_val;
    }
}

float kernel_dot(const float *a, const float *b, int n) {
#ifdef USE_BLAS
    return cblas_sdot(n, a, 1, b, 1);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

float kernel_sum_sq(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sum;
}

void kernel_bf16_to_f32(float *out, const uint16_t *in, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)in[i]) << 16;
        memcpy(&out[i], &bits, sizeof(float));
    }
}

/* ======================================================================== */
/* Softmax                                                                   */
/* ======================================================================== */

void kernel_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

/* ======================================================================== */
/* Sampling                                                                  */
/* ======================================================================== */

static float rand_uniform(float *state) {
    /* Simple xorshift-based RNG */
    uint32_t s;
    memcpy(&s, state, sizeof(uint32_t));
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    memcpy(state, &s, sizeof(uint32_t));
    return (float)(s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

void kernel_apply_repetition_penalty(float *logits, const int *token_ids,
                                     int n_tokens, int vocab_size, float penalty) {
    if (penalty == 1.0f) return;
    for (int i = 0; i < n_tokens; i++) {
        int t = token_ids[i];
        if (t >= 0 && t < vocab_size) {
            if (logits[t] > 0) logits[t] /= penalty;
            else logits[t] *= penalty;
        }
    }
}

int kernel_sample_top_k(const float *logits, int vocab_size, int top_k,
                        float top_p, float temperature, float *rng_state) {
    /* Apply temperature */
    float *probs = (float *)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) probs[i] = logits[i] / temperature;

    /* Softmax */
    kernel_softmax(probs, vocab_size);

    /* Top-K: find top_k largest and zero out the rest */
    if (top_k > 0 && top_k < vocab_size) {
        /* Find the kth-largest probability using partial sort */
        float *sorted = (float *)malloc(vocab_size * sizeof(float));
        memcpy(sorted, probs, vocab_size * sizeof(float));
        /* Simple selection of kth element (partial qsort) */
        for (int i = 0; i < top_k; i++) {
            int max_idx = i;
            for (int j = i + 1; j < vocab_size; j++) {
                if (sorted[j] > sorted[max_idx]) max_idx = j;
            }
            float tmp = sorted[i]; sorted[i] = sorted[max_idx]; sorted[max_idx] = tmp;
        }
        float threshold = sorted[top_k - 1];
        free(sorted);
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
    }

    /* Top-P (nucleus): sort by probability, accumulate until sum >= top_p */
    if (top_p < 1.0f) {
        /* Create index + prob pairs and sort descending */
        int *indices = (int *)malloc(vocab_size * sizeof(int));
        for (int i = 0; i < vocab_size; i++) indices[i] = i;
        /* Simple insertion sort for top elements */
        for (int i = 1; i < vocab_size; i++) {
            int idx = indices[i];
            float val = probs[idx];
            int j = i - 1;
            while (j >= 0 && probs[indices[j]] < val) {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = idx;
        }
        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[indices[i]];
            if (cumsum >= top_p) { cutoff = i + 1; break; }
        }
        for (int i = cutoff; i < vocab_size; i++) probs[indices[i]] = 0.0f;
        free(indices);
    }

    /* Renormalize */
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) sum += probs[i];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < vocab_size; i++) probs[i] *= inv;
    }

    /* Sample from the distribution */
    float r = rand_uniform(rng_state);
    float cumsum = 0.0f;
    int sampled = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= r) { sampled = i; break; }
    }

    free(probs);
    return sampled;
}

/* ======================================================================== */
/* RoPE (Rotary Position Embedding)                                          */
/* ======================================================================== */

void kernel_rope_apply(float *q, float *k, const float *cos, const float *sin,
                       int num_heads, int head_dim) {
    /* Standard RoPE: rotate pairs of dimensions
     * q/k shape: [num_heads * head_dim] (flattened)
     * cos/sin shape: [head_dim] (shared across heads)
     */
    int half = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * cos[i] - q1 * sin[i];
            qh[i + half] = q1 * cos[i] + q0 * sin[i];
        }
        if (k) {
            float *kh = k + h * head_dim;
            for (int i = 0; i < half; i++) {
                float k0 = kh[i], k1 = kh[i + half];
                kh[i]        = k0 * cos[i] - k1 * sin[i];
                kh[i + half] = k1 * cos[i] + k0 * sin[i];
            }
        }
    }
}

void kernel_mrope_apply(float *q, float *k, const float *cos, const float *sin,
                        int num_heads, int head_dim, const int *mrope_section) {
    /* M-RoPE: multimodal rotary position embedding
     * cos/sin shape: [3, head_dim] - 3 position streams  
     * mrope_section: [s0, s1, s2] - how many dims per section (doubled for rotate_half)
     *
     * The Python code does:
     *   mrope_section = mrope_section * 2  (doubled)
     *   cos = cat([m[i%3] for i, m in enumerate(cos.split(mrope_section, dim=-1))])
     *   sin = cat([m[i%3] for i, m in enumerate(sin.split(mrope_section, dim=-1))])
     *
     * So for sections [s0, s1, s2], doubled to [s0, s1, s2, s0, s1, s2]:
     *   dims [0..s0-1]:              use cos/sin from stream 0
     *   dims [s0..s0+s1-1]:          use cos/sin from stream 1
     *   dims [s0+s1..s0+s1+s2-1]:    use cos/sin from stream 2
     *   dims [s0+s1+s2..]:           repeat pattern (for the rotate_half second half)
     *
     * For TTS with all-text, all 3 position streams are the same,
     * so this reduces to standard RoPE. But we implement the full version.
     */

    /* mrope_section doubled */
    int sec[6];
    sec[0] = mrope_section[0]; sec[1] = mrope_section[1]; sec[2] = mrope_section[2];
    sec[3] = mrope_section[0]; sec[4] = mrope_section[1]; sec[5] = mrope_section[2];

    /* Build the interleaved cos/sin arrays for this position
     * cos_interleaved[d] picks from the appropriate stream based on which section d falls in
     * cos input: [3, head_dim/2] (half dim since the full cos is cat(freqs, freqs))
     * After interleaving: [head_dim]
     */
    float cos_merged[512], sin_merged[512]; /* max head_dim */
    int d = 0;
    for (int chunk = 0; chunk < 6; chunk++) {
        int stream = chunk % 3;
        /* In the Python code, cos shape is [3, seq, head_dim], and after cat of freqs,freqs it's doubled.
         * The split happens on the last dim of size head_dim (already doubled).
         * cos[stream] covers the full head_dim.
         * We split head_dim into sections sec[0..5] and pick stream accordingly.
         */
        for (int i = 0; i < sec[chunk] && d < head_dim; i++, d++) {
            cos_merged[d] = cos[stream * head_dim + d];
            sin_merged[d] = sin[stream * head_dim + d];
        }
    }

    /* Apply standard rotate_half RoPE with the merged cos/sin */
    int half = head_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * cos_merged[i] - q1 * sin_merged[i];
            qh[i + half] = q1 * cos_merged[i + half] + q0 * sin_merged[i + half];
        }
        if (k) {
            float *kh = k + h * head_dim;
            for (int i = 0; i < half; i++) {
                float k0 = kh[i], k1 = kh[i + half];
                kh[i]        = k0 * cos_merged[i] - k1 * sin_merged[i];
                kh[i + half] = k1 * cos_merged[i + half] + k0 * sin_merged[i + half];
            }
        }
    }
}

/* ======================================================================== */
/* Convolution operations                                                    */
/* ======================================================================== */

void kernel_causal_conv1d(float *out, const float *input, const float *weight,
                          const float *bias, int in_channels, int out_channels,
                          int kernel_size, int length, int dilation, int groups) {
    /* Causal Conv1d: left-padded by (kernel_size-1)*dilation
     * Input: [in_channels, length]
     * Weight: [out_channels, in_channels/groups, kernel_size]
     * Output: [out_channels, length]
     */
    int eff_kernel = (kernel_size - 1) * dilation + 1;
    int pad = eff_kernel - 1; /* causal: all padding on left */
    int ch_per_group = in_channels / groups;
    int out_per_group = out_channels / groups;

    /* Fast path for pointwise conv: k=1, dilation=1 (very common in vocoder). */
    if (kernel_size == 1 && dilation == 1) {
        if (groups == in_channels && in_channels == out_channels) {
            /* Depthwise pointwise: one scalar multiply per sample. */
            for (int c = 0; c < in_channels; c++) {
                float w = weight[c];
                float b = bias ? bias[c] : 0.0f;
                const float *in_ch = input + c * length;
                float *out_ch = out + c * length;
                for (int t = 0; t < length; t++) {
                    out_ch[t] = in_ch[t] * w + b;
                }
            }
            return;
        }

        for (int oc = 0; oc < out_channels; oc++) {
            int g = oc / out_per_group;
            const float *w_row = weight + (size_t)oc * ch_per_group;
            for (int t = 0; t < length; t++) {
                float sum = bias ? bias[oc] : 0.0f;
                int ic_base = g * ch_per_group;
                for (int ic = 0; ic < ch_per_group; ic++) {
                    sum += w_row[ic] * input[(ic_base + ic) * length + t];
                }
                out[oc * length + t] = sum;
            }
        }
        return;
    }

    /*
     * Fast path for common causal conv (dilation=1):
     * avoid inner-boundary checks for the steady-state region.
     */
    if (dilation == 1) {
        for (int oc = 0; oc < out_channels; oc++) {
            int g = oc / out_per_group;
            int ic_base = g * ch_per_group;
            for (int t = 0; t < length; t++) {
                float sum = bias ? bias[oc] : 0.0f;
                int k_start = (pad > t) ? (pad - t) : 0;
                int in_t0 = t - pad + k_start;
                for (int ic = 0; ic < ch_per_group; ic++) {
                    const float *w = weight + ((size_t)oc * ch_per_group + ic) * kernel_size;
                    const float *in = input + (size_t)(ic_base + ic) * length + in_t0;
                    for (int k = k_start; k < kernel_size; k++) {
                        sum += w[k] * (*in++);
                    }
                }
                out[oc * length + t] = sum;
            }
        }
        return;
    }

    for (int oc = 0; oc < out_channels; oc++) {
        int g = oc / out_per_group;
        for (int t = 0; t < length; t++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < ch_per_group; ic++) {
                int abs_ic = g * ch_per_group + ic;
                for (int k = 0; k < kernel_size; k++) {
                    int input_t = t - pad + k * dilation;
                    if (input_t >= 0 && input_t < length) {
                        int widx = (oc * ch_per_group + ic) * kernel_size + k;
                        sum += weight[widx] * input[abs_ic * length + input_t];
                    }
                }
            }
            out[oc * length + t] = sum;
        }
    }
}

void kernel_transposed_conv1d(float *out, const float *input, const float *weight,
                              const float *bias, int in_channels, int out_channels,
                              int kernel_size, int stride, int length, int *out_length) {
    /* Transposed Conv1d (for upsampling)
     * Input: [in_channels, length]
     * Weight: [in_channels, out_channels, kernel_size]
     * Output: [out_channels, out_len]
     *
     * out_len = (length - 1) * stride + kernel_size
     * Then trim right_pad = kernel_size - stride from the right (causal convention)
     */
    int raw_out_len = (length - 1) * stride + kernel_size;
    int right_pad = kernel_size - stride;
    int final_len = raw_out_len - right_pad;
    if (final_len < 0) final_len = 0;

    /*
     * Cache-friendly loop order:
     *   iterate output channel first so writes stay contiguous in out[oc, :].
     * Weight layout is [in_channels, out_channels, kernel_size].
     */
    for (int oc = 0; oc < out_channels; oc++) {
        float *out_ch = out + (size_t)oc * final_len;
        float b = bias ? bias[oc] : 0.0f;
        for (int t = 0; t < final_len; t++) out_ch[t] = b;

        for (int ic = 0; ic < in_channels; ic++) {
            const float *in_ch = input + (size_t)ic * length;
            const float *w = weight + (size_t)ic * out_channels * kernel_size + (size_t)oc * kernel_size;
            for (int t = 0; t < length; t++) {
                float val = in_ch[t];
                int base = t * stride;
                for (int k = 0; k < kernel_size; k++) {
                    int ot = base + k;
                    if (ot < final_len) out_ch[ot] += val * w[k];
                }
            }
        }
    }

    if (out_length) *out_length = final_len;
}

/* ======================================================================== */
/* Platform dispatch (no-op for now)                                         */
/* ======================================================================== */

void kernel_init(void) {
    /* In the future, detect NEON/AVX and set function pointers */
}
