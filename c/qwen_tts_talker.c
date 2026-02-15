/*
 * qwen_tts_talker.c - Talker Transformer Forward Pass
 *
 * Implements:
 *   - Prefill (multiple tokens, batch matmul)
 *   - Single-token generation (matvec + KV cache)
 *   - Sub-talker code predictor (generates remaining 31 code groups)
 *
 * Architecture:
 *   - 20 layers, hidden=1024, intermediate=2048
 *   - 16 Q heads, 2 KV heads (GQA 8:1), head_dim=64
 *   - QK-Norm (RMSNorm per head on Q/K before RoPE)
 *   - M-RoPE (3D position encoding, all same for text)
 *   - SwiGLU MLP
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern int qwen_tts_verbose;

static inline float st_dot(const float *a, const float *b, int n) {
#ifdef USE_BLAS
    return cblas_sdot(n, a, 1, b, 1);
#else
    return kernel_dot(a, b, n);
#endif
}

static inline void st_axpy(int n, float alpha, const float *x, float *y) {
#ifdef USE_BLAS
    cblas_saxpy(n, alpha, x, 1, y, 1);
#else
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
#endif
}

static int ensure_talker_prefill_buffers(qwen_tts_ctx_t *ctx, int seq_len) {
    qwen_tts_config_t *cfg = &ctx->config;
    int hidden = cfg->talker_hidden;
    int num_heads = cfg->talker_heads;
    int head_dim = cfg->talker_head_dim;
    int kv_dim = cfg->talker_kv_heads * head_dim;
    int intermediate = cfg->talker_intermediate;

    if (ctx->tk_pref_cap >= seq_len) return 0;

#define REALLOC_PREF_FIELD(field, count) do { \
    float *tmp = (float *)realloc(ctx->field, (size_t)(count) * sizeof(float)); \
    if (!tmp) return -1; \
    ctx->field = tmp; \
} while (0)

    REALLOC_PREF_FIELD(tk_pref_x, (size_t)seq_len * hidden);
    REALLOC_PREF_FIELD(tk_pref_x_norm, (size_t)seq_len * hidden);
    REALLOC_PREF_FIELD(tk_pref_q, (size_t)seq_len * num_heads * head_dim);
    REALLOC_PREF_FIELD(tk_pref_k, (size_t)seq_len * kv_dim);
    REALLOC_PREF_FIELD(tk_pref_v, (size_t)seq_len * kv_dim);
    REALLOC_PREF_FIELD(tk_pref_attn_out, (size_t)seq_len * num_heads * head_dim);
    REALLOC_PREF_FIELD(tk_pref_gate, (size_t)seq_len * intermediate);
    REALLOC_PREF_FIELD(tk_pref_gate_up, (size_t)seq_len * intermediate);

#undef REALLOC_PREF_FIELD

    ctx->tk_pref_cap = seq_len;
    return 0;
}

/* ========================================================================
 * RoPE cache computation
 * ======================================================================== */

static void compute_rope_cache(float *cos_cache, float *sin_cache,
                                int max_pos, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int pos = 0; pos < max_pos; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            /* Store as [pos, head_dim] where first half is cos, second half is cat(cos,cos) */
            cos_cache[pos * head_dim + i] = c;
            cos_cache[pos * head_dim + i + half] = c;
            sin_cache[pos * head_dim + i] = s;
            sin_cache[pos * head_dim + i + half] = s;
        }
    }
}

/* Compute M-RoPE cos/sin for a given position (3 streams, all same for text)
 * Output: cos_out[3 * head_dim], sin_out[3 * head_dim]
 */
static void compute_mrope_pos(float *cos_out, float *sin_out,
                               int pos, int head_dim, float theta) {
    int half = head_dim / 2;
    /* For text-only TTS, all 3 position streams use the same position */
    for (int stream = 0; stream < 3; stream++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            cos_out[stream * head_dim + i] = c;
            cos_out[stream * head_dim + i + half] = c;
            sin_out[stream * head_dim + i] = s;
            sin_out[stream * head_dim + i + half] = s;
        }
    }
}

/* ========================================================================
 * Talker attention - single token
 * ======================================================================== */

static void talker_attention_single(
    qwen_tts_ctx_t *ctx,
    int layer_idx,
    float *x,           /* [hidden], modified in-place */
    float *x_norm,      /* scratch [hidden] */
    float *q_buf,       /* scratch [num_heads * head_dim] */
    float *k_buf,       /* scratch [kv_heads * head_dim] */
    float *v_buf,       /* scratch [kv_heads * head_dim] */
    float *attn_out,    /* scratch [num_heads * head_dim] */
    const float *cos,   /* [3 * head_dim] for M-RoPE */
    const float *sin,
    int pos             /* current position in sequence */
) {
    qwen_tts_config_t *cfg = &ctx->config;
    qwen_tts_talker_layer_t *layer = &ctx->talker.layers[layer_idx];
    int hidden = cfg->talker_hidden;
    int num_heads = cfg->talker_heads;
    int kv_heads = cfg->talker_kv_heads;
    int head_dim = cfg->talker_head_dim;
    int kv_dim = kv_heads * head_dim;
    int groups_per_head = num_heads / kv_heads;
    float eps = cfg->talker_rms_norm_eps;

    /* 1. Input LayerNorm */
    kernel_rms_norm(x_norm, x, layer->input_norm, hidden, eps);

    /* 2. Q, K, V projections */
    kernel_matvec_bf16(q_buf, layer->wq_bf16, x_norm, num_heads * head_dim, hidden);
    kernel_matvec_bf16(k_buf, layer->wk_bf16, x_norm, kv_dim, hidden);
    kernel_matvec_bf16(v_buf, layer->wv_bf16, x_norm, kv_dim, hidden);

    /* 3. QK-Norm: per-head RMSNorm on Q and K */
    for (int h = 0; h < num_heads; h++) {
        kernel_rms_norm_inplace(q_buf + h * head_dim, layer->q_norm_weight, head_dim, eps);
    }
    for (int h = 0; h < kv_heads; h++) {
        kernel_rms_norm_inplace(k_buf + h * head_dim, layer->k_norm_weight, head_dim, eps);
    }

    /* 4. M-RoPE — apply with correct separate head counts for Q and K */
    {
        int sec[6];
        sec[0] = cfg->mrope_section[0]; sec[1] = cfg->mrope_section[1]; sec[2] = cfg->mrope_section[2];
        sec[3] = cfg->mrope_section[0]; sec[4] = cfg->mrope_section[1]; sec[5] = cfg->mrope_section[2];
        float cos_m[512], sin_m[512];
        int d = 0;
        for (int chunk = 0; chunk < 6; chunk++) {
            int stream = chunk % 3;
            for (int i = 0; i < sec[chunk] && d < head_dim; i++, d++) {
                cos_m[d] = cos[stream * head_dim + d];
                sin_m[d] = sin[stream * head_dim + d];
            }
        }
        int half = head_dim / 2;
        for (int h = 0; h < num_heads; h++) {
            float *qh = q_buf + h * head_dim;
            for (int i = 0; i < half; i++) {
                float q0 = qh[i], q1 = qh[i + half];
                qh[i]        = q0 * cos_m[i] - q1 * sin_m[i];
                qh[i + half] = q1 * cos_m[i + half] + q0 * sin_m[i + half];
            }
        }
        for (int h = 0; h < kv_heads; h++) {
            float *kh = k_buf + h * head_dim;
            for (int i = 0; i < half; i++) {
                float k0 = kh[i], k1 = kh[i + half];
                kh[i]        = k0 * cos_m[i] - k1 * sin_m[i];
                kh[i + half] = k1 * cos_m[i + half] + k0 * sin_m[i + half];
            }
        }
    }

    /* 5. Store K, V into KV cache */
    size_t kv_stride = (size_t)ctx->talker_kv_max * kv_dim;
    float *cache_k = ctx->talker_kv_k + (size_t)layer_idx * kv_stride + (size_t)pos * kv_dim;
    float *cache_v = ctx->talker_kv_v + (size_t)layer_idx * kv_stride + (size_t)pos * kv_dim;
    memcpy(cache_k, k_buf, kv_dim * sizeof(float));
    memcpy(cache_v, v_buf, kv_dim * sizeof(float));

    /* 6. Attention: Q @ K^T, scaled, causal (single query pos) */
    int seq_len = pos + 1;  /* total KV length */
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / groups_per_head;
        float *qh = q_buf + h * head_dim;

        /* Compute attention scores for this head */
        float *scores = ctx->tk_scores;

        for (int t = 0; t < seq_len; t++) {
            float *kt = ctx->talker_kv_k + (size_t)layer_idx * kv_stride + (size_t)t * kv_dim + kv_h * head_dim;
            float score = st_dot(qh, kt, head_dim) * scale;
            scores[t] = score;
        }

        /* Softmax */
        kernel_softmax(scores, seq_len);

        /* Weighted sum of V */
        float *oh = attn_out + h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            float *vt = ctx->talker_kv_v + (size_t)layer_idx * kv_stride + (size_t)t * kv_dim + kv_h * head_dim;
            st_axpy(head_dim, scores[t], vt, oh);
        }
    }

    /* 7. Output projection */
    float *proj_out = x_norm; /* reuse buffer */
    kernel_matvec_bf16(proj_out, layer->wo_bf16, attn_out, hidden, num_heads * head_dim);

    /* 8. Residual add */
    kernel_add_inplace(x, proj_out, hidden);

    /* 9. Post-attention norm + SwiGLU MLP */
    kernel_rms_norm(x_norm, x, layer->post_attn_norm, hidden, eps);

    float *gate_buf = ctx->tk_gate;

    /* Fused SwiGLU: gate_out = silu(gate @ x) * (up @ x) in one pass */
    kernel_swiglu_matvec_bf16(gate_buf, layer->gate_up_fused_bf16, x_norm,
                              cfg->talker_intermediate, hidden);

    /* down projection */
    kernel_matvec_bf16(proj_out, layer->down_bf16, gate_buf, hidden, cfg->talker_intermediate);

    /* Residual add */
    kernel_add_inplace(x, proj_out, hidden);
}

/* ========================================================================
 * Talker prefill (batch processing)
 * ======================================================================== */

void qwen_tts_talker_prefill(qwen_tts_ctx_t *ctx, const float *input_embeds, int seq_len) {
    qwen_tts_config_t *cfg = &ctx->config;
    int hidden = cfg->talker_hidden;
    int num_heads = cfg->talker_heads;
    int kv_heads = cfg->talker_kv_heads;
    int head_dim = cfg->talker_head_dim;
    int kv_dim = kv_heads * head_dim;
    int groups_per_head = num_heads / kv_heads;
    float eps = cfg->talker_rms_norm_eps;

    /* Ensure KV cache is large enough */
    if (ctx->talker_kv_max < seq_len + 4096) {
        int new_max = seq_len + 4096;
        size_t kv_size = (size_t)cfg->talker_layers * new_max * kv_dim * sizeof(float);
        ctx->talker_kv_k = (float *)realloc(ctx->talker_kv_k, kv_size);
        ctx->talker_kv_v = (float *)realloc(ctx->talker_kv_v, kv_size);
        ctx->talker_kv_max = new_max;
    }

    if (ensure_talker_prefill_buffers(ctx, seq_len) != 0) {
        fprintf(stderr, "Error: failed to allocate talker prefill buffers\n");
        return;
    }
    float *x = ctx->tk_pref_x;
    float *x_norm = ctx->tk_pref_x_norm;
    float *q_all = ctx->tk_pref_q;
    float *k_all = ctx->tk_pref_k;
    float *v_all = ctx->tk_pref_v;
    float *attn_out = ctx->tk_pref_attn_out;
    float *gate_all = ctx->tk_pref_gate;
    float *up_all = ctx->tk_pref_gate_up;
    float *scores = (float *)malloc((size_t)seq_len * sizeof(float));
    if (!scores) {
        fprintf(stderr, "Error: failed to allocate talker score buffer\n");
        return;
    }

    /* Copy input embeddings */
    memcpy(x, input_embeds, (size_t)seq_len * hidden * sizeof(float));

    /* Compute/cached M-RoPE cos/sin for all positions */
    if (!ctx->talker_rope_cos_cache || !ctx->talker_rope_sin_cache || ctx->talker_rope_cache_cap < seq_len) {
        size_t rope_size = (size_t)seq_len * 3 * head_dim;
        float *cos_new = (float *)malloc(rope_size * sizeof(float));
        float *sin_new = (float *)malloc(rope_size * sizeof(float));
        if (!cos_new || !sin_new) {
            free(cos_new);
            free(sin_new);
            free(scores);
            fprintf(stderr, "Error: failed to allocate talker M-RoPE cache\n");
            return;
        }
        free(ctx->talker_rope_cos_cache);
        free(ctx->talker_rope_sin_cache);
        ctx->talker_rope_cos_cache = cos_new;
        ctx->talker_rope_sin_cache = sin_new;
        for (int p = 0; p < seq_len; p++) {
            compute_mrope_pos(ctx->talker_rope_cos_cache + (size_t)p * 3 * head_dim,
                              ctx->talker_rope_sin_cache + (size_t)p * 3 * head_dim,
                              p, head_dim, cfg->talker_rope_theta);
        }
        ctx->talker_rope_cache_cap = seq_len;
    }
    float *cos_all = ctx->talker_rope_cos_cache;
    float *sin_all = ctx->talker_rope_sin_cache;

    for (int layer = 0; layer < cfg->talker_layers; layer++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[layer];
        size_t kv_stride = (size_t)ctx->talker_kv_max * kv_dim;

        /* 1. Input LayerNorm (per token) */
        for (int t = 0; t < seq_len; t++) {
            kernel_rms_norm(x_norm + t * hidden, x + t * hidden, l->input_norm, hidden, eps);
        }

        /* 2. Q, K, V projections (batch matmul) */
        kernel_matmul_bf16(q_all, x_norm, l->wq_bf16, seq_len, num_heads * head_dim, hidden);
        kernel_matmul_bf16(k_all, x_norm, l->wk_bf16, seq_len, kv_dim, hidden);
        kernel_matmul_bf16(v_all, x_norm, l->wv_bf16, seq_len, kv_dim, hidden);

        /* 3. QK-Norm per head */
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < num_heads; h++) {
                kernel_rms_norm_inplace(q_all + t * num_heads * head_dim + h * head_dim,
                                        l->q_norm_weight, head_dim, eps);
            }
            for (int h = 0; h < kv_heads; h++) {
                kernel_rms_norm_inplace(k_all + t * kv_dim + h * head_dim,
                                        l->k_norm_weight, head_dim, eps);
            }
        }

        /* 4. M-RoPE */
        {
            /* Build merged cos/sin per position */
            int sec[6];
            sec[0] = cfg->mrope_section[0]; sec[1] = cfg->mrope_section[1]; sec[2] = cfg->mrope_section[2];
            sec[3] = cfg->mrope_section[0]; sec[4] = cfg->mrope_section[1]; sec[5] = cfg->mrope_section[2];

            for (int t = 0; t < seq_len; t++) {
                float cos_m[512], sin_m[512];
                const float *cos_pos = cos_all + t * 3 * head_dim;
                const float *sin_pos = sin_all + t * 3 * head_dim;
                int d = 0;
                for (int chunk = 0; chunk < 6; chunk++) {
                    int stream = chunk % 3;
                    for (int i = 0; i < sec[chunk] && d < head_dim; i++, d++) {
                        cos_m[d] = cos_pos[stream * head_dim + d];
                        sin_m[d] = sin_pos[stream * head_dim + d];
                    }
                }
                int half = head_dim / 2;
                for (int h = 0; h < num_heads; h++) {
                    float *qh = q_all + t * num_heads * head_dim + h * head_dim;
                    for (int i = 0; i < half; i++) {
                        float q0 = qh[i], q1 = qh[i + half];
                        qh[i]        = q0 * cos_m[i] - q1 * sin_m[i];
                        qh[i + half] = q1 * cos_m[i + half] + q0 * sin_m[i + half];
                    }
                }
                for (int h = 0; h < kv_heads; h++) {
                    float *kh = k_all + t * kv_dim + h * head_dim;
                    for (int i = 0; i < half; i++) {
                        float k0 = kh[i], k1 = kh[i + half];
                        kh[i]        = k0 * cos_m[i] - k1 * sin_m[i];
                        kh[i + half] = k1 * cos_m[i + half] + k0 * sin_m[i + half];
                    }
                }
            }
        }

        /* 5. Store K, V in cache */
        for (int t = 0; t < seq_len; t++) {
            memcpy(ctx->talker_kv_k + layer * kv_stride + t * kv_dim,
                   k_all + t * kv_dim, kv_dim * sizeof(float));
            memcpy(ctx->talker_kv_v + layer * kv_stride + t * kv_dim,
                   v_all + t * kv_dim, kv_dim * sizeof(float));
        }

        /* 6. Attention: [seq, heads, head_dim] @ [seq, kv_heads, head_dim]^T
         * Use per-head attention computation */
        float scale = 1.0f / sqrtf((float)head_dim);
        memset(attn_out, 0, (size_t)seq_len * num_heads * head_dim * sizeof(float));

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / groups_per_head;

            /* Compute attention scores: [seq, seq] for this head */
            for (int qi = 0; qi < seq_len; qi++) {
                float *qh = q_all + qi * num_heads * head_dim + h * head_dim;

                for (int ki = 0; ki <= qi; ki++) {  /* causal */
                    float *kh = k_all + ki * kv_dim + kv_h * head_dim;
                    float score = st_dot(qh, kh, head_dim) * scale;
                    scores[ki] = score;
                }

                /* Softmax over causal window */
                kernel_softmax(scores, qi + 1);

                /* Weighted sum of V */
                float *oh = attn_out + qi * num_heads * head_dim + h * head_dim;
                for (int ki = 0; ki <= qi; ki++) {
                    float w = scores[ki];
                    float *vh = v_all + ki * kv_dim + kv_h * head_dim;
                    st_axpy(head_dim, w, vh, oh);
                }
            }
        }

        /* 7. Output projection (batch matmul) */
        kernel_matmul_bf16(x_norm, attn_out, l->wo_bf16, seq_len, hidden, num_heads * head_dim);

        /* 8. Residual */
        for (int t = 0; t < seq_len; t++)
            kernel_add_inplace(x + t * hidden, x_norm + t * hidden, hidden);

        /* 9. Post-attention norm + SwiGLU MLP */
        for (int t = 0; t < seq_len; t++)
            kernel_rms_norm(x_norm + t * hidden, x + t * hidden, l->post_attn_norm, hidden, eps);

        /* Gate + Up projections (batch) */
        kernel_matmul_bf16(gate_all, x_norm, l->gate_bf16, seq_len, cfg->talker_intermediate, hidden);
        kernel_matmul_bf16(up_all, x_norm, l->up_bf16, seq_len, cfg->talker_intermediate, hidden);

        /* SiLU(gate) * up */
        for (int t = 0; t < seq_len; t++) {
            float *g = gate_all + t * cfg->talker_intermediate;
            float *u = up_all + t * cfg->talker_intermediate;
            kernel_silu_inplace(g, cfg->talker_intermediate);
            kernel_mul_inplace(g, u, cfg->talker_intermediate);
        }

        /* Down projection */
        kernel_matmul_bf16(x_norm, gate_all, l->down_bf16, seq_len, hidden, cfg->talker_intermediate);

        /* Residual */
        for (int t = 0; t < seq_len; t++)
            kernel_add_inplace(x + t * hidden, x_norm + t * hidden, hidden);

    }

    /* Final norm */
    for (int t = 0; t < seq_len; t++)
        kernel_rms_norm_inplace(x + t * hidden, ctx->talker.norm, hidden, eps);

    /* Store the normed hidden states for later use */
    /* We keep the last token's hidden state for generation */
    if (!ctx->tk_x) ctx->tk_x = (float *)malloc(hidden * sizeof(float));
    memcpy(ctx->tk_x, x + (seq_len - 1) * hidden, hidden * sizeof(float));

    ctx->talker_kv_len = seq_len;

    free(scores);

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "Talker prefill complete: %d tokens\n", seq_len);
    }
}

/* ========================================================================
 * Talker single-token forward pass
 * ======================================================================== */

void qwen_tts_talker_forward(qwen_tts_ctx_t *ctx, const float *input_embed, float *logits) {
    qwen_tts_config_t *cfg = &ctx->config;
    int hidden = cfg->talker_hidden;
    int head_dim = cfg->talker_head_dim;
    int num_heads = cfg->talker_heads;
    int kv_heads = cfg->talker_kv_heads;
    int kv_dim = kv_heads * head_dim;
    int pos = ctx->talker_kv_len;
    float eps = cfg->talker_rms_norm_eps;

    /* Ensure KV cache space */
    if (pos >= ctx->talker_kv_max) {
        int new_max = ctx->talker_kv_max + 2048;
        size_t kv_size = (size_t)cfg->talker_layers * new_max * kv_dim * sizeof(float);
        ctx->talker_kv_k = (float *)realloc(ctx->talker_kv_k, kv_size);
        ctx->talker_kv_v = (float *)realloc(ctx->talker_kv_v, kv_size);
        ctx->tk_scores = (float *)realloc(ctx->tk_scores, new_max * sizeof(float));
        ctx->talker_kv_max = new_max;
    }

    /* Allocate scratch buffers if needed */
    if (!ctx->tk_x) ctx->tk_x = (float *)malloc(hidden * sizeof(float));
    if (!ctx->tk_x_norm) ctx->tk_x_norm = (float *)malloc(hidden * sizeof(float));
    if (!ctx->tk_q) ctx->tk_q = (float *)malloc(num_heads * head_dim * sizeof(float));
    if (!ctx->tk_k) ctx->tk_k = (float *)malloc(kv_dim * sizeof(float));
    if (!ctx->tk_v) ctx->tk_v = (float *)malloc(kv_dim * sizeof(float));
    if (!ctx->tk_attn_out) ctx->tk_attn_out = (float *)malloc(num_heads * head_dim * sizeof(float));
    if (!ctx->tk_gate) ctx->tk_gate = (float *)malloc(cfg->talker_intermediate * sizeof(float));
    if (!ctx->tk_up) ctx->tk_up = (float *)malloc(cfg->talker_intermediate * sizeof(float));
    /* Scores buffer is reallocated when KV cache grows */
    if (!ctx->tk_scores) {
        ctx->tk_scores = (float *)malloc(ctx->talker_kv_max * sizeof(float));
    }

    float *x = ctx->tk_x;
    memcpy(x, input_embed, hidden * sizeof(float));

    /* Compute M-RoPE cos/sin for this position */
    float cos_mrope[3 * 512], sin_mrope[3 * 512];
    compute_mrope_pos(cos_mrope, sin_mrope, pos, head_dim, cfg->talker_rope_theta);

    /* Process each layer */
    for (int layer = 0; layer < cfg->talker_layers; layer++) {
        talker_attention_single(ctx, layer, x, ctx->tk_x_norm,
                                ctx->tk_q, ctx->tk_k, ctx->tk_v,
                                ctx->tk_attn_out, cos_mrope, sin_mrope, pos);
    }

    /* Final norm */
    kernel_rms_norm_inplace(x, ctx->talker.norm, hidden, eps);

    /* Codec head: logits = x @ codec_head^T */
    kernel_matvec_bf16(logits, ctx->talker.codec_head_bf16, x, cfg->talker_vocab_size, hidden);

    ctx->talker_kv_len = pos + 1;
}

/* ========================================================================
 * Sub-talker: generate remaining 31 code groups
 * ======================================================================== */

void qwen_tts_subtalker_generate(
    qwen_tts_ctx_t *ctx,
    const float *talker_hidden,
    int first_code,
    int *out_codes
) {
    qwen_tts_config_t *cfg = &ctx->config;
    int st_hidden = cfg->subtalker_hidden;
    int st_heads = cfg->subtalker_heads;
    int st_kv_heads = cfg->subtalker_kv_heads;
    int st_head_dim = cfg->subtalker_head_dim;
    int st_kv_dim = st_kv_heads * st_head_dim;
    int st_intermediate = cfg->subtalker_intermediate;
    int st_layers = cfg->subtalker_layers;
    int st_vocab = cfg->subtalker_vocab_size;
    int groups_per_head = st_heads / st_kv_heads;
    float eps = cfg->talker_rms_norm_eps;
    int num_groups = cfg->num_code_groups;
    int talker_hidden_dim = cfg->talker_hidden;

    out_codes[0] = first_code;

    /* Allocate KV cache for sub-talker (small, num_groups+1 positions) */
    int max_seq = num_groups + 2;
    if (!ctx->subtalker_kv_k || ctx->subtalker_kv_max < max_seq) {
        size_t kv_size = (size_t)st_layers * max_seq * st_kv_dim * sizeof(float);
        ctx->subtalker_kv_k = (float *)realloc(ctx->subtalker_kv_k, kv_size);
        ctx->subtalker_kv_v = (float *)realloc(ctx->subtalker_kv_v, kv_size);
        ctx->subtalker_kv_max = max_seq;
    }
    ctx->subtalker_kv_len = 0;

    /* Persistent scratch buffers */
    if (!ctx->st_x) ctx->st_x = (float *)malloc(st_hidden * sizeof(float));
    if (!ctx->st_x_norm) ctx->st_x_norm = (float *)malloc(st_hidden * sizeof(float));
    if (!ctx->st_q) ctx->st_q = (float *)malloc(st_heads * st_head_dim * sizeof(float));
    if (!ctx->st_k) ctx->st_k = (float *)malloc(st_kv_dim * sizeof(float));
    if (!ctx->st_v) ctx->st_v = (float *)malloc(st_kv_dim * sizeof(float));
    if (!ctx->st_attn_out) ctx->st_attn_out = (float *)malloc(st_heads * st_head_dim * sizeof(float));
    if (!ctx->st_logits) ctx->st_logits = (float *)malloc(st_vocab * sizeof(float));
    if (!ctx->st_gate) ctx->st_gate = (float *)malloc(st_intermediate * sizeof(float));
    if (!ctx->st_up) ctx->st_up = (float *)malloc(st_intermediate * sizeof(float));
    if (!ctx->st_proj_hidden) ctx->st_proj_hidden = (float *)malloc(st_hidden * sizeof(float));
    if (!ctx->st_embed || ctx->st_embed_cap < talker_hidden_dim) {
        ctx->st_embed = (float *)realloc(ctx->st_embed, talker_hidden_dim * sizeof(float));
        ctx->st_embed_cap = talker_hidden_dim;
    }
    if (!ctx->st_scores || ctx->st_scores_cap < max_seq) {
        ctx->st_scores = (float *)realloc(ctx->st_scores, max_seq * sizeof(float));
        ctx->st_scores_cap = max_seq;
    }

    /* Process: prefill with [talker_hidden_proj, embed(first_code)]
     * Then generate group 1..30 autoregressively
     *
     * Python flow:
     *   inputs_embeds = cat(past_hidden, last_id_hidden)  -- shape [1, 2, hidden]
     *   code_predictor.generate(inputs_embeds, max_new_tokens=31)
     *   - Prefill with the 2-token input
     *   - Then decode tokens one at a time:
     *     generation_steps tracks which group we're predicting
     *     input = codec_embedding[generation_steps-1](token) 
     *     logits = lm_head[generation_steps](hidden)
     *
     * Simplified approach for C: process token by token through the transformer
     */

    /* We'll do a simple sequential approach:
     * Step 0: input = projected_talker_hidden, position 0
     * Step 1: input = codec_embed(first_code), position 1 → logits from lm_head[0] → group 1
     * Step 2: input = subtalker_embed[0](group1_code), position 2 → logits from lm_head[1] → group 2
     * ...
     * Step k: input = subtalker_embed[k-2](group_{k-1}_code), pos k → logits from lm_head[k-1] → group k
     */

    /* Compute RoPE cache once and reuse across calls */
    if (!ctx->st_rope_cos || !ctx->st_rope_sin || ctx->st_rope_cap < max_seq) {
        size_t rope_bytes = (size_t)max_seq * st_head_dim * sizeof(float);
        ctx->st_rope_cos = (float *)realloc(ctx->st_rope_cos, rope_bytes);
        ctx->st_rope_sin = (float *)realloc(ctx->st_rope_sin, rope_bytes);
        compute_rope_cache(ctx->st_rope_cos, ctx->st_rope_sin, max_seq, st_head_dim, cfg->talker_rope_theta);
        ctx->st_rope_cap = max_seq;
    }

    float *x = ctx->st_x;
    float *x_norm = ctx->st_x_norm;
    float *q_buf = ctx->st_q;
    float *k_buf = ctx->st_k;
    float *v_buf = ctx->st_v;
    float *attn_out = ctx->st_attn_out;
    float *logits_buf = ctx->st_logits;
    float *st_gate_buf = ctx->st_gate;
    float *st_up_buf = ctx->st_up;
    float *embed_buf = ctx->st_embed;
    float *proj_hidden = ctx->st_proj_hidden;
    size_t kv_stride = (size_t)ctx->subtalker_kv_max * st_kv_dim;
    float attn_scale = 1.0f / sqrtf((float)st_head_dim);

    /* Forward function for one sub-talker token */
    #define ST_FORWARD(input_vec, pos_idx) do { \
        memcpy(x, input_vec, st_hidden * sizeof(float)); \
        for (int sl = 0; sl < st_layers; sl++) { \
            qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[sl]; \
            kernel_rms_norm(x_norm, x, l->input_norm, st_hidden, eps); \
            kernel_matvec_bf16(q_buf, l->wq_bf16, x_norm, st_heads * st_head_dim, st_hidden); \
            kernel_matvec_bf16(k_buf, l->wk_bf16, x_norm, st_kv_dim, st_hidden); \
            kernel_matvec_bf16(v_buf, l->wv_bf16, x_norm, st_kv_dim, st_hidden); \
            for (int h = 0; h < st_heads; h++) \
                kernel_rms_norm_inplace(q_buf + h * st_head_dim, l->q_norm_weight, st_head_dim, eps); \
            for (int h = 0; h < st_kv_heads; h++) \
                kernel_rms_norm_inplace(k_buf + h * st_head_dim, l->k_norm_weight, st_head_dim, eps); \
            kernel_rope_apply(q_buf, NULL, ctx->st_rope_cos + (pos_idx) * st_head_dim, \
                              ctx->st_rope_sin + (pos_idx) * st_head_dim, st_heads, st_head_dim); \
            kernel_rope_apply(k_buf, NULL, ctx->st_rope_cos + (pos_idx) * st_head_dim, \
                              ctx->st_rope_sin + (pos_idx) * st_head_dim, st_kv_heads, st_head_dim); \
            memcpy(ctx->subtalker_kv_k + sl * kv_stride + (pos_idx) * st_kv_dim, k_buf, st_kv_dim * sizeof(float)); \
            memcpy(ctx->subtalker_kv_v + sl * kv_stride + (pos_idx) * st_kv_dim, v_buf, st_kv_dim * sizeof(float)); \
            for (int h = 0; h < st_heads; h++) { \
                int kvh = h / groups_per_head; \
                float *_q = q_buf + h * st_head_dim; \
                float *_o = attn_out + h * st_head_dim; \
                float *_scores = ctx->st_scores; \
                memset(_o, 0, st_head_dim * sizeof(float)); \
                for (int t = 0; t <= (pos_idx); t++) { \
                    float *_k = ctx->subtalker_kv_k + sl * kv_stride + t * st_kv_dim + kvh * st_head_dim; \
                    _scores[t] = st_dot(_q, _k, st_head_dim) * attn_scale; \
                } \
                kernel_softmax(_scores, (pos_idx) + 1); \
                for (int t = 0; t <= (pos_idx); t++) { \
                    float w = _scores[t]; \
                    float *_v = ctx->subtalker_kv_v + sl * kv_stride + t * st_kv_dim + kvh * st_head_dim; \
                    st_axpy(st_head_dim, w, _v, _o); \
                } \
            } \
            kernel_matvec_bf16(x_norm, l->wo_bf16, attn_out, st_hidden, st_heads * st_head_dim); \
            kernel_add_inplace(x, x_norm, st_hidden); \
            kernel_rms_norm(x_norm, x, l->post_attn_norm, st_hidden, eps); \
            float *_gate = st_gate_buf; \
            float *_up = st_up_buf; \
            if (l->gate_up_fused_bf16) { \
                kernel_swiglu_matvec_bf16(_gate, l->gate_up_fused_bf16, x_norm, st_intermediate, st_hidden); \
            } else { \
                kernel_matvec_bf16(_gate, l->gate_bf16, x_norm, st_intermediate, st_hidden); \
                kernel_matvec_bf16(_up, l->up_bf16, x_norm, st_intermediate, st_hidden); \
                kernel_silu_inplace(_gate, st_intermediate); \
                kernel_mul_inplace(_gate, _up, st_intermediate); \
            } \
            kernel_matvec_bf16(x_norm, l->down_bf16, _gate, st_hidden, st_intermediate); \
            kernel_add_inplace(x, x_norm, st_hidden); \
        } \
        ctx->subtalker_kv_len = (pos_idx) + 1; \
        kernel_rms_norm_inplace(x, ctx->subtalker.norm, st_hidden, eps); \
    } while(0)

    #define ST_PROJECT_INPUT(dst, src, src_dim) do { \
        if (ctx->subtalker.input_proj_bf16) { \
            kernel_matvec_bf16(dst, ctx->subtalker.input_proj_bf16, src, st_hidden, src_dim); \
            if (ctx->subtalker.input_proj_bias) kernel_add_inplace(dst, ctx->subtalker.input_proj_bias, st_hidden); \
        } else { \
            int copy_dim = (src_dim) < st_hidden ? (src_dim) : st_hidden; \
            memcpy(dst, src, copy_dim * sizeof(float)); \
            if (copy_dim < st_hidden) memset(dst + copy_dim, 0, (st_hidden - copy_dim) * sizeof(float)); \
        } \
    } while(0)

    /* Step 0: Process projected talker hidden (no logits generated) */
    ST_PROJECT_INPUT(proj_hidden, talker_hidden, talker_hidden_dim);
    ST_FORWARD(proj_hidden, 0);

    /* Get embedding for first_code from talker's codec_embedding (group 0 uses talker embedding) */
    {
        const uint16_t *emb = ctx->talker.codec_embedding_bf16;
        kernel_bf16_to_f32(embed_buf, emb + first_code * talker_hidden_dim, talker_hidden_dim);
    }
    ST_PROJECT_INPUT(proj_hidden, embed_buf, talker_hidden_dim);
    ST_FORWARD(proj_hidden, 1);

    /* Generate group 1 from lm_head[0] */
    kernel_matvec_bf16(logits_buf, ctx->subtalker.lm_heads_bf16[0], x, st_vocab, st_hidden);
    float rng = (float)ctx->sample_seed;
    out_codes[1] = kernel_sample_top_k(logits_buf, st_vocab, ctx->subtalker_top_k,
                                        ctx->subtalker_top_p, ctx->subtalker_temperature, &rng);

    /* Steps 2..30: generate groups 2..31 */
    for (int g = 2; g < num_groups; g++) {
        /* Embed the previous group's code using sub-talker embedding */
        kernel_bf16_to_f32(embed_buf, ctx->subtalker.codec_embeddings_bf16[g - 2] +
                           (size_t)out_codes[g - 1] * talker_hidden_dim, talker_hidden_dim);
        ST_PROJECT_INPUT(proj_hidden, embed_buf, talker_hidden_dim);
        ST_FORWARD(proj_hidden, g);
        kernel_matvec_bf16(logits_buf, ctx->subtalker.lm_heads_bf16[g - 1], x, st_vocab, st_hidden);
        out_codes[g] = kernel_sample_top_k(logits_buf, st_vocab, ctx->subtalker_top_k,
                                            ctx->subtalker_top_p, ctx->subtalker_temperature, &rng);
    }

    #undef ST_PROJECT_INPUT
    #undef ST_FORWARD
}
