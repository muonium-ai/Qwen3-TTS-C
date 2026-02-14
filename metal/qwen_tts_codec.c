/*
 * qwen_tts_codec.c - Codec Decoder (Speech Tokenizer Decoder)
 *
 * Converts codec tokens to waveform:
 *   1. SplitResidualVectorQuantizer: dequantize tokens → continuous embeddings
 *   2. Pre-conv: CausalConv1d (codebook_dim → latent_dim, k=3)
 *   3. Transformer: 8-layer sliding-window transformer (latent → hidden → latent)
 *   4. Upsample: 2× (TransConv + ConvNeXt) stages
 *   5. Vocoder (BigVGAN): initial conv → 4 blocks × (SnakeBeta + TransConv + 3×ResUnit) → final conv
 *   6. Clamp to [-1, 1]
 *
 * Total upsampling: 2 × 2 × 8 × 5 × 4 × 3 = 1920×
 * At 12.5 Hz codec rate, produces 24000 Hz audio.
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int qwen_tts_verbose;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int codec_decoder_weights_ready(const qwen_tts_ctx_t *ctx) {
    const qwen_tts_config_t *cfg = &ctx->config;
    const qwen_tts_codec_decoder_t *codec = &ctx->codec;

    if (!codec->rvq.semantic_codebooks[0].cluster_usage ||
        !codec->rvq.semantic_codebooks[0].embedding_sum ||
        !codec->rvq.semantic_output_proj ||
        !codec->rvq.acoustic_output_proj ||
        !codec->pre_conv_weight ||
        !codec->transformer_input_proj_weight ||
        !codec->transformer_output_proj_weight ||
        !codec->vocoder_pre_conv_weight ||
        !codec->vocoder_final_conv_weight) {
        return 0;
    }

    for (int q = 0; q < cfg->codec_num_quantizers - 1; q++) {
        if (!codec->rvq.acoustic_codebooks[q].cluster_usage ||
            !codec->rvq.acoustic_codebooks[q].embedding_sum) {
            return 0;
        }
    }

    for (int i = 0; i < cfg->codec_layers; i++) {
        const qwen_tts_codec_transformer_layer_t *l = &codec->transformer_layers[i];
        if (!l->input_norm || !l->post_attn_norm || !l->wq || !l->wk ||
            !l->wv || !l->wo || !l->gate || !l->up || !l->down) {
            return 0;
        }
    }

    for (int s = 0; s < 2; s++) {
        const qwen_tts_convnext_block_t *cn = &codec->upsample_convnext[s];
        if (!codec->upsample_transconv_weight[s] || !codec->upsample_transconv_bias[s] ||
            !cn->dwconv_weight || !cn->norm_weight || !cn->norm_bias ||
            !cn->pwconv1_weight || !cn->pwconv1_bias ||
            !cn->pwconv2_weight || !cn->pwconv2_bias || !cn->gamma) {
            return 0;
        }
    }

    for (int b = 0; b < 4; b++) {
        const qwen_tts_vocoder_block_t *vb = &codec->vocoder_blocks[b];
        if (!vb->act_alpha || !vb->act_beta || !vb->transconv_weight || !vb->transconv_bias) {
            return 0;
        }
        for (int r = 0; r < 3; r++) {
            const qwen_tts_vocoder_resunit_t *ru = &vb->resunits[r];
            if (!ru->act1_alpha || !ru->act1_beta || !ru->conv1_weight || !ru->conv1_bias ||
                !ru->act2_alpha || !ru->act2_beta || !ru->conv2_weight || !ru->conv2_bias) {
                return 0;
            }
        }
    }

    return 1;
}

static inline float codec_dot(const float *a, const float *b, int n) {
#ifdef USE_BLAS
    return cblas_sdot(n, a, 1, b, 1);
#else
    return kernel_dot(a, b, n);
#endif
}

static inline void codec_axpy(int n, float alpha, const float *x, float *y) {
#ifdef USE_BLAS
    cblas_saxpy(n, alpha, x, 1, y, 1);
#else
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
#endif
}

/* ========================================================================
 * RVQ Dequantization
 * ======================================================================== */

/*
 * Dequantize codes[time_steps][num_quantizers] into continuous embeddings.
 *
 * SplitResidualVectorQuantizer layout:
 *   semantic: 1 codebook (quantizer 0)
 *   acoustic: 15 codebooks (quantizers 1-15)
 *
 * For each VQ layer:
 *   embedding = embedding_sum / cluster_usage  (EuclideanCodebook)
 *   quantized = F.embedding(codes, embedding)
 *   quantized = project_out(quantized)  -- Linear if codebook_dim != dim, else Identity
 *   quantized = quantized.transpose(1, 2)
 *
 * Then: quantized = sum of all VQ layers → output_proj(Conv1d k=1)
 *
 * Output: [codebook_dim, time_steps]
 */

static float *codec_rvq_dequantize(qwen_tts_ctx_t *ctx, const int *codes,
                                     int time_steps, int num_quantizers) {
    qwen_tts_config_t *cfg = &ctx->config;
    qwen_tts_rvq_t *rvq = &ctx->codec.rvq;
    int codebook_size = cfg->codec_codebook_size;

    /* SplitResidualVectorQuantizer splits input into two halves:
     *   half_latent = latent_dim / 2 = 512
     *   vq_dim = codebook_dim / 2 = 256
     *   Each branch: input_proj(half_latent→vq_dim) → VQ(vq_dim) → output_proj(vq_dim→half_latent)
     *   Then concatenate semantic(512) + acoustic(512) = latent_dim(1024)
     */

    int latent_dim = cfg->codec_latent;           /* 1024 */
    int half_latent = latent_dim / 2;              /* 512 */
    int vq_dim = cfg->codec_codebook_dim / 2;     /* 256 */

    /* Allocate VQ-domain partial sums */
    float *semantic_sum = (float *)calloc((size_t)vq_dim * time_steps, sizeof(float));
    float *acoustic_sum = (float *)calloc((size_t)vq_dim * time_steps, sizeof(float));

    /* Process semantic codebook (quantizer 0) */
    {
        qwen_tts_codebook_t *cb = &rvq->semantic_codebooks[0];
        const float *embeddings = cb->embeddings;
        float *tmp_embeddings = NULL;
        if (!embeddings) {
            tmp_embeddings = (float *)malloc((size_t)codebook_size * vq_dim * sizeof(float));
            embeddings = tmp_embeddings;
            for (int c = 0; c < codebook_size; c++) {
                float usage = cb->cluster_usage[c];
                if (usage < 1e-5f) usage = 1e-5f;
                float inv_usage = 1.0f / usage;
                for (int d = 0; d < vq_dim; d++) {
                    tmp_embeddings[c * vq_dim + d] = cb->embedding_sum[c * vq_dim + d] * inv_usage;
                }
            }
        }

        /* Look up codes and sum */
        for (int t = 0; t < time_steps; t++) {
            int code = codes[t * num_quantizers + 0];
            if (code < 0) code = 0;
            if (code >= codebook_size) code = 0;
            for (int d = 0; d < vq_dim; d++) {
                semantic_sum[d * time_steps + t] += embeddings[code * vq_dim + d];
            }
        }
        free(tmp_embeddings);
    }

    /* Apply semantic output_proj: Conv1d(vq_dim, half_latent, 1, bias=False) */
    float *semantic_out = (float *)calloc((size_t)half_latent * time_steps, sizeof(float));
    if (rvq->semantic_output_proj) {
        /* output_proj is [half_latent, vq_dim, 1] pointwise conv = matmul per timestep */
        for (int t = 0; t < time_steps; t++) {
            for (int od = 0; od < half_latent; od++) {
                float sum = 0;
                for (int id = 0; id < vq_dim; id++) {
                    sum += rvq->semantic_output_proj[od * vq_dim + id] * semantic_sum[id * time_steps + t];
                }
                semantic_out[od * time_steps + t] = sum;
            }
        }
    } else {
        if (half_latent != vq_dim) {
            fprintf(stderr, "Error: missing semantic output projection for RVQ dequantization\n");
            free(semantic_sum); free(acoustic_sum);
            free(semantic_out);
            return NULL;
        }
        memcpy(semantic_out, semantic_sum, (size_t)half_latent * time_steps * sizeof(float));
    }

    /* Process acoustic codebooks (quantizers 1..num_quantizers-1) */
    for (int q = 1; q < num_quantizers; q++) {
        qwen_tts_codebook_t *cb = &rvq->acoustic_codebooks[q - 1];
        const float *embeddings = cb->embeddings;
        float *tmp_embeddings = NULL;
        if (!embeddings) {
            tmp_embeddings = (float *)malloc((size_t)codebook_size * vq_dim * sizeof(float));
            embeddings = tmp_embeddings;
            for (int c = 0; c < codebook_size; c++) {
                float usage = cb->cluster_usage[c];
                if (usage < 1e-5f) usage = 1e-5f;
                float inv_usage = 1.0f / usage;
                for (int d = 0; d < vq_dim; d++) {
                    tmp_embeddings[c * vq_dim + d] = cb->embedding_sum[c * vq_dim + d] * inv_usage;
                }
            }
        }
        for (int t = 0; t < time_steps; t++) {
            int code = codes[t * num_quantizers + q];
            if (code < 0) code = 0;
            if (code >= codebook_size) code = 0;
            for (int d = 0; d < vq_dim; d++) {
                acoustic_sum[d * time_steps + t] += embeddings[code * vq_dim + d];
            }
        }
        free(tmp_embeddings);
    }

    /* Apply acoustic output_proj */
    float *acoustic_out = (float *)calloc((size_t)half_latent * time_steps, sizeof(float));
    if (rvq->acoustic_output_proj) {
        for (int t = 0; t < time_steps; t++) {
            for (int od = 0; od < half_latent; od++) {
                float sum = 0;
                for (int id = 0; id < vq_dim; id++) {
                    sum += rvq->acoustic_output_proj[od * vq_dim + id] * acoustic_sum[id * time_steps + t];
                }
                acoustic_out[od * time_steps + t] = sum;
            }
        }
    } else {
        if (half_latent != vq_dim) {
            fprintf(stderr, "Error: missing acoustic output projection for RVQ dequantization\n");
            free(semantic_sum); free(acoustic_sum);
            free(semantic_out); free(acoustic_out);
            return NULL;
        }
        memcpy(acoustic_out, acoustic_sum, (size_t)half_latent * time_steps * sizeof(float));
    }

    /* Sum semantic + acoustic → [codebook_dim=512, time_steps] */
    float *output = (float *)malloc((size_t)half_latent * time_steps * sizeof(float));
    for (size_t i = 0; i < (size_t)half_latent * time_steps; i++) {
        output[i] = semantic_out[i] + acoustic_out[i];
    }

    free(semantic_sum); free(acoustic_sum);
    free(semantic_out); free(acoustic_out);

    return output; /* [codebook_dim, time_steps] in channels-first format */
}

/* ========================================================================
 * Codec Transformer (8 layers, sliding window attention, LayerScale)
 * ======================================================================== */

static void codec_transformer_forward(qwen_tts_ctx_t *ctx, float *hidden,
                                        int seq_len) {
    qwen_tts_config_t *cfg = &ctx->config;
    int codec_hidden = cfg->codec_hidden;
    int latent = cfg->codec_latent;
    int layers = cfg->codec_layers;
    int heads = cfg->codec_heads;
    int kv_heads = cfg->codec_kv_heads;
    int head_dim = codec_hidden / heads;
    int kv_dim = kv_heads * head_dim;
    int intermediate = cfg->codec_intermediate;
    int sliding_window = cfg->codec_sliding_window;
    int groups_per_head = heads / kv_heads;
    float eps = cfg->codec_rms_norm_eps;

    /* hidden comes as [seq_len, latent] format (already transposed from channels-first) */

    /* Input projection: latent_dim -> codec_hidden (batch GEMM) */
    float *x = (float *)malloc((size_t)seq_len * codec_hidden * sizeof(float));
    kernel_matmul_f32(
        x,
        hidden,
        ctx->codec.transformer_input_proj_weight,
        seq_len,
        codec_hidden,
        latent
    );
    if (ctx->codec.transformer_input_proj_bias) {
        for (int t = 0; t < seq_len; t++) {
            kernel_add_inplace(
                x + t * codec_hidden,
                ctx->codec.transformer_input_proj_bias,
                codec_hidden
            );
        }
    }

    /* Compute RoPE cache */
    float *rope_cos = (float *)malloc((size_t)seq_len * head_dim * sizeof(float));
    float *rope_sin = (float *)malloc((size_t)seq_len * head_dim * sizeof(float));
    {
        int half = head_dim / 2;
        float theta = 10000.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                float angle = (float)pos * freq;
                rope_cos[pos * head_dim + i] = cosf(angle);
                rope_cos[pos * head_dim + i + half] = cosf(angle);
                rope_sin[pos * head_dim + i] = sinf(angle);
                rope_sin[pos * head_dim + i + half] = sinf(angle);
            }
        }
    }

    /* Scratch buffers */
    float *x_norm = (float *)malloc((size_t)seq_len * codec_hidden * sizeof(float));
    float *q_all = (float *)malloc((size_t)seq_len * heads * head_dim * sizeof(float));
    float *k_all = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *v_all = (float *)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)seq_len * heads * head_dim * sizeof(float));
    float *attn_scores = (float *)malloc((size_t)seq_len * sizeof(float));
    float *gate_all = (float *)malloc((size_t)seq_len * intermediate * sizeof(float));
    float *up_all = (float *)malloc((size_t)seq_len * intermediate * sizeof(float));

    for (int layer = 0; layer < layers; layer++) {
        qwen_tts_codec_transformer_layer_t *l = &ctx->codec.transformer_layers[layer];

        /* 1. Input RMSNorm */
        for (int t = 0; t < seq_len; t++)
            kernel_rms_norm(x_norm + t * codec_hidden, x + t * codec_hidden,
                           l->input_norm, codec_hidden, eps);

        /* 2. Q, K, V projections (batch GEMM) */
        kernel_matmul_f32(q_all, x_norm, l->wq, seq_len, heads * head_dim, codec_hidden);
        kernel_matmul_f32(k_all, x_norm, l->wk, seq_len, kv_dim, codec_hidden);
        kernel_matmul_f32(v_all, x_norm, l->wv, seq_len, kv_dim, codec_hidden);

        /* 3. RoPE (standard, not M-RoPE. No QK-Norm for codec decoder) */
        for (int t = 0; t < seq_len; t++) {
            kernel_rope_apply(q_all + t * heads * head_dim, NULL,
                             rope_cos + t * head_dim, rope_sin + t * head_dim,
                             heads, head_dim);
            kernel_rope_apply(k_all + t * kv_dim, NULL,
                             rope_cos + t * head_dim, rope_sin + t * head_dim,
                             kv_heads, head_dim);
        }

        /* 4. Sliding-window causal attention */
        float scale = 1.0f / sqrtf((float)head_dim);
        memset(attn_out, 0, (size_t)seq_len * heads * head_dim * sizeof(float));

        for (int h = 0; h < heads; h++) {
            int kv_h = h / groups_per_head;
            for (int qi = 0; qi < seq_len; qi++) {
                float *qh = q_all + qi * heads * head_dim + h * head_dim;
                int start = qi - sliding_window + 1;
                if (start < 0) start = 0;

                /* Compute attention scores within window */
                int wlen = qi - start + 1;
                for (int i = 0; i < wlen; i++) {
                    int ki = start + i;
                    float *kh = k_all + ki * kv_dim + kv_h * head_dim;
                    attn_scores[i] = codec_dot(qh, kh, head_dim) * scale;
                }
                kernel_softmax(attn_scores, wlen);

                float *oh = attn_out + qi * heads * head_dim + h * head_dim;
                for (int i = 0; i < wlen; i++) {
                    int ki = start + i;
                    float w = attn_scores[i];
                    float *vh = v_all + ki * kv_dim + kv_h * head_dim;
                    codec_axpy(head_dim, w, vh, oh);
                }
            }
        }

        /* 5. Output projection + LayerScale + residual */
        kernel_matmul_f32(
            x_norm,
            attn_out,
            l->wo,
            seq_len,
            codec_hidden,
            heads * head_dim
        );
        for (int t = 0; t < seq_len; t++) {
            if (l->attn_layer_scale)
                kernel_mul_inplace(x_norm + t * codec_hidden, l->attn_layer_scale, codec_hidden);
            kernel_add_inplace(x + t * codec_hidden, x_norm + t * codec_hidden, codec_hidden);
        }

        /* 6. Post-attention norm + SwiGLU MLP + LayerScale */
        for (int t = 0; t < seq_len; t++)
            kernel_rms_norm(x_norm + t * codec_hidden, x + t * codec_hidden,
                           l->post_attn_norm, codec_hidden, eps);

        /* MLP: gate + up -> silu -> mul -> down (batch GEMM) */
        kernel_matmul_f32(gate_all, x_norm, l->gate, seq_len, intermediate, codec_hidden);
        kernel_matmul_f32(up_all, x_norm, l->up, seq_len, intermediate, codec_hidden);

        for (int t = 0; t < seq_len; t++) {
            float *gate = gate_all + (size_t)t * intermediate;
            float *up = up_all + (size_t)t * intermediate;
            kernel_silu_inplace(gate, intermediate);
            kernel_mul_inplace(gate, up, intermediate);
        }

        kernel_matmul_f32(
            x_norm,
            gate_all,
            l->down,
            seq_len,
            codec_hidden,
            intermediate
        );

        for (int t = 0; t < seq_len; t++) {
            /* LayerScale */
            if (l->mlp_layer_scale)
                kernel_mul_inplace(x_norm + t * codec_hidden, l->mlp_layer_scale, codec_hidden);
            kernel_add_inplace(x + t * codec_hidden, x_norm + t * codec_hidden, codec_hidden);
        }
    }

    /* Final norm */
    if (ctx->codec.transformer_norm) {
        for (int t = 0; t < seq_len; t++)
            kernel_rms_norm_inplace(x + t * codec_hidden, ctx->codec.transformer_norm, codec_hidden, eps);
    }

    /* Output projection: codec_hidden -> latent_dim (batch GEMM) */
    kernel_matmul_f32(
        hidden,
        x,
        ctx->codec.transformer_output_proj_weight,
        seq_len,
        latent,
        codec_hidden
    );
    if (ctx->codec.transformer_output_proj_bias) {
        for (int t = 0; t < seq_len; t++) {
            kernel_add_inplace(
                hidden + t * latent,
                ctx->codec.transformer_output_proj_bias,
                latent
            );
        }
    }

    free(x); free(x_norm); free(q_all); free(k_all); free(v_all);
    free(attn_out); free(attn_scores); free(gate_all); free(up_all);
    free(rope_cos); free(rope_sin);
}

/* ========================================================================
 * ConvNeXt upsampling block
 * ======================================================================== */

static void codec_convnext_forward(qwen_tts_convnext_block_t *block,
                                     float *hidden, int dim, int *length) {
    int len = *length;

    /* Residual */
    float *residual = (float *)malloc((size_t)dim * len * sizeof(float));
    memcpy(residual, hidden, (size_t)dim * len * sizeof(float));

    /* Depthwise causal conv (k=7, groups=dim) */
    float *conv_out = (float *)malloc((size_t)dim * len * sizeof(float));
    kernel_causal_conv1d(conv_out, hidden, block->dwconv_weight, block->dwconv_bias,
                         dim, dim, 7, len, 1, dim);

    /* permute to [len, dim] for LayerNorm and pointwise ops */
    float *x_ld = (float *)malloc((size_t)len * dim * sizeof(float));
    for (int c = 0; c < dim; c++)
        for (int t = 0; t < len; t++)
            x_ld[t * dim + c] = conv_out[c * len + t];

    /* LayerNorm */
    for (int t = 0; t < len; t++)
        kernel_layer_norm(x_ld + t * dim, x_ld + t * dim, block->norm_weight, block->norm_bias, dim, 1e-6f);

    /* pwconv1: [dim] → [4*dim] */
    int dim4 = 4 * dim;
    float *pw1 = (float *)malloc((size_t)len * dim4 * sizeof(float));
    for (int t = 0; t < len; t++) {
        kernel_matvec_f32(pw1 + t * dim4, block->pwconv1_weight, x_ld + t * dim, dim4, dim);
        if (block->pwconv1_bias)
            kernel_add_inplace(pw1 + t * dim4, block->pwconv1_bias, dim4);
    }

    /* GELU */
    kernel_gelu_inplace(pw1, len * dim4);

    /* pwconv2: [4*dim] → [dim] */
    for (int t = 0; t < len; t++) {
        kernel_matvec_f32(x_ld + t * dim, block->pwconv2_weight, pw1 + t * dim4, dim, dim4);
        if (block->pwconv2_bias)
            kernel_add_inplace(x_ld + t * dim, block->pwconv2_bias, dim);
    }

    /* Apply gamma (learnable residual scale) */
    for (int t = 0; t < len; t++)
        kernel_mul_inplace(x_ld + t * dim, block->gamma, dim);

    /* permute back to [dim, len] */
    for (int c = 0; c < dim; c++)
        for (int t = 0; t < len; t++)
            hidden[c * len + t] = x_ld[t * dim + c];

    /* Skip connection */
    kernel_add_inplace(hidden, residual, dim * len);

    free(residual); free(conv_out); free(x_ld); free(pw1);
}

/* ========================================================================
 * Vocoder residual unit
 * ======================================================================== */

typedef struct {
    float *residual;
    float *conv1_out;
    size_t cap;
} vocoder_resunit_scratch_t;

static int ensure_vocoder_resunit_scratch(vocoder_resunit_scratch_t *s, size_t n) {
    if (s->cap >= n) return 0;

    float *p = (float *)realloc(s->residual, n * sizeof(float));
    if (!p) return -1;
    s->residual = p;

    p = (float *)realloc(s->conv1_out, n * sizeof(float));
    if (!p) return -1;
    s->conv1_out = p;

    s->cap = n;
    return 0;
}

static int vocoder_resunit_forward(qwen_tts_vocoder_resunit_t *unit,
                                      float *hidden, int dim, int length, int dilation,
                                      vocoder_resunit_scratch_t *scratch) {
    size_t n = (size_t)dim * length;
    if (ensure_vocoder_resunit_scratch(scratch, n) != 0) return -1;
    float *residual = scratch->residual;
    float *conv1_out = scratch->conv1_out;
    memcpy(residual, hidden, n * sizeof(float));

    /* SnakeBeta activation 1 (in-place) */
    kernel_snake_beta(hidden, hidden, unit->act1_alpha, unit->act1_beta, dim, length);

    /* Causal conv1 (k=7, dilation) */
    kernel_causal_conv1d(conv1_out, hidden, unit->conv1_weight, unit->conv1_bias,
                         dim, dim, 7, length, dilation, 1);

    /* SnakeBeta activation 2 (in-place) */
    kernel_snake_beta(conv1_out, conv1_out, unit->act2_alpha, unit->act2_beta, dim, length);

    /* Causal conv2 (k=1, dilation=1) */
    kernel_causal_conv1d(hidden, conv1_out, unit->conv2_weight, unit->conv2_bias,
                         dim, dim, 1, length, 1, 1);

    /* Skip connection */
    kernel_add_inplace(hidden, residual, dim * length);
    return 0;
}

/* ========================================================================
 * Full codec decode pipeline
 * ======================================================================== */

float *qwen_tts_codec_decode(qwen_tts_ctx_t *ctx, const int *codes,
                              int time_steps, int *out_samples) {
    if (!ctx || !codes || !out_samples || time_steps <= 0) {
        if (out_samples) *out_samples = 0;
        return NULL;
    }
    if (!codec_decoder_weights_ready(ctx)) {
        fprintf(stderr, "Error: codec decoder is not fully loaded; cannot decode audio\n");
        *out_samples = 0;
        return NULL;
    }

    qwen_tts_config_t *cfg = &ctx->config;
    int num_quantizers = cfg->codec_num_quantizers;
    int latent_dim = cfg->codec_latent;
    int codebook_dim = cfg->codec_codebook_dim;

    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Codec decode: %d timesteps, %d quantizers\n", time_steps, num_quantizers);

    double stage_t0 = now_ms();
    double stage_rvq_ms = 0.0;
    double stage_preconv_ms = 0.0;
    double stage_transformer_ms = 0.0;
    double stage_upsample_ms = 0.0;
    double stage_vocoder_ms = 0.0;

    /* 1. RVQ Dequantize → [codebook_dim, time_steps] */
    float *hidden = codec_rvq_dequantize(ctx, codes, time_steps, num_quantizers);
    if (!hidden) {
        *out_samples = 0;
        return NULL;
    }
    stage_rvq_ms = now_ms() - stage_t0;

    /* 2. Pre-conv: CausalConv1d(codebook_dim=512, latent_dim=1024, k=3) */
    stage_t0 = now_ms();
    float *pre_conv_out = (float *)malloc((size_t)latent_dim * time_steps * sizeof(float));
    kernel_causal_conv1d(pre_conv_out, hidden, ctx->codec.pre_conv_weight,
                         ctx->codec.pre_conv_bias, codebook_dim, latent_dim,
                         3, time_steps, 1, 1);
    free(hidden);
    stage_preconv_ms = now_ms() - stage_t0;

    /* 3. Transpose to [time_steps, latent_dim] for transformer */
    float *hidden_seq = (float *)malloc((size_t)time_steps * latent_dim * sizeof(float));
    for (int c = 0; c < latent_dim; c++)
        for (int t = 0; t < time_steps; t++)
            hidden_seq[t * latent_dim + c] = pre_conv_out[c * time_steps + t];
    free(pre_conv_out);

    /* 4. Transformer forward pass */
    stage_t0 = now_ms();
    codec_transformer_forward(ctx, hidden_seq, time_steps);
    stage_transformer_ms = now_ms() - stage_t0;

    /* 5. Transpose back to [latent_dim, time_steps] */
    hidden = (float *)malloc((size_t)latent_dim * time_steps * sizeof(float));
    for (int c = 0; c < latent_dim; c++)
        for (int t = 0; t < time_steps; t++)
            hidden[c * time_steps + t] = hidden_seq[t * latent_dim + c];
    free(hidden_seq);

    /* 6. Upsample stages (2× TransConv + ConvNeXt) */
    stage_t0 = now_ms();
    int current_len = time_steps;
    for (int stage = 0; stage < 2; stage++) {
        int factor = cfg->codec_upsampling_ratios[stage];
        int new_len;

        /* TransposedConv1d upsample */
        float *up_out = (float *)malloc((size_t)latent_dim * (current_len * factor + factor) * sizeof(float));
        kernel_transposed_conv1d(up_out, hidden, ctx->codec.upsample_transconv_weight[stage],
                                  ctx->codec.upsample_transconv_bias[stage],
                                  latent_dim, latent_dim, factor, factor, current_len, &new_len);
        free(hidden);
        hidden = (float *)realloc(up_out, (size_t)latent_dim * new_len * sizeof(float));
        current_len = new_len;

        /* ConvNeXt block */
        codec_convnext_forward(&ctx->codec.upsample_convnext[stage], hidden, latent_dim, &current_len);
    }
    stage_upsample_ms = now_ms() - stage_t0;

    /* 7. Vocoder */
    stage_t0 = now_ms();

    /* 7a. Initial conv: CausalConv1d(latent_dim, decoder_dim, k=7) */
    int decoder_dim = cfg->codec_decoder_dim;
    float *voc = (float *)malloc((size_t)decoder_dim * current_len * sizeof(float));
    kernel_causal_conv1d(voc, hidden, ctx->codec.vocoder_pre_conv_weight,
                         ctx->codec.vocoder_pre_conv_bias,
                         latent_dim, decoder_dim, 7, current_len, 1, 1);
    free(hidden);

    /* 7b. 4 decoder blocks: SnakeBeta → TransConv → 3 ResUnits */
    int upsample_rates[4] = {
        cfg->codec_upsample_rates[0],
        cfg->codec_upsample_rates[1],
        cfg->codec_upsample_rates[2],
        cfg->codec_upsample_rates[3],
    };
    int current_dim = decoder_dim;
    vocoder_resunit_scratch_t ru_scratch = {0};

    for (int block = 0; block < 4; block++) {
        int in_dim = current_dim;
        int out_dim = in_dim / 2;
        int rate = upsample_rates[block];
        qwen_tts_vocoder_block_t *vb = &ctx->codec.vocoder_blocks[block];

        /* SnakeBeta activation (in-place) */
        kernel_snake_beta(voc, voc, vb->act_alpha, vb->act_beta, in_dim, current_len);

        /* TransposedConv1d: [in_dim, current_len] → [out_dim, new_len] */
        int new_len;
        int kernel = 2 * rate;
        float *transconv_out = (float *)malloc((size_t)out_dim * (current_len * rate + kernel) * sizeof(float));
        kernel_transposed_conv1d(transconv_out, voc, vb->transconv_weight, vb->transconv_bias,
                                  in_dim, out_dim, kernel, rate, current_len, &new_len);
        free(voc);

        voc = (float *)realloc(transconv_out, (size_t)out_dim * new_len * sizeof(float));
        current_len = new_len;
        current_dim = out_dim;

        /* 3 Residual units with dilations 1, 3, 9 */
        int dilations[3] = {1, 3, 9};
        for (int ru = 0; ru < 3; ru++) {
            if (vocoder_resunit_forward(&vb->resunits[ru], voc, current_dim, current_len,
                                        dilations[ru], &ru_scratch) != 0) {
                free(voc);
                free(ru_scratch.residual);
                free(ru_scratch.conv1_out);
                *out_samples = 0;
                return NULL;
            }
        }
    }

    /* 7c. Final: SnakeBeta → CausalConv1d → output channel 1 */
    kernel_snake_beta(voc, voc, ctx->codec.vocoder_final_act_alpha,
                      ctx->codec.vocoder_final_act_beta, current_dim, current_len);

    /* Final conv: [current_dim, current_len] → [1, current_len] */
    float *wav = (float *)malloc((size_t)current_len * sizeof(float));
    kernel_causal_conv1d(wav, voc, ctx->codec.vocoder_final_conv_weight,
                         ctx->codec.vocoder_final_conv_bias,
                         current_dim, 1, 7, current_len, 1, 1);
    free(voc);
    free(ru_scratch.residual);
    free(ru_scratch.conv1_out);

    /* 8. Clamp to [-1, 1] */
    kernel_clamp(wav, current_len, -1.0f, 1.0f);
    stage_vocoder_ms = now_ms() - stage_t0;

    *out_samples = current_len;

    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Codec decode complete: %d samples (%.2f seconds)\n",
                current_len, (float)current_len / QWEN_TTS_SAMPLE_RATE);
    if (qwen_tts_verbose >= 2) {
        fprintf(stderr, "Codec stages (ms): rvq=%.1f preconv=%.1f transformer=%.1f upsample=%.1f vocoder=%.1f\n",
                stage_rvq_ms, stage_preconv_ms, stage_transformer_ms, stage_upsample_ms, stage_vocoder_ms);
    }

    return wav;
}
