/*
 * qwen_tts.h - Qwen3-TTS Pure C Inference Engine
 *
 * Two-stage text-to-speech:
 *   Stage 1 (Talker):    Text tokens -> Codec tokens  (autoregressive LM)
 *   Stage 2 (Decoder):   Codec tokens -> Waveform     (neural codec decoder)
 *
 * Supports CustomVoice mode (speaker ID + optional language).
 */

#ifndef QWEN_TTS_H
#define QWEN_TTS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ========================================================================
 * Constants
 * ======================================================================== */

#define QWEN_TTS_SAMPLE_RATE      24000
#define QWEN_TTS_DECODE_UPSAMPLE  1920    /* samples per codec frame at 12.5Hz */

/* Talker defaults */
#define QWEN_TTS_TALKER_VOCAB       3072
#define QWEN_TTS_TALKER_HIDDEN      1024
#define QWEN_TTS_TALKER_INTERMEDIATE 2048
#define QWEN_TTS_TALKER_LAYERS      20
#define QWEN_TTS_TALKER_HEADS       16
#define QWEN_TTS_TALKER_KV_HEADS    2
#define QWEN_TTS_TALKER_HEAD_DIM    64    /* hidden/heads */
#define QWEN_TTS_TALKER_TEXT_HIDDEN 2048
#define QWEN_TTS_TALKER_TEXT_VOCAB  151936
#define QWEN_TTS_NUM_CODE_GROUPS    32

/* Sub-talker (Code Predictor) defaults */
#define QWEN_TTS_SUBTALKER_VOCAB         2048
#define QWEN_TTS_SUBTALKER_HIDDEN        1024
#define QWEN_TTS_SUBTALKER_INTERMEDIATE  3072
#define QWEN_TTS_SUBTALKER_LAYERS        5
#define QWEN_TTS_SUBTALKER_HEADS         16
#define QWEN_TTS_SUBTALKER_KV_HEADS      8
#define QWEN_TTS_SUBTALKER_HEAD_DIM      128

/* Codec decoder defaults (12Hz) */
#define QWEN_TTS_CODEC_NUM_QUANTIZERS  16
#define QWEN_TTS_CODEC_CODEBOOK_SIZE   2048
#define QWEN_TTS_CODEC_HIDDEN          1024
#define QWEN_TTS_CODEC_LATENT          1024
#define QWEN_TTS_CODEC_LAYERS          8
#define QWEN_TTS_CODEC_HEADS           16
#define QWEN_TTS_CODEC_KV_HEADS        16
#define QWEN_TTS_CODEC_INTERMEDIATE    3072
#define QWEN_TTS_CODEC_SLIDING_WINDOW  72
#define QWEN_TTS_CODEC_DECODER_DIM     1536

/* Max layer counts for static array sizing */
#define QWEN_TTS_MAX_TALKER_LAYERS    32
#define QWEN_TTS_MAX_SUBTALKER_LAYERS 8
#define QWEN_TTS_MAX_CODEC_LAYERS     12

/* Special token IDs - text domain (Qwen2 tokenizer) */
#define QWEN_TTS_TOKEN_IM_START     151644
#define QWEN_TTS_TOKEN_IM_END       151645
#define QWEN_TTS_TOKEN_ENDOFTEXT    151643
#define QWEN_TTS_TOKEN_TTS_PAD      151671
#define QWEN_TTS_TOKEN_TTS_BOS      151672
#define QWEN_TTS_TOKEN_TTS_EOS      151673

/* Special token IDs - codec domain (defaults, overridden by config) */
#define QWEN_TTS_CODEC_PAD          2148
#define QWEN_TTS_CODEC_BOS          2149
#define QWEN_TTS_CODEC_EOS          2150
#define QWEN_TTS_CODEC_THINK        2154
#define QWEN_TTS_CODEC_NOTHINK      2155
#define QWEN_TTS_CODEC_THINK_BOS    2156
#define QWEN_TTS_CODEC_THINK_EOS    2157

/* ========================================================================
 * Configuration
 * ======================================================================== */

typedef struct {
    /* Talker (main LM) */
    int talker_vocab_size;
    int talker_hidden;
    int talker_intermediate;
    int talker_layers;
    int talker_heads;
    int talker_kv_heads;
    int talker_head_dim;
    int talker_text_hidden;
    int talker_text_vocab;
    int num_code_groups;
    float talker_rms_norm_eps;
    float talker_rope_theta;

    /* M-RoPE section sizes (3 sections for temporal, height, width) */
    int mrope_section[3];

    /* Sub-talker (code predictor) */
    int subtalker_vocab_size;
    int subtalker_hidden;
    int subtalker_intermediate;
    int subtalker_layers;
    int subtalker_heads;
    int subtalker_kv_heads;
    int subtalker_head_dim;

    /* Codec decoder */
    int codec_num_quantizers;
    int codec_codebook_size;
    int codec_codebook_dim;       /* e.g. 128; VQ internal dim = codebook_dim/2 */
    int codec_hidden;
    int codec_latent;
    int codec_layers;
    int codec_heads;
    int codec_kv_heads;
    int codec_intermediate;
    int codec_sliding_window;
    int codec_decoder_dim;
    float codec_rms_norm_eps;
    float codec_layer_scale;
    int codec_upsample_rates[4];
    int codec_upsampling_ratios[2];

    /* Speaker/language maps - loaded from config.json */
    int n_speakers;
    char **speaker_names;
    int *speaker_ids;
    int n_languages;
    char **language_names;
    int *language_ids;

    /* Codec special token IDs (from config) */
    int codec_pad_id;
    int codec_bos_id;
    int codec_eos_id;
    int codec_nothink_id;
    int codec_think_id;
    int codec_think_bos_id;
    int codec_think_eos_id;
} qwen_tts_config_t;

/* ========================================================================
 * Talker Layer Weights
 * ======================================================================== */

typedef struct {
    /* Self-attention (no bias) - kept as BF16 for mmap */
    uint16_t *wq_bf16;          /* [num_heads*head_dim, hidden] */
    uint16_t *wk_bf16;          /* [num_kv_heads*head_dim, hidden] */
    uint16_t *wv_bf16;          /* [num_kv_heads*head_dim, hidden] */
    uint16_t *wo_bf16;          /* [hidden, num_heads*head_dim] */

    /* Per-head Q/K RMSNorm */
    float *q_norm_weight;       /* [head_dim] */
    float *k_norm_weight;       /* [head_dim] */

    /* RMSNorm (no bias) */
    float *input_norm;          /* [hidden] */
    float *post_attn_norm;      /* [hidden] */

    /* SwiGLU MLP (no bias) */
    uint16_t *gate_bf16;        /* [intermediate, hidden] */
    uint16_t *up_bf16;          /* [intermediate, hidden] */
    uint16_t *down_bf16;        /* [hidden, intermediate] */

    /* Fused gate+up for single-token matvec */
    uint16_t *gate_up_fused_bf16; /* [2*intermediate, hidden] */

#ifdef USE_METAL
    /* Metal buffer IDs for GPU-accelerated forward pass */
    int mtl_wq, mtl_wk, mtl_wv, mtl_wo;
    int mtl_q_norm, mtl_k_norm;
    int mtl_input_norm, mtl_post_attn_norm;
    int mtl_gate, mtl_up, mtl_down;
    int mtl_gate_up_fused;
#endif
} qwen_tts_talker_layer_t;

typedef struct {
    /* Token embeddings */
    uint16_t *codec_embedding_bf16;   /* [vocab, hidden] */
    uint16_t *text_embedding_bf16;    /* [text_vocab, text_hidden] */

    /* Text projection MLP: text_hidden -> text_hidden -> hidden */
    uint16_t *text_proj_fc1_bf16;     /* [text_hidden, text_hidden] */
    float    *text_proj_fc1_bias;     /* [text_hidden] */
    uint16_t *text_proj_fc2_bf16;     /* [hidden, text_hidden] */
    float    *text_proj_fc2_bias;     /* [hidden] */

    /* Transformer layers */
    qwen_tts_talker_layer_t layers[QWEN_TTS_MAX_TALKER_LAYERS];

    /* Final RMSNorm */
    float *norm;                      /* [hidden] */

    /* Codec head (logit projection, tied or separate) */
    uint16_t *codec_head_bf16;        /* [vocab, hidden] */

#ifdef USE_METAL
    int mtl_codec_embedding, mtl_text_embedding;
    int mtl_text_proj_fc1, mtl_text_proj_fc1_bias;
    int mtl_text_proj_fc2, mtl_text_proj_fc2_bias;
    int mtl_norm;
    int mtl_codec_head;
#endif
} qwen_tts_talker_t;

/* ========================================================================
 * Sub-Talker (Code Predictor) Layer Weights
 * ======================================================================== */

typedef struct {
    uint16_t *wq_bf16;
    uint16_t *wk_bf16;
    uint16_t *wv_bf16;
    uint16_t *wo_bf16;
    float *q_norm_weight;
    float *k_norm_weight;
    float *input_norm;
    float *post_attn_norm;
    uint16_t *gate_bf16;
    uint16_t *up_bf16;
    uint16_t *down_bf16;
    uint16_t *gate_up_fused_bf16;
#ifdef USE_METAL
    int mtl_wq, mtl_wk, mtl_wv, mtl_wo;
    int mtl_q_norm, mtl_k_norm;
    int mtl_input_norm, mtl_post_attn_norm;
    int mtl_gate, mtl_up, mtl_down;
    int mtl_gate_up_fused;
#endif
} qwen_tts_subtalker_layer_t;

typedef struct {
    /* 31 codec embeddings (for groups 1-31) */
    uint16_t *codec_embeddings_bf16[QWEN_TTS_NUM_CODE_GROUPS - 1];  /* each [subtalker_vocab, embedding_dim] */

    /* Input projection (talker hidden -> subtalker hidden) */
    uint16_t *input_proj_bf16;    /* [subtalker_hidden, talker_hidden] or NULL if same dim */
    float    *input_proj_bias;

    /* Transformer layers */
    qwen_tts_subtalker_layer_t layers[QWEN_TTS_MAX_SUBTALKER_LAYERS];

    /* Final RMSNorm */
    float *norm;

    /* 31 LM heads (one per code group 1-31) */
    uint16_t *lm_heads_bf16[QWEN_TTS_NUM_CODE_GROUPS - 1];  /* each [subtalker_vocab, subtalker_hidden] */
#ifdef USE_METAL
    int mtl_codec_embeddings[QWEN_TTS_NUM_CODE_GROUPS - 1];
    int mtl_input_proj, mtl_input_proj_bias;
    int mtl_norm;
    int mtl_lm_heads[QWEN_TTS_NUM_CODE_GROUPS - 1];
#endif
} qwen_tts_subtalker_t;

/* ========================================================================
 * Codec Decoder Weights
 * ======================================================================== */

/* SplitResidualVectorQuantizer codebook */
typedef struct {
    float *cluster_usage;     /* [codebook_size] */
    float *embedding_sum;     /* [codebook_size, codebook_dim] */
    float *embeddings;        /* [codebook_size, codebook_dim] = embedding_sum / cluster_usage */
    float *project_out_weight;/* [dim, codebook_dim] or NULL if same dim */
    float *project_out_bias;  /* [dim] or NULL */
} qwen_tts_codebook_t;

typedef struct {
    /* Semantic (1 codebook) and acoustic (15 codebooks) quantizers */
    qwen_tts_codebook_t semantic_codebooks[1];
    qwen_tts_codebook_t acoustic_codebooks[QWEN_TTS_CODEC_NUM_QUANTIZERS - 1];

    /* Input/output projections for semantic and acoustic */
    float *semantic_input_proj;   /* Conv1d weight [dim, input_dim, 1] */
    float *semantic_output_proj;  /* Conv1d weight [output_dim, dim, 1] */
    float *acoustic_input_proj;
    float *acoustic_output_proj;
} qwen_tts_rvq_t;

/* Codec transformer layer */
typedef struct {
    float *input_norm;
    float *post_attn_norm;
    float *attn_layer_scale;       /* [hidden] LayerScale */
    float *mlp_layer_scale;        /* [hidden] LayerScale */

    /* Attention (no bias) */
    float *wq;                     /* [num_heads*head_dim, hidden] */
    float *wk;                     /* [num_kv_heads*head_dim, hidden] */
    float *wv;                     /* [num_kv_heads*head_dim, hidden] */
    float *wo;                     /* [hidden, num_heads*head_dim] */

    /* SwiGLU MLP */
    float *gate;                   /* [intermediate, hidden] */
    float *up;                     /* [intermediate, hidden] */
    float *down;                   /* [hidden, intermediate] */

#ifdef USE_METAL
    int mtl_input_norm, mtl_post_attn_norm;
    int mtl_attn_layer_scale, mtl_mlp_layer_scale;
    int mtl_wq, mtl_wk, mtl_wv, mtl_wo;
    int mtl_gate, mtl_up, mtl_down;
#endif
} qwen_tts_codec_transformer_layer_t;

/* ConvNeXt block */
typedef struct {
    /* Depthwise causal conv (kernel=7) */
    float *dwconv_weight;          /* [dim, 1, 7] */
    float *dwconv_bias;            /* [dim] - from conv */

    /* LayerNorm */
    float *norm_weight;            /* [dim] */
    float *norm_bias;              /* [dim] */

    /* Pointwise convolutions */
    float *pwconv1_weight;         /* [4*dim, dim] */
    float *pwconv1_bias;           /* [4*dim] */
    float *pwconv2_weight;         /* [dim, 4*dim] */
    float *pwconv2_bias;           /* [dim] */

    /* Learnable gamma */
    float *gamma;                  /* [dim] */
} qwen_tts_convnext_block_t;

/* Vocoder residual unit (SnakeBeta + Conv + SnakeBeta + Conv) */
typedef struct {
    float *act1_alpha;             /* [dim] */
    float *act1_beta;              /* [dim] */
    float *conv1_weight;           /* [dim, dim, 7] */
    float *conv1_bias;             /* [dim] */
    float *act2_alpha;             /* [dim] */
    float *act2_beta;              /* [dim] */
    float *conv2_weight;           /* [dim, dim, 1] */
    float *conv2_bias;             /* [dim] */
#ifdef USE_METAL
    int mtl_act1_alpha, mtl_act1_beta;
    int mtl_conv1_weight, mtl_conv1_bias;
    int mtl_act2_alpha, mtl_act2_beta;
    int mtl_conv2_weight, mtl_conv2_bias;
#endif
} qwen_tts_vocoder_resunit_t;

/* Vocoder decoder block (SnakeBeta + TransConv + 3 residual units) */
typedef struct {
    float *act_alpha;              /* [in_dim] */
    float *act_beta;               /* [in_dim] */
    float *transconv_weight;       /* [in_dim, out_dim, kernel] */
    float *transconv_bias;         /* [out_dim] */
    qwen_tts_vocoder_resunit_t resunits[3];  /* dilations 1, 3, 9 */
#ifdef USE_METAL
    int mtl_act_alpha, mtl_act_beta;
    int mtl_transconv_weight, mtl_transconv_bias;
#endif
} qwen_tts_vocoder_block_t;

typedef struct {
    /* RVQ */
    qwen_tts_rvq_t rvq;

    /* Pre-conv: CausalConv1d(codebook_dim, latent, kernel=3) */
    float *pre_conv_weight;        /* [latent, codebook_dim, 3] */
    float *pre_conv_bias;          /* [latent] */

    /* Pre-transformer */
    float *transformer_input_proj_weight;   /* [hidden, latent] */
    float *transformer_input_proj_bias;     /* [hidden] */
    float *transformer_output_proj_weight;  /* [latent, hidden] */
    float *transformer_output_proj_bias;    /* [latent] */
    qwen_tts_codec_transformer_layer_t transformer_layers[QWEN_TTS_MAX_CODEC_LAYERS];
    float *transformer_norm;                /* [hidden] */

    /* Upsampling stages (2x ConvNeXt) */
    float *upsample_transconv_weight[2];    /* [latent, latent, factor] */
    float *upsample_transconv_bias[2];      /* [latent] */
    qwen_tts_convnext_block_t upsample_convnext[2];

    /* Vocoder: initial conv */
    float *vocoder_pre_conv_weight;        /* [decoder_dim, latent, 7] */
    float *vocoder_pre_conv_bias;          /* [decoder_dim] */

    /* Vocoder: 4 decoder blocks */
    qwen_tts_vocoder_block_t vocoder_blocks[4];

    /* Vocoder: final output (SnakeBeta + Conv -> 1 channel) */
    float *vocoder_final_act_alpha;        /* [final_dim] */
    float *vocoder_final_act_beta;         /* [final_dim] */
    float *vocoder_final_conv_weight;      /* [1, final_dim, 7] */
    float *vocoder_final_conv_bias;        /* [1] */

#ifdef USE_METAL
    int mtl_input_proj_weight, mtl_input_proj_bias;
    int mtl_output_proj_weight, mtl_output_proj_bias;
    int mtl_transformer_norm;
    int mtl_vocoder_pre_conv_weight, mtl_vocoder_pre_conv_bias;
    int mtl_vocoder_final_act_alpha, mtl_vocoder_final_act_beta;
    int mtl_vocoder_final_conv_weight, mtl_vocoder_final_conv_bias;
#endif
} qwen_tts_codec_decoder_t;

/* ========================================================================
 * Token Callback
 * ======================================================================== */

typedef void (*qwen_tts_progress_cb)(int step, int total, void *userdata);

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    qwen_tts_config_t config;
    qwen_tts_talker_t talker;
    qwen_tts_subtalker_t subtalker;
    qwen_tts_codec_decoder_t codec;

    /* SafeTensors files */
    void *safetensors;              /* multi_safetensors_t* */
    void *codec_safetensors;        /* multi_safetensors_t* for speech_tokenizer */
    char model_dir[512];

    /* Talker KV cache */
    float *talker_kv_k;             /* [layers, max_seq, kv_heads*head_dim] */
    float *talker_kv_v;
    int talker_kv_len;
    int talker_kv_max;

    /* Sub-talker KV cache */
    float *subtalker_kv_k;
    float *subtalker_kv_v;
    int subtalker_kv_len;
    int subtalker_kv_max;

    /* Codec transformer KV cache */
    float *codec_kv_k;
    float *codec_kv_v;
    int codec_kv_len;
    int codec_kv_max;

    /* Persistent talker buffers (single-token generation) */
    float *tk_x, *tk_x_norm, *tk_q, *tk_k, *tk_v;
    float *tk_attn_out, *tk_proj_out;
    float *tk_gate, *tk_up, *tk_ffn_out;
    float *tk_scores;  /* attention scores [talker_kv_max] */
    float *tk_rope_cos, *tk_rope_sin;

    /* Persistent talker prefill buffers */
    float *tk_pref_x, *tk_pref_x_norm, *tk_pref_q, *tk_pref_k, *tk_pref_v;
    float *tk_pref_attn_out, *tk_pref_proj_out;
    float *tk_pref_gate, *tk_pref_gate_up, *tk_pref_ffn_out;
    int tk_pref_cap;

    /* Persistent sub-talker scratch buffers */
    float *st_x, *st_x_norm, *st_q, *st_k, *st_v;
    float *st_attn_out, *st_logits;
    float *st_gate, *st_up;
    float *st_embed, *st_proj_hidden;
    float *st_scores;              /* attention scores [subtalker_kv_max] */
    float *st_rope_cos, *st_rope_sin;
    int st_embed_cap;              /* dims for st_embed */
    int st_scores_cap;             /* entries for st_scores */
    int st_rope_cap;               /* positions for st_rope_* */

    /* RoPE caches */
    float *talker_rope_cos_cache;   /* [max_pos, head_dim*3] for M-RoPE */
    float *talker_rope_sin_cache;
    int talker_rope_cache_cap;

    /* Generation parameters */
    float temperature;
    float subtalker_temperature;
    int top_k;
    int subtalker_top_k;
    float top_p;
    float subtalker_top_p;
    float repetition_penalty;
    int max_new_tokens;
    int fixed_codec_tokens;
    int sample_seed;

    /* Progress callback */
    qwen_tts_progress_cb progress_cb;
    void *progress_cb_userdata;

    /* Performance stats */
    double perf_total_ms;
    double perf_talker_ms;
    double perf_codec_ms;
    int perf_codec_tokens;

#ifdef USE_METAL
    /* Metal scratch buffers for talker single-token path */
    int mtl_x, mtl_x_norm, mtl_q, mtl_k, mtl_v;
    int mtl_attn_out, mtl_gate, mtl_up, mtl_logits;
    int mtl_rope_cos, mtl_rope_sin;
    int mtl_scores;
    int mtl_kv_k, mtl_kv_v;     /* KV cache as Metal buffers */
    int mtl_kv_max;              /* current Metal KV buffer capacity */
    int mtl_sub_scores;
    int mtl_sub_kv_k, mtl_sub_kv_v;
    int mtl_sub_kv_max;
    int mtl_sub_embed;
    int mtl_sub_token;
    int mtl_sub_codes;           /* GPU buffer for all subtalker codes [32 ints] */
    int mtl_sub_rope_cos_all;    /* Full RoPE cos table on GPU [max_seq * head_dim] */
    int mtl_sub_rope_sin_all;    /* Full RoPE sin table on GPU [max_seq * head_dim] */
#endif
} qwen_tts_ctx_t;

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory containing safetensors + config.json */
qwen_tts_ctx_t *qwen_tts_load(const char *model_dir);

/* Free all resources */
void qwen_tts_free(qwen_tts_ctx_t *ctx);

/* Set progress callback */
void qwen_tts_set_progress_callback(qwen_tts_ctx_t *ctx, qwen_tts_progress_cb cb, void *userdata);

/* Generate speech from text using CustomVoice mode.
 * Returns malloc'd float32 PCM audio at 24kHz. Caller must free.
 * out_samples receives the number of samples. */
float *qwen_tts_generate(
    qwen_tts_ctx_t *ctx,
    const char *text,
    const char *speaker,      /* speaker name or NULL for default */
    const char *language,     /* "auto", "chinese", "english", etc. or NULL */
    int *out_samples
);

/* Write PCM float32 audio to WAV file */
int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate);

/* ========================================================================
 * Internal Functions
 * ======================================================================== */

/* Convert BF16 to F32 */
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float result;
    __builtin_memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

/* Talker forward pass - prefill (multiple tokens) */
void qwen_tts_talker_prefill(qwen_tts_ctx_t *ctx, const float *input_embeds, int seq_len);

/* Talker forward pass - single token, returns logits */
void qwen_tts_talker_forward(qwen_tts_ctx_t *ctx, const float *input_embed, float *logits);

/* Sub-talker: generate remaining code groups given talker hidden + first codebook token */
void qwen_tts_subtalker_generate(
    qwen_tts_ctx_t *ctx,
    const float *talker_hidden,  /* [hidden] from talker's last hidden state */
    int first_code,              /* first codebook token from talker */
    int *out_codes               /* [num_code_groups] output - first slot = first_code */
);

/* Codec decoder: convert codec tokens to waveform */
float *qwen_tts_codec_decode(
    qwen_tts_ctx_t *ctx,
    const int *codes,           /* [time_steps, num_quantizers] */
    int time_steps,
    int *out_samples
);

/* Global verbose flag */
extern int qwen_tts_verbose;

#endif /* QWEN_TTS_H */
