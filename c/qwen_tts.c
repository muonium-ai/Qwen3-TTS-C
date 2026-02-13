/*
 * qwen_tts.c - Main API for Qwen3-TTS C inference engine
 *
 * Contains:
 *   - Minimal JSON helpers for config.json parsing
 *   - Config loading (talker + speech_tokenizer)
 *   - Weight loading from SafeTensors (mmap)
 *   - generate() function (embedding construction + autoregressive loop)
 *   - Free / cleanup
 */

#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int qwen_tts_verbose = 0;

/* ========================================================================
 * Timing helpers
 * ======================================================================== */

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ========================================================================
 * Minimal JSON helpers
 *
 * These work on raw JSON text, finding keys at a given nesting level.
 * Not a full parser — just enough for config.json.
 * ======================================================================== */

static void jskip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static void jskip_value(const char **p) {
    jskip_ws(p);
    if (**p == '"') {
        (*p)++;
        while (**p && !(**p == '"' && *(*p-1) != '\\')) (*p)++;
        if (**p == '"') (*p)++;
    } else if (**p == '{') {
        int depth = 1; (*p)++;
        while (**p && depth > 0) {
            if (**p == '{') depth++;
            else if (**p == '}') depth--;
            else if (**p == '"') { (*p)++; while (**p && !(**p == '"' && *(*p-1) != '\\')) (*p)++; }
            (*p)++;
        }
    } else if (**p == '[') {
        int depth = 1; (*p)++;
        while (**p && depth > 0) {
            if (**p == '[') depth++;
            else if (**p == ']') depth--;
            else if (**p == '"') { (*p)++; while (**p && !(**p == '"' && *(*p-1) != '\\')) (*p)++; }
            (*p)++;
        }
    } else {
        while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
    }
}

/* Find a key in the current JSON object level. Returns pointer to the value or NULL. */
static const char *jfind_key(const char *json, const char *key) {
    const char *p = json;
    jskip_ws(&p);
    if (*p != '{') return NULL;
    p++;
    while (1) {
        jskip_ws(&p);
        if (*p == '}' || *p == '\0') return NULL;
        /* Parse key string */
        if (*p != '"') return NULL;
        p++;
        const char *ks = p;
        while (*p && *p != '"') { if (*p == '\\') p++; p++; }
        int klen = (int)(p - ks);
        if (*p == '"') p++;
        jskip_ws(&p);
        if (*p == ':') p++;
        jskip_ws(&p);
        if (klen == (int)strlen(key) && memcmp(ks, key, klen) == 0) {
            return p;  /* Points to start of value */
        }
        jskip_value(&p);
        jskip_ws(&p);
        if (*p == ',') p++;
    }
}

/* Navigate nested path: "talker_config.vocab_size" */
static const char *jfind_path(const char *json, const char *path) {
    char buf[256];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    const char *p = json;
    char *tok = strtok(buf, ".");
    while (tok) {
        p = jfind_key(p, tok);
        if (!p) return NULL;
        char *next = strtok(NULL, ".");
        if (!next) return p;
        tok = next;
    }
    return p;
}

static int jget_int(const char *json, const char *path, int def) {
    const char *v = jfind_path(json, path);
    if (!v) return def;
    return (int)strtol(v, NULL, 10);
}

static float jget_float(const char *json, const char *path, float def) {
    const char *v = jfind_path(json, path);
    if (!v) return def;
    return strtof(v, NULL);
}

/* Get a JSON string value. Returns pointer to out buffer. */
__attribute__((unused))
static const char *jget_str(const char *json, const char *path, char *out, int max_len) {
    const char *v = jfind_path(json, path);
    if (!v || *v != '"') { out[0] = '\0'; return out; }
    v++;
    int i = 0;
    while (*v && *v != '"' && i < max_len - 1) {
        if (*v == '\\') { v++; }
        out[i++] = *v++;
    }
    out[i] = '\0';
    return out;
}

/* Get a JSON integer array value. Returns number of elements parsed. */
static int jget_int_array(const char *json, const char *path, int *out, int max_n) {
    const char *v = jfind_path(json, path);
    if (!v || *v != '[') return 0;
    v++;
    int n = 0;
    while (*v && *v != ']' && n < max_n) {
        jskip_ws(&v);
        if (*v == ']') break;
        out[n++] = (int)strtol(v, (char **)&v, 10);
        jskip_ws(&v);
        if (*v == ',') v++;
    }
    return n;
}

/* Parse speaker map: "spk_id": {"name": [id1, id2, ...], ...} */
static void jparse_speaker_map(const char *json, const char *path,
                                int *n_speakers, char ***names, int **ids) {
    const char *v = jfind_path(json, path);
    if (!v || *v != '{') { *n_speakers = 0; return; }
    const char *p = v + 1;

    /* Count speakers first */
    int count = 0;
    const char *pp = p;
    while (*pp && *pp != '}') {
        jskip_ws(&pp);
        if (*pp == '"') { jskip_value(&pp); jskip_ws(&pp); if (*pp == ':') pp++; jskip_value(&pp); count++; }
        jskip_ws(&pp);
        if (*pp == ',') pp++;
    }

    *n_speakers = count;
    *names = (char **)calloc(count, sizeof(char *));
    *ids = (int *)calloc(count, sizeof(int));

    p = v + 1;
    for (int i = 0; i < count; i++) {
        jskip_ws(&p);
        if (*p != '"') break;
        p++;
        const char *ns = p;
        while (*p && *p != '"') p++;
        int nlen = (int)(p - ns);
        (*names)[i] = (char *)malloc(nlen + 1);
        memcpy((*names)[i], ns, nlen);
        (*names)[i][nlen] = '\0';
        if (*p == '"') p++;
        jskip_ws(&p);
        if (*p == ':') p++;
        jskip_ws(&p);
        /* Value can be an integer or array; take first int from array or the integer */
        if (*p == '[') {
            p++;
            jskip_ws(&p);
            (*ids)[i] = (int)strtol(p, (char **)&p, 10);
            while (*p && *p != ']') p++;
            if (*p == ']') p++;
        } else {
            (*ids)[i] = (int)strtol(p, (char **)&p, 10);
        }
        jskip_ws(&p);
        if (*p == ',') p++;
    }
}

/* Read entire file into malloc'd buffer, null-terminated. */
static char *read_file_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);
    return buf;
}

/* ========================================================================
 * Config Loading
 * ======================================================================== */

static int load_config(qwen_tts_ctx_t *ctx) {
    char path[1024];
    qwen_tts_config_t *cfg = &ctx->config;

    /* ---- Load main config.json ---- */
    snprintf(path, sizeof(path), "%s/config.json", ctx->model_dir);
    char *json = read_file_text(path);
    if (!json) {
        fprintf(stderr, "Error: cannot read %s\n", path);
        return -1;
    }

    /* Talker config */
    cfg->talker_vocab_size      = jget_int(json, "talker_config.vocab_size", QWEN_TTS_TALKER_VOCAB);
    cfg->talker_hidden          = jget_int(json, "talker_config.hidden_size", QWEN_TTS_TALKER_HIDDEN);
    cfg->talker_intermediate    = jget_int(json, "talker_config.intermediate_size", QWEN_TTS_TALKER_INTERMEDIATE);
    cfg->talker_layers          = jget_int(json, "talker_config.num_hidden_layers", QWEN_TTS_TALKER_LAYERS);
    cfg->talker_heads           = jget_int(json, "talker_config.num_attention_heads", QWEN_TTS_TALKER_HEADS);
    cfg->talker_kv_heads        = jget_int(json, "talker_config.num_key_value_heads", QWEN_TTS_TALKER_KV_HEADS);
    cfg->talker_head_dim        = cfg->talker_hidden / cfg->talker_heads;
    cfg->talker_text_hidden     = jget_int(json, "talker_config.text_hidden_size", QWEN_TTS_TALKER_TEXT_HIDDEN);
    cfg->talker_text_vocab      = jget_int(json, "talker_config.text_vocab_size", QWEN_TTS_TALKER_TEXT_VOCAB);
    cfg->num_code_groups        = jget_int(json, "talker_config.num_code_groups", QWEN_TTS_NUM_CODE_GROUPS);
    cfg->talker_rms_norm_eps    = jget_float(json, "talker_config.rms_norm_eps", 1e-6f);
    cfg->talker_rope_theta      = jget_float(json, "talker_config.rope_theta", 10000.0f);

    /* M-RoPE sections */
    cfg->mrope_section[0] = 16; cfg->mrope_section[1] = 16; cfg->mrope_section[2] = 0;
    jget_int_array(json, "talker_config.rope_scaling.mrope_section", cfg->mrope_section, 3);

    /* Sub-talker config */
    cfg->subtalker_vocab_size   = jget_int(json, "talker_config.code_predictor_config.vocab_size", QWEN_TTS_SUBTALKER_VOCAB);
    cfg->subtalker_hidden       = jget_int(json, "talker_config.code_predictor_config.hidden_size", QWEN_TTS_SUBTALKER_HIDDEN);
    cfg->subtalker_intermediate = jget_int(json, "talker_config.code_predictor_config.intermediate_size", QWEN_TTS_SUBTALKER_INTERMEDIATE);
    cfg->subtalker_layers       = jget_int(json, "talker_config.code_predictor_config.num_hidden_layers", QWEN_TTS_SUBTALKER_LAYERS);
    cfg->subtalker_heads        = jget_int(json, "talker_config.code_predictor_config.num_attention_heads", QWEN_TTS_SUBTALKER_HEADS);
    cfg->subtalker_kv_heads     = jget_int(json, "talker_config.code_predictor_config.num_key_value_heads", QWEN_TTS_SUBTALKER_KV_HEADS);
    cfg->subtalker_head_dim     = jget_int(json, "talker_config.code_predictor_config.head_dim", QWEN_TTS_SUBTALKER_HEAD_DIM);

    /* Codec special token IDs */
    cfg->codec_pad_id       = jget_int(json, "talker_config.codec_pad_id", QWEN_TTS_CODEC_PAD);
    cfg->codec_bos_id       = jget_int(json, "talker_config.codec_bos_id", QWEN_TTS_CODEC_BOS);
    cfg->codec_eos_id       = jget_int(json, "talker_config.codec_eos_token_id", QWEN_TTS_CODEC_EOS);
    cfg->codec_nothink_id   = jget_int(json, "talker_config.codec_nothink_id", QWEN_TTS_CODEC_NOTHINK);
    cfg->codec_think_id     = jget_int(json, "talker_config.codec_think_id", QWEN_TTS_CODEC_THINK);
    cfg->codec_think_bos_id = jget_int(json, "talker_config.codec_think_bos_id", QWEN_TTS_CODEC_THINK_BOS);
    cfg->codec_think_eos_id = jget_int(json, "talker_config.codec_think_eos_id", QWEN_TTS_CODEC_THINK_EOS);

    /* Speaker and language maps */
    jparse_speaker_map(json, "talker_config.spk_id", &cfg->n_speakers, &cfg->speaker_names, &cfg->speaker_ids);
    jparse_speaker_map(json, "talker_config.codec_language_id", &cfg->n_languages, &cfg->language_names, &cfg->language_ids);

    free(json);

    /* ---- Load speech_tokenizer config ---- */
    snprintf(path, sizeof(path), "%s/speech_tokenizer/config.json", ctx->model_dir);
    json = read_file_text(path);
    if (!json) {
        fprintf(stderr, "Error: cannot read %s\n", path);
        return -1;
    }

    cfg->codec_num_quantizers   = jget_int(json, "decoder_config.num_quantizers", QWEN_TTS_CODEC_NUM_QUANTIZERS);
    cfg->codec_codebook_size    = jget_int(json, "decoder_config.codebook_size", QWEN_TTS_CODEC_CODEBOOK_SIZE);
    cfg->codec_codebook_dim     = jget_int(json, "decoder_config.codebook_dim", 128);
    cfg->codec_hidden           = jget_int(json, "decoder_config.hidden_size", QWEN_TTS_CODEC_HIDDEN);
    cfg->codec_latent           = jget_int(json, "decoder_config.latent_dim", QWEN_TTS_CODEC_LATENT);
    cfg->codec_layers           = jget_int(json, "decoder_config.num_hidden_layers", QWEN_TTS_CODEC_LAYERS);
    cfg->codec_heads            = jget_int(json, "decoder_config.num_attention_heads", QWEN_TTS_CODEC_HEADS);
    cfg->codec_kv_heads         = jget_int(json, "decoder_config.num_key_value_heads", QWEN_TTS_CODEC_KV_HEADS);
    cfg->codec_intermediate     = jget_int(json, "decoder_config.intermediate_size", QWEN_TTS_CODEC_INTERMEDIATE);
    cfg->codec_sliding_window   = jget_int(json, "decoder_config.sliding_window", QWEN_TTS_CODEC_SLIDING_WINDOW);
    cfg->codec_decoder_dim      = jget_int(json, "decoder_config.decoder_dim", QWEN_TTS_CODEC_DECODER_DIM);
    cfg->codec_rms_norm_eps     = jget_float(json, "decoder_config.rms_norm_eps", 1e-5f);
    cfg->codec_layer_scale      = jget_float(json, "decoder_config.layer_scale_initial_scale", 0.01f);

    int rates[4] = {8, 5, 4, 3};
    jget_int_array(json, "decoder_config.upsample_rates", rates, 4);
    memcpy(cfg->codec_upsample_rates, rates, sizeof(rates));

    int ratios[2] = {2, 2};
    jget_int_array(json, "decoder_config.upsampling_ratios", ratios, 2);
    memcpy(cfg->codec_upsampling_ratios, ratios, sizeof(ratios));

    free(json);

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "Config loaded:\n");
        fprintf(stderr, "  Talker: %d layers, hidden=%d, heads=%d/%d, head_dim=%d\n",
                cfg->talker_layers, cfg->talker_hidden, cfg->talker_heads, cfg->talker_kv_heads, cfg->talker_head_dim);
        fprintf(stderr, "  Sub-talker: %d layers, hidden=%d, heads=%d/%d, head_dim=%d\n",
                cfg->subtalker_layers, cfg->subtalker_hidden, cfg->subtalker_heads, cfg->subtalker_kv_heads, cfg->subtalker_head_dim);
        fprintf(stderr, "  Codec: %d layers, hidden=%d, codebook_dim=%d, decoder_dim=%d\n",
                cfg->codec_layers, cfg->codec_hidden, cfg->codec_codebook_dim, cfg->codec_decoder_dim);
        fprintf(stderr, "  M-RoPE sections: [%d, %d, %d]\n",
                cfg->mrope_section[0], cfg->mrope_section[1], cfg->mrope_section[2]);
        fprintf(stderr, "  Speakers: %d, Languages: %d\n", cfg->n_speakers, cfg->n_languages);
    }

    return 0;
}

/* ========================================================================
 * Weight Loading Helpers
 * ======================================================================== */

/* Convenience macros for loading weights from safetensors */
#define GET_BF16(ms, name)      (uint16_t *)multi_safetensors_get_bf16(ms, name, NULL, NULL)
#define GET_F32(ms, name)       (float *)multi_safetensors_get_f32(ms, name, NULL, NULL)
#define LOAD_F32(ms, name)      multi_safetensors_load_f32(ms, name, NULL, NULL)

#define GET_BF16_CHECK(dst, ms, name) do { \
    dst = GET_BF16(ms, name); \
    if (!dst && qwen_tts_verbose >= 2) fprintf(stderr, "  Warning: tensor not found: %s\n", name); \
} while(0)

#define GET_F32_CHECK(dst, ms, name) do { \
    dst = GET_F32(ms, name); \
    if (!dst && qwen_tts_verbose >= 2) fprintf(stderr, "  Warning: tensor not found: %s\n", name); \
} while(0)

#define LOAD_F32_CHECK(dst, ms, name) do { \
    dst = LOAD_F32(ms, name); \
    if (!dst && qwen_tts_verbose >= 2) fprintf(stderr, "  Warning: tensor not found: %s\n", name); \
} while(0)

/* ========================================================================
 * Load Talker Weights
 * ======================================================================== */

static void load_talker_weights(qwen_tts_ctx_t *ctx, const multi_safetensors_t *ms) {
    qwen_tts_config_t *cfg = &ctx->config;
    char name[512];

    if (qwen_tts_verbose >= 1) fprintf(stderr, "Loading talker weights...\n");

    /* Embeddings */
    GET_BF16_CHECK(ctx->talker.codec_embedding_bf16, ms, "talker.model.codec_embedding.weight");
    GET_BF16_CHECK(ctx->talker.text_embedding_bf16, ms, "talker.model.text_embedding.weight");

    /* Text projection MLP */
    GET_BF16_CHECK(ctx->talker.text_proj_fc1_bf16, ms, "talker.text_projection.linear_fc1.weight");
    LOAD_F32_CHECK(ctx->talker.text_proj_fc1_bias, ms, "talker.text_projection.linear_fc1.bias");
    GET_BF16_CHECK(ctx->talker.text_proj_fc2_bf16, ms, "talker.text_projection.linear_fc2.weight");
    LOAD_F32_CHECK(ctx->talker.text_proj_fc2_bias, ms, "talker.text_projection.linear_fc2.bias");

    /* Transformer layers */
    for (int i = 0; i < cfg->talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];

        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_proj.weight", i);
        GET_BF16_CHECK(l->wq_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_proj.weight", i);
        GET_BF16_CHECK(l->wk_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.v_proj.weight", i);
        GET_BF16_CHECK(l->wv_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.o_proj.weight", i);
        GET_BF16_CHECK(l->wo_bf16, ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.q_norm.weight", i);
        LOAD_F32_CHECK(l->q_norm_weight, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.self_attn.k_norm.weight", i);
        LOAD_F32_CHECK(l->k_norm_weight, ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.input_layernorm.weight", i);
        LOAD_F32_CHECK(l->input_norm, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.post_attention_layernorm.weight", i);
        LOAD_F32_CHECK(l->post_attn_norm, ms, name);

        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.gate_proj.weight", i);
        GET_BF16_CHECK(l->gate_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.up_proj.weight", i);
        GET_BF16_CHECK(l->up_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.model.layers.%d.mlp.down_proj.weight", i);
        GET_BF16_CHECK(l->down_bf16, ms, name);
    }

    /* Final norm */
    LOAD_F32_CHECK(ctx->talker.norm, ms, "talker.model.norm.weight");

    /* Codec head */
    GET_BF16_CHECK(ctx->talker.codec_head_bf16, ms, "talker.codec_head.weight");

    if (qwen_tts_verbose >= 1) fprintf(stderr, "  Talker: %d layers loaded\n", cfg->talker_layers);
}

/* ========================================================================
 * Load Sub-Talker (Code Predictor) Weights
 * ======================================================================== */

static void load_subtalker_weights(qwen_tts_ctx_t *ctx, const multi_safetensors_t *ms) {
    qwen_tts_config_t *cfg = &ctx->config;
    char name[512];

    if (qwen_tts_verbose >= 1) fprintf(stderr, "Loading sub-talker weights...\n");

    /* 31 codec embeddings (groups 1-31) */
    for (int g = 0; g < cfg->num_code_groups - 1; g++) {
        snprintf(name, sizeof(name), "talker.code_predictor.model.codec_embedding.%d.weight", g);
        GET_BF16_CHECK(ctx->subtalker.codec_embeddings_bf16[g], ms, name);
    }

    /* Input projection */
    GET_BF16_CHECK(ctx->subtalker.input_proj_bf16, ms, "talker.code_predictor.small_to_mtp_projection.weight");
    LOAD_F32_CHECK(ctx->subtalker.input_proj_bias, ms, "talker.code_predictor.small_to_mtp_projection.bias");

    /* Transformer layers */
    for (int i = 0; i < cfg->subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.q_proj.weight", i);
        GET_BF16_CHECK(l->wq_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.k_proj.weight", i);
        GET_BF16_CHECK(l->wk_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.v_proj.weight", i);
        GET_BF16_CHECK(l->wv_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.o_proj.weight", i);
        GET_BF16_CHECK(l->wo_bf16, ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.q_norm.weight", i);
        LOAD_F32_CHECK(l->q_norm_weight, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.self_attn.k_norm.weight", i);
        LOAD_F32_CHECK(l->k_norm_weight, ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.input_layernorm.weight", i);
        LOAD_F32_CHECK(l->input_norm, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.post_attention_layernorm.weight", i);
        LOAD_F32_CHECK(l->post_attn_norm, ms, name);

        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.gate_proj.weight", i);
        GET_BF16_CHECK(l->gate_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.up_proj.weight", i);
        GET_BF16_CHECK(l->up_bf16, ms, name);
        snprintf(name, sizeof(name), "talker.code_predictor.model.layers.%d.mlp.down_proj.weight", i);
        GET_BF16_CHECK(l->down_bf16, ms, name);
    }

    /* Final norm */
    LOAD_F32_CHECK(ctx->subtalker.norm, ms, "talker.code_predictor.model.norm.weight");

    /* 31 LM heads */
    for (int g = 0; g < cfg->num_code_groups - 1; g++) {
        snprintf(name, sizeof(name), "talker.code_predictor.lm_head.%d.weight", g);
        GET_BF16_CHECK(ctx->subtalker.lm_heads_bf16[g], ms, name);
    }

    if (qwen_tts_verbose >= 1) fprintf(stderr, "  Sub-talker: %d layers loaded\n", cfg->subtalker_layers);
}

/* ========================================================================
 * Load Codec Decoder (Speech Tokenizer) Weights
 * ======================================================================== */

static void load_codec_weights(qwen_tts_ctx_t *ctx, const multi_safetensors_t *ms) {
    qwen_tts_config_t *cfg = &ctx->config;
    qwen_tts_codec_decoder_t *codec = &ctx->codec;
    char name[512];

    if (qwen_tts_verbose >= 1) fprintf(stderr, "Loading codec decoder weights...\n");

    /* ---- RVQ: SplitResidualVectorQuantizer ---- */

    /* Semantic codebook (quantizer 0): rvq_first has 1 VQ layer */
    LOAD_F32_CHECK(codec->rvq.semantic_codebooks[0].cluster_usage, ms,
                   "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage");
    LOAD_F32_CHECK(codec->rvq.semantic_codebooks[0].embedding_sum, ms,
                   "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum");

    /* Semantic output_proj: Conv1d(vq_dim, codebook_dim, 1) */
    LOAD_F32_CHECK(codec->rvq.semantic_output_proj, ms,
                   "decoder.quantizer.rvq_first.output_proj.weight");

    /* Acoustic codebooks (quantizers 1-15): rvq_rest has 15 VQ layers */
    for (int q = 0; q < cfg->codec_num_quantizers - 1; q++) {
        snprintf(name, sizeof(name), "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.cluster_usage", q);
        LOAD_F32_CHECK(codec->rvq.acoustic_codebooks[q].cluster_usage, ms, name);
        snprintf(name, sizeof(name), "decoder.quantizer.rvq_rest.vq.layers.%d._codebook.embedding_sum", q);
        LOAD_F32_CHECK(codec->rvq.acoustic_codebooks[q].embedding_sum, ms, name);
    }

    LOAD_F32_CHECK(codec->rvq.acoustic_output_proj, ms,
                   "decoder.quantizer.rvq_rest.output_proj.weight");

    /* ---- Pre-conv ---- */
    LOAD_F32_CHECK(codec->pre_conv_weight, ms, "decoder.pre_conv.conv.weight");
    LOAD_F32_CHECK(codec->pre_conv_bias, ms, "decoder.pre_conv.conv.bias");

    /* ---- Transformer ---- */
    LOAD_F32_CHECK(codec->transformer_input_proj_weight, ms, "decoder.pre_transformer.input_proj.weight");
    LOAD_F32_CHECK(codec->transformer_input_proj_bias, ms, "decoder.pre_transformer.input_proj.bias");
    LOAD_F32_CHECK(codec->transformer_output_proj_weight, ms, "decoder.pre_transformer.output_proj.weight");
    LOAD_F32_CHECK(codec->transformer_output_proj_bias, ms, "decoder.pre_transformer.output_proj.bias");
    LOAD_F32_CHECK(codec->transformer_norm, ms, "decoder.pre_transformer.norm.weight");

    for (int i = 0; i < cfg->codec_layers; i++) {
        qwen_tts_codec_transformer_layer_t *l = &codec->transformer_layers[i];

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.input_layernorm.weight", i);
        LOAD_F32_CHECK(l->input_norm, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.post_attention_layernorm.weight", i);
        LOAD_F32_CHECK(l->post_attn_norm, ms, name);

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn_layer_scale.scale", i);
        LOAD_F32_CHECK(l->attn_layer_scale, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp_layer_scale.scale", i);
        LOAD_F32_CHECK(l->mlp_layer_scale, ms, name);

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.q_proj.weight", i);
        LOAD_F32_CHECK(l->wq, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.k_proj.weight", i);
        LOAD_F32_CHECK(l->wk, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.v_proj.weight", i);
        LOAD_F32_CHECK(l->wv, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.self_attn.o_proj.weight", i);
        LOAD_F32_CHECK(l->wo, ms, name);

        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.gate_proj.weight", i);
        LOAD_F32_CHECK(l->gate, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.up_proj.weight", i);
        LOAD_F32_CHECK(l->up, ms, name);
        snprintf(name, sizeof(name), "decoder.pre_transformer.layers.%d.mlp.down_proj.weight", i);
        LOAD_F32_CHECK(l->down, ms, name);
    }

    /* ---- Upsample stages ---- */
    for (int s = 0; s < 2; s++) {
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.weight", s);
        LOAD_F32_CHECK(codec->upsample_transconv_weight[s], ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.0.conv.bias", s);
        LOAD_F32_CHECK(codec->upsample_transconv_bias[s], ms, name);

        qwen_tts_convnext_block_t *cn = &codec->upsample_convnext[s];
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.weight", s);
        LOAD_F32_CHECK(cn->dwconv_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.dwconv.conv.bias", s);
        LOAD_F32_CHECK(cn->dwconv_bias, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.weight", s);
        LOAD_F32_CHECK(cn->norm_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.norm.bias", s);
        LOAD_F32_CHECK(cn->norm_bias, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.weight", s);
        LOAD_F32_CHECK(cn->pwconv1_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv1.bias", s);
        LOAD_F32_CHECK(cn->pwconv1_bias, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.weight", s);
        LOAD_F32_CHECK(cn->pwconv2_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.pwconv2.bias", s);
        LOAD_F32_CHECK(cn->pwconv2_bias, ms, name);
        snprintf(name, sizeof(name), "decoder.upsample.%d.1.gamma", s);
        LOAD_F32_CHECK(cn->gamma, ms, name);
    }

    /* ---- Vocoder ---- */
    /* decoder.decoder.[0..6] in Python's ModuleList:
     *   [0] = initial CausalConv (latent -> decoder_dim, k=7)
     *   [1..4] = DecoderBlock (each has .block = [SnakeBeta, TransConv, ResUnit, ResUnit, ResUnit])
     *   [5] = final SnakeBeta
     *   [6] = final CausalConv (out_dim -> 1, k=7)
     */
    LOAD_F32_CHECK(codec->vocoder_pre_conv_weight, ms, "decoder.decoder.0.conv.weight");
    LOAD_F32_CHECK(codec->vocoder_pre_conv_bias, ms, "decoder.decoder.0.conv.bias");

    for (int b = 0; b < 4; b++) {
        qwen_tts_vocoder_block_t *vb = &codec->vocoder_blocks[b];
        int idx = b + 1;  /* Python module index: decoder.decoder.{b+1} */

        /* SnakeBeta activation at block[0] */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.alpha", idx);
        LOAD_F32_CHECK(vb->act_alpha, ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.0.beta", idx);
        LOAD_F32_CHECK(vb->act_beta, ms, name);

        /* Transposed conv at block[1] */
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.weight", idx);
        LOAD_F32_CHECK(vb->transconv_weight, ms, name);
        snprintf(name, sizeof(name), "decoder.decoder.%d.block.1.conv.bias", idx);
        LOAD_F32_CHECK(vb->transconv_bias, ms, name);

        /* 3 residual units at block[2], block[3], block[4] */
        for (int r = 0; r < 3; r++) {
            qwen_tts_vocoder_resunit_t *ru = &vb->resunits[r];
            int ridx = r + 2;

            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.alpha", idx, ridx);
            LOAD_F32_CHECK(ru->act1_alpha, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act1.beta", idx, ridx);
            LOAD_F32_CHECK(ru->act1_beta, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.weight", idx, ridx);
            LOAD_F32_CHECK(ru->conv1_weight, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv1.conv.bias", idx, ridx);
            LOAD_F32_CHECK(ru->conv1_bias, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.alpha", idx, ridx);
            LOAD_F32_CHECK(ru->act2_alpha, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.act2.beta", idx, ridx);
            LOAD_F32_CHECK(ru->act2_beta, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.weight", idx, ridx);
            LOAD_F32_CHECK(ru->conv2_weight, ms, name);
            snprintf(name, sizeof(name), "decoder.decoder.%d.block.%d.conv2.conv.bias", idx, ridx);
            LOAD_F32_CHECK(ru->conv2_bias, ms, name);
        }
    }

    /* Final SnakeBeta + Conv (decoder.decoder.5 and decoder.decoder.6) */
    LOAD_F32_CHECK(codec->vocoder_final_act_alpha, ms, "decoder.decoder.5.alpha");
    LOAD_F32_CHECK(codec->vocoder_final_act_beta, ms, "decoder.decoder.5.beta");
    LOAD_F32_CHECK(codec->vocoder_final_conv_weight, ms, "decoder.decoder.6.conv.weight");
    LOAD_F32_CHECK(codec->vocoder_final_conv_bias, ms, "decoder.decoder.6.conv.bias");

    if (qwen_tts_verbose >= 1) fprintf(stderr, "  Codec decoder loaded\n");
}

/* ========================================================================
 * Text projection helper
 *
 * Projects text embeddings: text_hidden -> text_hidden (SiLU) -> hidden
 * ======================================================================== */

static void text_projection(qwen_tts_ctx_t *ctx, const float *text_embed,
                             float *out, int text_hidden, int hidden) {
    float *fc1_out = (float *)malloc(text_hidden * sizeof(float));
    kernel_matvec_bf16(fc1_out, ctx->talker.text_proj_fc1_bf16, text_embed, text_hidden, text_hidden);
    if (ctx->talker.text_proj_fc1_bias)
        kernel_add_inplace(fc1_out, ctx->talker.text_proj_fc1_bias, text_hidden);
    kernel_silu_inplace(fc1_out, text_hidden);
    kernel_matvec_bf16(out, ctx->talker.text_proj_fc2_bf16, fc1_out, hidden, text_hidden);
    if (ctx->talker.text_proj_fc2_bias)
        kernel_add_inplace(out, ctx->talker.text_proj_fc2_bias, hidden);
    free(fc1_out);
}

/* ========================================================================
 * Embed a text token: text_embedding -> text_projection
 * ======================================================================== */

static void embed_text_token(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int text_hidden = ctx->config.talker_text_hidden;
    int hidden = ctx->config.talker_hidden;
    float *text_embed = (float *)malloc(text_hidden * sizeof(float));
    kernel_bf16_to_f32(text_embed, ctx->talker.text_embedding_bf16 + (size_t)token_id * text_hidden, text_hidden);
    text_projection(ctx, text_embed, out, text_hidden, hidden);
    free(text_embed);
}

/* ========================================================================
 * Embed a codec token: lookup from codec_embedding
 * ======================================================================== */

static void embed_codec_token(qwen_tts_ctx_t *ctx, int token_id, float *out) {
    int hidden = ctx->config.talker_hidden;
    kernel_bf16_to_f32(out, ctx->talker.codec_embedding_bf16 + (size_t)token_id * hidden, hidden);
}

/* ========================================================================
 * Load Model
 * ======================================================================== */

qwen_tts_ctx_t *qwen_tts_load(const char *model_dir) {
    double t0 = time_ms();

    qwen_tts_ctx_t *ctx = (qwen_tts_ctx_t *)calloc(1, sizeof(qwen_tts_ctx_t));
    if (!ctx) return NULL;

    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);

    /* Set default generation parameters */
    ctx->temperature = 0.9f;
    ctx->subtalker_temperature = 0.9f;
    ctx->top_k = 50;
    ctx->subtalker_top_k = 50;
    ctx->top_p = 1.0f;
    ctx->subtalker_top_p = 1.0f;
    ctx->repetition_penalty = 1.05f;
    ctx->max_new_tokens = 4096;

    /* Load config */
    if (load_config(ctx) != 0) {
        free(ctx);
        return NULL;
    }

    /* Open talker safetensors */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "Error: cannot open model safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    load_talker_weights(ctx, ms);
    load_subtalker_weights(ctx, ms);

    /* Open codec decoder safetensors */
    char codec_dir[1024];
    snprintf(codec_dir, sizeof(codec_dir), "%s/speech_tokenizer", model_dir);
    multi_safetensors_t *cms = multi_safetensors_open(codec_dir);
    if (!cms) {
        fprintf(stderr, "Error: cannot open speech_tokenizer safetensors in %s\n", codec_dir);
        /* Continue without codec (can still generate tokens) */
    } else {
        ctx->codec_safetensors = cms;
        load_codec_weights(ctx, cms);
    }

    kernel_init();

    double t1 = time_ms();
    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Model loaded in %.1f ms\n", t1 - t0);

    return ctx;
}

/* ========================================================================
 * Free
 * ======================================================================== */

void qwen_tts_free(qwen_tts_ctx_t *ctx) {
    if (!ctx) return;

    /* Close safetensors (this frees mmap'd weights) */
    if (ctx->safetensors) multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
    if (ctx->codec_safetensors) multi_safetensors_close((multi_safetensors_t *)ctx->codec_safetensors);

    /* Free codec weights that were LOAD_F32'd (malloc'd copies) */
    /* RVQ */
    for (int i = 0; i < 1; i++) {
        free(ctx->codec.rvq.semantic_codebooks[i].cluster_usage);
        free(ctx->codec.rvq.semantic_codebooks[i].embedding_sum);
    }
    for (int i = 0; i < ctx->config.codec_num_quantizers - 1; i++) {
        free(ctx->codec.rvq.acoustic_codebooks[i].cluster_usage);
        free(ctx->codec.rvq.acoustic_codebooks[i].embedding_sum);
    }
    free(ctx->codec.rvq.semantic_output_proj);
    free(ctx->codec.rvq.acoustic_output_proj);
    free(ctx->codec.pre_conv_weight);
    free(ctx->codec.pre_conv_bias);
    free(ctx->codec.transformer_input_proj_weight);
    free(ctx->codec.transformer_input_proj_bias);
    free(ctx->codec.transformer_output_proj_weight);
    free(ctx->codec.transformer_output_proj_bias);
    free(ctx->codec.transformer_norm);

    for (int i = 0; i < ctx->config.codec_layers; i++) {
        qwen_tts_codec_transformer_layer_t *l = &ctx->codec.transformer_layers[i];
        free(l->input_norm); free(l->post_attn_norm);
        free(l->attn_layer_scale); free(l->mlp_layer_scale);
        free(l->wq); free(l->wk); free(l->wv); free(l->wo);
        free(l->gate); free(l->up); free(l->down);
    }

    for (int s = 0; s < 2; s++) {
        free(ctx->codec.upsample_transconv_weight[s]);
        free(ctx->codec.upsample_transconv_bias[s]);
        qwen_tts_convnext_block_t *cn = &ctx->codec.upsample_convnext[s];
        free(cn->dwconv_weight); free(cn->dwconv_bias);
        free(cn->norm_weight); free(cn->norm_bias);
        free(cn->pwconv1_weight); free(cn->pwconv1_bias);
        free(cn->pwconv2_weight); free(cn->pwconv2_bias);
        free(cn->gamma);
    }

    free(ctx->codec.vocoder_pre_conv_weight); free(ctx->codec.vocoder_pre_conv_bias);
    for (int b = 0; b < 4; b++) {
        qwen_tts_vocoder_block_t *vb = &ctx->codec.vocoder_blocks[b];
        free(vb->act_alpha); free(vb->act_beta);
        free(vb->transconv_weight); free(vb->transconv_bias);
        for (int r = 0; r < 3; r++) {
            qwen_tts_vocoder_resunit_t *ru = &vb->resunits[r];
            free(ru->act1_alpha); free(ru->act1_beta);
            free(ru->conv1_weight); free(ru->conv1_bias);
            free(ru->act2_alpha); free(ru->act2_beta);
            free(ru->conv2_weight); free(ru->conv2_bias);
        }
    }
    free(ctx->codec.vocoder_final_act_alpha); free(ctx->codec.vocoder_final_act_beta);
    free(ctx->codec.vocoder_final_conv_weight); free(ctx->codec.vocoder_final_conv_bias);

    /* Free talker LOAD_F32'd weights (norms + biases) */
    free(ctx->talker.text_proj_fc1_bias);
    free(ctx->talker.text_proj_fc2_bias);
    free(ctx->talker.norm);
    for (int i = 0; i < ctx->config.talker_layers; i++) {
        qwen_tts_talker_layer_t *l = &ctx->talker.layers[i];
        free(l->q_norm_weight); free(l->k_norm_weight);
        free(l->input_norm); free(l->post_attn_norm);
    }

    /* Free subtalker LOAD_F32'd weights (norms + biases) */
    free(ctx->subtalker.input_proj_bias);
    free(ctx->subtalker.norm);
    for (int i = 0; i < ctx->config.subtalker_layers; i++) {
        qwen_tts_subtalker_layer_t *l = &ctx->subtalker.layers[i];
        free(l->q_norm_weight); free(l->k_norm_weight);
        free(l->input_norm); free(l->post_attn_norm);
    }

    /* Free KV caches and scratch buffers */
    free(ctx->talker_kv_k); free(ctx->talker_kv_v);
    free(ctx->subtalker_kv_k); free(ctx->subtalker_kv_v);
    free(ctx->codec_kv_k); free(ctx->codec_kv_v);
    free(ctx->tk_x); free(ctx->tk_x_norm);
    free(ctx->tk_q); free(ctx->tk_k); free(ctx->tk_v);
    free(ctx->tk_attn_out); free(ctx->tk_proj_out);
    free(ctx->tk_gate); free(ctx->tk_up); free(ctx->tk_ffn_out);
    free(ctx->tk_scores);
    free(ctx->tk_rope_cos); free(ctx->tk_rope_sin);
    free(ctx->tk_pref_x); free(ctx->tk_pref_x_norm);
    free(ctx->tk_pref_q); free(ctx->tk_pref_k); free(ctx->tk_pref_v);
    free(ctx->tk_pref_attn_out); free(ctx->tk_pref_proj_out);
    free(ctx->tk_pref_gate); free(ctx->tk_pref_gate_up); free(ctx->tk_pref_ffn_out);
    free(ctx->talker_rope_cos_cache); free(ctx->talker_rope_sin_cache);

    /* Free config maps */
    for (int i = 0; i < ctx->config.n_speakers; i++) free(ctx->config.speaker_names[i]);
    free(ctx->config.speaker_names); free(ctx->config.speaker_ids);
    for (int i = 0; i < ctx->config.n_languages; i++) free(ctx->config.language_names[i]);
    free(ctx->config.language_names); free(ctx->config.language_ids);

    free(ctx);
}

void qwen_tts_set_progress_callback(qwen_tts_ctx_t *ctx, qwen_tts_progress_cb cb, void *userdata) {
    ctx->progress_cb = cb;
    ctx->progress_cb_userdata = userdata;
}

/* ========================================================================
 * Generate - CustomVoice Mode
 *
 * Builds the embedding sequence, runs talker prefill + autoregressive
 * generation, then decodes codec tokens to waveform.
 *
 * Input text_token_ids: pre-tokenized text in chat format:
 *   [im_start, assistant_id, \n, TEXT..., im_end, \n, im_start, assistant_id, \n]
 * ======================================================================== */

float *qwen_tts_generate(
    qwen_tts_ctx_t *ctx,
    const char *text,         /* unused for now, pass NULL */
    const char *speaker,
    const char *language,
    int *out_samples
) {
    /* For now, we require pre-tokenized IDs stored in ctx.
     * This function implements the full generate flow with hardcoded token IDs.
     *
     * TODO: Add BPE tokenizer for direct text input.
     *
     * For demonstration, we accept text as a comma-separated list of token IDs.
     */

    /* Parse text as comma-separated token IDs */
    int *text_tokens = NULL;
    int n_text_tokens = 0;

    if (text) {
        /* Count tokens */
        const char *p = text;
        int count = 1;
        while (*p) { if (*p == ',') count++; p++; }

        text_tokens = (int *)malloc(count * sizeof(int));
        p = text;
        while (*p) {
            while (*p == ' ' || *p == ',') p++;
            if (*p == '\0') break;
            text_tokens[n_text_tokens++] = (int)strtol(p, (char **)&p, 10);
        }
    }

    if (n_text_tokens < 8) {
        fprintf(stderr, "Error: need at least 8 text tokens (chat template format)\n");
        free(text_tokens);
        *out_samples = 0;
        return NULL;
    }

    qwen_tts_config_t *cfg = &ctx->config;
    int hidden = cfg->talker_hidden;
    int num_groups = cfg->num_code_groups;

    double t_start = time_ms();

    /* ---- Look up speaker and language IDs ---- */
    int speaker_codec_id = -1;
    if (speaker && strlen(speaker) > 0) {
        for (int i = 0; i < cfg->n_speakers; i++) {
            if (strcasecmp(cfg->speaker_names[i], speaker) == 0) {
                speaker_codec_id = cfg->speaker_ids[i];
                break;
            }
        }
        if (speaker_codec_id < 0) {
            fprintf(stderr, "Warning: speaker '%s' not found, using no speaker embedding\n", speaker);
        }
    }

    int language_codec_id = -1;
    if (language && strlen(language) > 0 && strcasecmp(language, "auto") != 0) {
        for (int i = 0; i < cfg->n_languages; i++) {
            if (strcasecmp(cfg->language_names[i], language) == 0) {
                language_codec_id = cfg->language_ids[i];
                break;
            }
        }
        if (language_codec_id < 0) {
            fprintf(stderr, "Warning: language '%s' not found\n", language);
        }
    }

    /* ---- Build prefix embedding sequence ---- */

    /* Input format: [im_start, assistant, \n, TEXT..., im_end, \n, im_start, assistant, \n]
     * Positions: [0:3] = role, [3:-5] = content text, [-5:] = trailing template
     *
     * role_tokens = text_proj(text_embed(input[0:3]))   -- 3 tokens
     * first_text = text_proj(text_embed(input[3]))       -- 1 token
     * remaining_text = text_proj(text_embed(input[4:-5])) -- variable
     * trailing_text = [remaining_text..., tts_eos_embed]
     */

    int n_content = n_text_tokens - 8;  /* text tokens minus template (3 + 5) */
    if (n_content < 0) n_content = 0;

    /* Build codec prefix tokens */
    int codec_prefix[8];
    int n_codec_prefix = 0;

    if (language_codec_id < 0) {
        /* No language specified: nothink, think_bos, think_eos */
        codec_prefix[n_codec_prefix++] = cfg->codec_nothink_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_bos_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_eos_id;
    } else {
        /* Language specified: think, think_bos, language_id, think_eos */
        codec_prefix[n_codec_prefix++] = cfg->codec_think_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_bos_id;
        codec_prefix[n_codec_prefix++] = language_codec_id;
        codec_prefix[n_codec_prefix++] = cfg->codec_think_eos_id;
    }

    if (speaker_codec_id >= 0) {
        codec_prefix[n_codec_prefix++] = speaker_codec_id;
    }
    codec_prefix[n_codec_prefix++] = cfg->codec_pad_id;
    codec_prefix[n_codec_prefix++] = cfg->codec_bos_id;

    /* Total prefill length:
     *   3 (role) + (n_codec_prefix - 1) (tts_pad/bos + codec without last) + 1 (first_text + codec_bos)
     *   = 3 + n_codec_prefix
     */
    int prefill_len = 3 + n_codec_prefix;

    float *input_embeds = (float *)calloc((size_t)prefill_len * hidden, sizeof(float));

    /* 1. Role tokens: text_proj(text_embed(role[0:3])) */
    for (int i = 0; i < 3; i++) {
        embed_text_token(ctx, text_tokens[i], input_embeds + i * hidden);
    }

    /* 2. Pad/bos section:
     *   Positions 3..3+n_codec_prefix-2:
     *     text side: tts_pad_embed (for all but last), tts_bos_embed (for the last in this group)
     *     codec side: codec_embed(codec_prefix[0..n_codec_prefix-2])
     *     These are added element-wise.
     */
    float *tts_pad_proj = (float *)malloc(hidden * sizeof(float));
    float *tts_bos_proj = (float *)malloc(hidden * sizeof(float));
    float *tts_eos_proj = (float *)malloc(hidden * sizeof(float));
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_PAD, tts_pad_proj);
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_BOS, tts_bos_proj);
    embed_text_token(ctx, QWEN_TTS_TOKEN_TTS_EOS, tts_eos_proj);

    for (int i = 0; i < n_codec_prefix - 1; i++) {
        float *dst = input_embeds + (3 + i) * hidden;
        /* Text part: tts_pad for all except the last which gets tts_bos */
        if (i < n_codec_prefix - 2) {
            memcpy(dst, tts_pad_proj, hidden * sizeof(float));
        } else {
            memcpy(dst, tts_bos_proj, hidden * sizeof(float));
        }
        /* Codec part: add codec_embed(codec_prefix[i]) */
        float *codec_emb = (float *)malloc(hidden * sizeof(float));
        embed_codec_token(ctx, codec_prefix[i], codec_emb);
        kernel_add_inplace(dst, codec_emb, hidden);
        free(codec_emb);
    }

    /* 3. First text token + codec_bos:
     *   text_proj(text_embed(text_tokens[3])) + codec_embed(codec_bos)
     */
    {
        int pos = 3 + n_codec_prefix - 1;
        float *dst = input_embeds + pos * hidden;
        embed_text_token(ctx, text_tokens[3], dst);
        float *codec_emb = (float *)malloc(hidden * sizeof(float));
        embed_codec_token(ctx, cfg->codec_bos_id, codec_emb);
        kernel_add_inplace(dst, codec_emb, hidden);
        free(codec_emb);
    }

    /* Build trailing text embeddings (remaining text + tts_eos) */
    int n_trailing = (n_text_tokens - 4 - 5) + 1;  /* remaining text + eos */
    if (n_trailing < 1) n_trailing = 1;
    float *trailing_text = (float *)calloc((size_t)n_trailing * hidden, sizeof(float));
    for (int i = 0; i < n_trailing - 1; i++) {
        embed_text_token(ctx, text_tokens[4 + i], trailing_text + i * hidden);
    }
    memcpy(trailing_text + (n_trailing - 1) * hidden, tts_eos_proj, hidden * sizeof(float));

    /* ---- Prefill ---- */
    double t_prefill = time_ms();

    /* Reset KV cache */
    ctx->talker_kv_len = 0;
    qwen_tts_talker_prefill(ctx, input_embeds, prefill_len);

    double t_prefill_done = time_ms();
    if (qwen_tts_verbose >= 1)
        fprintf(stderr, "Prefill: %d tokens in %.1f ms\n", prefill_len, t_prefill_done - t_prefill);

    free(input_embeds);

    /* ---- Autoregressive generation ---- */
    int max_tokens = ctx->max_new_tokens;
    int *all_codes = (int *)calloc((size_t)max_tokens * num_groups, sizeof(int));
    int *generated_tokens = (int *)calloc(max_tokens, sizeof(int));
    int n_generated = 0;

    float *logits = (float *)malloc(cfg->talker_vocab_size * sizeof(float));
    float *next_embed = (float *)malloc(hidden * sizeof(float));
    float rng_state = 42.0f;

    /* Suppress tokens: [vocab-1024, vocab) except EOS */
    int suppress_start = cfg->talker_vocab_size - 1024;
    int *suppress_tokens = (int *)malloc(1024 * sizeof(int));
    int n_suppress = 0;
    for (int i = suppress_start; i < cfg->talker_vocab_size; i++) {
        if (i != cfg->codec_eos_id) suppress_tokens[n_suppress++] = i;
    }

    double t_gen = time_ms();

    for (int step = 0; step < max_tokens; step++) {
        /* If this is the first step after prefill, use the last hidden state from prefill */
        if (step == 0) {
            /* The prefill already stored the last hidden state; we need to compute logits.
             * Actually, the talker_forward function expects an input embedding.
             * After prefill, we need the next input embedding which is already prepared
             * as the prefill's output representation. Let's handle this:
             *
             * Actually, in the Python code, after prefill, the first generation step
             * uses the last hidden state to produce logits directly.
             * Then subsequent steps use the sampled token's embedding.
             *
             * For simplicity, let's compute logits from the last hidden state: */
            kernel_matvec_bf16(logits, ctx->talker.codec_head_bf16, ctx->tk_x,
                               cfg->talker_vocab_size, hidden);
        } else {
            /* Forward pass with the next embedding */
            qwen_tts_talker_forward(ctx, next_embed, logits);
        }

        /* Apply suppress tokens */
        for (int i = 0; i < n_suppress; i++) {
            logits[suppress_tokens[i]] = -1e9f;
        }

        /* Apply repetition penalty */
        kernel_apply_repetition_penalty(logits, generated_tokens, n_generated,
                                        cfg->talker_vocab_size, ctx->repetition_penalty);

        /* Sample */
        int token = kernel_sample_top_k(logits, cfg->talker_vocab_size, ctx->top_k,
                                         ctx->top_p, ctx->temperature, &rng_state);

        /* Check for EOS */
        if (token == cfg->codec_eos_id) {
            if (qwen_tts_verbose >= 1)
                fprintf(stderr, "EOS at step %d\n", step);
            break;
        }

        generated_tokens[n_generated] = token;

        /* Generate remaining code groups via sub-talker */
        int codes[QWEN_TTS_NUM_CODE_GROUPS];
        qwen_tts_subtalker_generate(ctx, ctx->tk_x, token, codes);

        /* Store all codes */
        memcpy(all_codes + n_generated * num_groups, codes, num_groups * sizeof(int));
        n_generated++;

        /* Build next input embedding:
         *   sum of all 32 group embeddings + trailing_text[step] or tts_pad_embed
         */
        memset(next_embed, 0, hidden * sizeof(float));

        /* Group 0: talker codec embedding */
        float *emb_tmp = (float *)malloc(hidden * sizeof(float));
        embed_codec_token(ctx, token, emb_tmp);
        kernel_add_inplace(next_embed, emb_tmp, hidden);

        /* Groups 1-31: sub-talker codec embeddings */
        for (int g = 1; g < num_groups; g++) {
            int emb_dim = hidden;  /* sub-talker embeddings have talker_hidden_size dim */
            kernel_bf16_to_f32(emb_tmp, ctx->subtalker.codec_embeddings_bf16[g - 1] +
                               (size_t)codes[g] * emb_dim, emb_dim);
            kernel_add_inplace(next_embed, emb_tmp, hidden);
        }
        free(emb_tmp);

        /* Add trailing text embedding */
        if (step < n_trailing) {
            kernel_add_inplace(next_embed, trailing_text + step * hidden, hidden);
        } else {
            kernel_add_inplace(next_embed, tts_pad_proj, hidden);
        }

        /* Progress callback */
        if (ctx->progress_cb) {
            ctx->progress_cb(step + 1, max_tokens, ctx->progress_cb_userdata);
        }
        if (qwen_tts_verbose >= 1 && (n_generated % 10 == 0)) {
            double elapsed = time_ms() - t_gen;
            fprintf(stderr, "\r  Token %d (%.1f ms/token)...", n_generated, elapsed / n_generated);
        }
    }

    double t_gen_done = time_ms();
    ctx->perf_talker_ms = t_gen_done - t_gen;
    ctx->perf_codec_tokens = n_generated;

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "\r                                        \r");  /* clear progress line */
        fprintf(stderr, "Generated %d codec tokens in %.1f ms (%.1f ms/token)\n",
                n_generated, ctx->perf_talker_ms,
                n_generated > 0 ? ctx->perf_talker_ms / n_generated : 0);
    }

    free(logits); free(generated_tokens); free(suppress_tokens);
    free(trailing_text); free(tts_pad_proj); free(tts_bos_proj); free(tts_eos_proj);
    free(text_tokens);

    if (n_generated == 0) {
        free(all_codes); free(next_embed);
        *out_samples = 0;
        return NULL;
    }

    /* ---- Codec Decode ---- */
    double t_codec = time_ms();

    float *audio = qwen_tts_codec_decode(ctx, all_codes, n_generated, out_samples);

    double t_codec_done = time_ms();
    ctx->perf_codec_ms = t_codec_done - t_codec;
    ctx->perf_total_ms = t_codec_done - t_start;

    if (qwen_tts_verbose >= 1) {
        fprintf(stderr, "Codec decode: %d samples in %.1f ms\n", *out_samples, ctx->perf_codec_ms);
        fprintf(stderr, "Total: %.1f ms (%.2f s audio, %.2fx realtime)\n",
                ctx->perf_total_ms,
                *out_samples > 0 ? (float)*out_samples / QWEN_TTS_SAMPLE_RATE : 0,
                *out_samples > 0 ? ((float)*out_samples / QWEN_TTS_SAMPLE_RATE) / (ctx->perf_total_ms / 1000.0) : 0);
    }

    free(all_codes); free(next_embed);

    return audio;
}
