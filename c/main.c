/*
 * main.c - CLI entry point for Qwen3-TTS C inference
 *
 * Usage:
 *   ./qwen_tts -d <model_dir> -t "token_ids" [-s speaker] [-l language] [-o output.wav]
 *
 * Accepts pre-tokenized text as comma-separated BPE token IDs in
 * the Qwen2 chat template format:
 *   <|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n
 */

#include "qwen_tts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Qwen3-TTS - Pure C text-to-speech inference engine\n"
        "\n"
        "Usage: %s [options]\n"
        "\n"
        "Required:\n"
        "  -d <path>       Model directory (containing config.json + safetensors)\n"
        "  -t <ids>        Comma-separated BPE token IDs in chat template format\n"
        "  -f <file>       Read token IDs from file (one ID per line, or comma-separated)\n"
        "\n"
        "Optional:\n"
        "  -s <speaker>    Speaker name (from config.json spk_id map)\n"
        "  -l <language>   Language: auto, chinese, english, etc.\n"
        "  -o <path>       Output WAV file (default: output.wav)\n"
        "  -v              Verbose (repeat for more: -v -v)\n"
        "\n"
        "Generation parameters:\n"
        "  --temperature <f>            Talker temperature (default: 0.9)\n"
        "  --top-k <n>                  Talker top-K (default: 50)\n"
        "  --top-p <f>                  Talker top-P (default: 1.0)\n"
        "  --repetition-penalty <f>     Repetition penalty (default: 1.05)\n"
        "  --max-tokens <n>             Max codec tokens to generate (default: 4096)\n"
        "  --subtalker-temperature <f>  Sub-talker temperature (default: 0.9)\n"
        "  --subtalker-top-k <n>        Sub-talker top-K (default: 50)\n"
        "\n"
        "Token format:\n"
        "  Token IDs should follow the Qwen2 chat template:\n"
        "    im_start(151644), assistant_id, newline, TEXT_IDS...,\n"
        "    im_end(151645), newline, im_start(151644), assistant_id, newline\n"
        "\n"
        "  Use the Python tokenizer to generate IDs:\n"
        "    from transformers import AutoTokenizer\n"
        "    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-TTS')\n"
        "    ids = tok('<|im_start|>assistant\\nHello world.<|im_end|>\\n<|im_start|>assistant\\n')\n"
        "    print(','.join(str(i) for i in ids.input_ids))\n"
        "\n",
        prog
    );
}

/* Read token IDs from a file: comma-separated or one per line */
static char *read_tokens_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open token file: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);

    /* Replace newlines with commas for uniform parsing */
    for (long i = 0; i < size; i++) {
        if (buf[i] == '\n' || buf[i] == '\r') buf[i] = ',';
    }
    return buf;
}

static void progress_cb(int step, int total, void *userdata) {
    (void)total; (void)userdata;
    if (step % 50 == 0 || step < 5) {
        fprintf(stderr, "\rGenerating... step %d", step);
        fflush(stderr);
    }
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *token_ids = NULL;
    const char *token_file = NULL;
    const char *speaker = NULL;
    const char *language = NULL;
    const char *output_path = "output.wav";
    int verbose = 0;

    /* Generation params (use 0 = not set, will use ctx defaults) */
    float temperature = -1;
    float subtalker_temperature = -1;
    int top_k = -1;
    int subtalker_top_k = -1;
    float top_p = -1;
    float repetition_penalty = -1;
    int max_tokens = -1;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            token_ids = argv[++i];
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            token_file = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            speaker = argv[++i];
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            language = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose++;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--repetition-penalty") == 0 && i + 1 < argc) {
            repetition_penalty = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--subtalker-temperature") == 0 && i + 1 < argc) {
            subtalker_temperature = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--subtalker-top-k") == 0 && i + 1 < argc) {
            subtalker_top_k = (int)strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "Error: model directory required (-d)\n\n");
        usage(argv[0]);
        return 1;
    }

    if (!token_ids && !token_file) {
        fprintf(stderr, "Error: token IDs required (-t or -f)\n\n");
        usage(argv[0]);
        return 1;
    }

    /* Read tokens from file if -f was used */
    char *file_tokens = NULL;
    if (token_file) {
        file_tokens = read_tokens_file(token_file);
        if (!file_tokens) return 1;
        token_ids = file_tokens;
    }

    /* Set global verbose level */
    qwen_tts_verbose = verbose;

    /* ---- Load model ---- */
    if (verbose >= 1) fprintf(stderr, "Loading model from %s...\n", model_dir);

    qwen_tts_ctx_t *ctx = qwen_tts_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Error: failed to load model\n");
        free(file_tokens);
        return 1;
    }

    /* Apply generation parameters */
    if (temperature >= 0)          ctx->temperature = temperature;
    if (subtalker_temperature >= 0) ctx->subtalker_temperature = subtalker_temperature;
    if (top_k >= 0)                ctx->top_k = top_k;
    if (subtalker_top_k >= 0)      ctx->subtalker_top_k = subtalker_top_k;
    if (top_p >= 0)                ctx->top_p = top_p;
    if (repetition_penalty >= 0)   ctx->repetition_penalty = repetition_penalty;
    if (max_tokens >= 0)           ctx->max_new_tokens = max_tokens;

    /* Set progress callback for non-verbose mode */
    if (verbose == 0) {
        qwen_tts_set_progress_callback(ctx, progress_cb, NULL);
    }

    /* Print run parameters */
    if (verbose >= 1) {
        fprintf(stderr, "Generation params: temp=%.2f top_k=%d top_p=%.2f rep_penalty=%.2f max_tokens=%d\n",
                ctx->temperature, ctx->top_k, ctx->top_p, ctx->repetition_penalty, ctx->max_new_tokens);
        if (speaker) fprintf(stderr, "Speaker: %s\n", speaker);
        if (language) fprintf(stderr, "Language: %s\n", language);
    }

    /* ---- Generate ---- */
    int n_samples = 0;
    float *audio = qwen_tts_generate(ctx, token_ids, speaker, language, &n_samples);

    if (verbose == 0) fprintf(stderr, "\n");  /* newline after progress */

    if (!audio || n_samples == 0) {
        fprintf(stderr, "Error: generation produced no audio\n");
        qwen_tts_free(ctx);
        free(file_tokens);
        return 1;
    }

    /* ---- Write WAV ---- */
    if (qwen_tts_write_wav(output_path, audio, n_samples, QWEN_TTS_SAMPLE_RATE) != 0) {
        fprintf(stderr, "Error: failed to write %s\n", output_path);
        free(audio);
        qwen_tts_free(ctx);
        free(file_tokens);
        return 1;
    }

    float duration = (float)n_samples / QWEN_TTS_SAMPLE_RATE;
    fprintf(stderr, "Wrote %s: %.2f seconds (%d samples at %d Hz)\n",
            output_path, duration, n_samples, QWEN_TTS_SAMPLE_RATE);

    /* Print performance stats */
    if (verbose >= 1) {
        fprintf(stderr, "\nPerformance:\n");
        fprintf(stderr, "  Talker:  %.1f ms (%d tokens, %.1f ms/token)\n",
                ctx->perf_talker_ms, ctx->perf_codec_tokens,
                ctx->perf_codec_tokens > 0 ? ctx->perf_talker_ms / ctx->perf_codec_tokens : 0);
        fprintf(stderr, "  Codec:   %.1f ms\n", ctx->perf_codec_ms);
        fprintf(stderr, "  Total:   %.1f ms\n", ctx->perf_total_ms);
        fprintf(stderr, "  RTF:     %.2fx realtime\n",
                duration > 0 ? duration / (ctx->perf_total_ms / 1000.0) : 0);
    }

    free(audio);
    qwen_tts_free(ctx);
    free(file_tokens);

    return 0;
}
