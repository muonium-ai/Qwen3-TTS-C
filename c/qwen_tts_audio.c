/*
 * qwen_tts_audio.c - WAV file writer for Qwen3-TTS
 *
 * Writes 16-bit PCM WAV files from float32 samples.
 */

#include "qwen_tts.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

int qwen_tts_write_wav(const char *path, const float *samples, int n_samples, int sample_rate) {
    char tmp_path[4096];
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (n < 0 || n >= (int)sizeof(tmp_path)) {
        fprintf(stderr, "Error: output path too long: %s\n", path);
        return -1;
    }

    FILE *f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s for writing\n", tmp_path);
        return -1;
    }

    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int data_size = n_samples * block_align;

    /* RIFF header */
    uint8_t header[44];
    memcpy(header + 0, "RIFF", 4);
    uint32_t chunk_size = 36 + data_size;
    memcpy(header + 4, &chunk_size, 4);
    memcpy(header + 8, "WAVE", 4);

    /* fmt subchunk */
    memcpy(header + 12, "fmt ", 4);
    uint32_t subchunk1_size = 16;
    memcpy(header + 16, &subchunk1_size, 4);
    uint16_t audio_format = 1;  /* PCM */
    memcpy(header + 20, &audio_format, 2);
    uint16_t nc = (uint16_t)num_channels;
    memcpy(header + 22, &nc, 2);
    uint32_t sr = (uint32_t)sample_rate;
    memcpy(header + 24, &sr, 4);
    uint32_t br = (uint32_t)byte_rate;
    memcpy(header + 28, &br, 4);
    uint16_t ba = (uint16_t)block_align;
    memcpy(header + 32, &ba, 2);
    uint16_t bps = (uint16_t)bits_per_sample;
    memcpy(header + 34, &bps, 2);

    /* data subchunk */
    memcpy(header + 36, "data", 4);
    uint32_t ds = (uint32_t)data_size;
    memcpy(header + 40, &ds, 4);

    if (fwrite(header, 1, 44, f) != 44) {
        fprintf(stderr, "Error: failed to write WAV header to %s\n", tmp_path);
        fclose(f);
        remove(tmp_path);
        return -1;
    }

    /* Convert float32 [-1, 1] to int16 */
    for (int i = 0; i < n_samples; i++) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t sample = (int16_t)(s * 32767.0f);
        if (fwrite(&sample, 2, 1, f) != 1) {
            fprintf(stderr, "Error: failed to write WAV data to %s\n", tmp_path);
            fclose(f);
            remove(tmp_path);
            return -1;
        }
    }

    if (fclose(f) != 0) {
        fprintf(stderr, "Error: failed to close %s\n", tmp_path);
        remove(tmp_path);
        return -1;
    }

    if (rename(tmp_path, path) != 0) {
        fprintf(stderr, "Error: failed to rename %s -> %s: %s\n", tmp_path, path, strerror(errno));
        remove(tmp_path);
        return -1;
    }

    return 0;
}
