/*
 * qwen_tts_safetensors.h - SafeTensors file format reader (mmap based)
 */

#ifndef QWEN_TTS_SAFETENSORS_H
#define QWEN_TTS_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

/* A single tensor within a safetensors file */
typedef struct {
    char *name;
    char *dtype;            /* "BF16", "F32", "F16", "I64", etc. */
    int ndim;
    int64_t shape[8];       /* up to 8 dimensions */
    size_t data_offset;     /* byte offset from start of data section */
    size_t data_size;       /* byte size of this tensor's data */
} safetensor_t;

/* A single safetensors file (mmap'd) */
typedef struct {
    char *path;
    int fd;
    void *mmap_addr;
    int is_heap_copy;       /* 1 when file data is loaded into malloc buffer */
    size_t mmap_size;
    size_t header_size;     /* JSON header byte count */
    uint8_t *data_start;    /* pointer to start of tensor data */

    safetensor_t *tensors;
    int n_tensors;
} safetensors_file_t;

/* Multiple safetensors files (sharding) */
typedef struct {
    safetensors_file_t *files;
    int n_files;
    char base_dir[512];
} multi_safetensors_t;

/* Open all safetensors files in a directory */
multi_safetensors_t *multi_safetensors_open(const char *dir);

/* Close and free all resources */
void multi_safetensors_close(multi_safetensors_t *ms);

/* Find a tensor by name. Returns pointer to tensor metadata, sets *data to the
 * raw data pointer (in mmap'd memory). Returns NULL if not found. */
const safetensor_t *multi_safetensors_find(const multi_safetensors_t *ms,
                                            const char *name,
                                            void **data);

/* Get tensor data as BF16 pointer (for BF16 tensors) */
const uint16_t *multi_safetensors_get_bf16(const multi_safetensors_t *ms,
                                            const char *name,
                                            int64_t *shape, int *ndim);

/* Get tensor data as F32 pointer (for F32 tensors) */
const float *multi_safetensors_get_f32(const multi_safetensors_t *ms,
                                        const char *name,
                                        int64_t *shape, int *ndim);

/* Allocate and copy a BF16 tensor to F32 */
float *multi_safetensors_load_f32(const multi_safetensors_t *ms,
                                   const char *name,
                                   int64_t *shape, int *ndim);

#endif /* QWEN_TTS_SAFETENSORS_H */
