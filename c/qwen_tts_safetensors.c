/*
 * qwen_tts_safetensors.c - SafeTensors mmap-based reader
 *
 * SafeTensors format:
 *   [8 bytes LE: header_size]
 *   [header_size bytes: JSON header]
 *   [remaining bytes: tensor data]
 *
 * The JSON header maps tensor names to their dtype, shape, and data_offsets.
 */

#include "qwen_tts_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

extern int qwen_tts_verbose;

/* ---- Minimal JSON parser for safetensors header ---- */

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

/* Parse a JSON string, returns malloc'd copy. Advances *p past closing quote. */
static char *parse_json_string(const char **p) {
    skip_ws(p);
    if (**p != '"') return NULL;
    (*p)++;
    const char *start = *p;
    while (**p && **p != '"') {
        if (**p == '\\') (*p)++;  /* skip escaped char */
        (*p)++;
    }
    int len = (int)(*p - start);
    char *s = (char *)malloc(len + 1);
    memcpy(s, start, len);
    s[len] = '\0';
    if (**p == '"') (*p)++;
    return s;
}

/* Parse a JSON integer */
static int64_t parse_json_int(const char **p) {
    skip_ws(p);
    int64_t val = 0;
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static void skip_json_value(const char **p) {
    skip_ws(p);
    if (**p == '"') {
        char *s = parse_json_string(p);
        free(s);
    } else if (**p == '{') {
        (*p)++;
        skip_ws(p);
        if (**p != '}') {
            while (1) {
                char *key = parse_json_string(p);
                free(key);
                skip_ws(p);
                if (**p == ':') (*p)++;
                skip_json_value(p);
                skip_ws(p);
                if (**p == ',') { (*p)++; continue; }
                break;
            }
        }
        if (**p == '}') (*p)++;
    } else if (**p == '[') {
        (*p)++;
        skip_ws(p);
        if (**p != ']') {
            while (1) {
                skip_json_value(p);
                skip_ws(p);
                if (**p == ',') { (*p)++; continue; }
                break;
            }
        }
        if (**p == ']') (*p)++;
    } else {
        /* number, bool, null */
        while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
    }
}

/* Parse a tensor entry from the JSON header */
static int parse_tensor_entry(const char **p, safetensor_t *t) {
    skip_ws(p);
    if (**p != '{') return -1;
    (*p)++;

    t->ndim = 0;
    t->data_offset = 0;
    t->data_size = 0;

    while (1) {
        skip_ws(p);
        if (**p == '}') { (*p)++; break; }
        char *key = parse_json_string(p);
        if (!key) return -1;
        skip_ws(p);
        if (**p == ':') (*p)++;
        skip_ws(p);

        if (strcmp(key, "dtype") == 0) {
            t->dtype = parse_json_string(p);
        } else if (strcmp(key, "shape") == 0) {
            /* Parse array of ints */
            if (**p == '[') {
                (*p)++;
                skip_ws(p);
                t->ndim = 0;
                while (**p != ']' && t->ndim < 8) {
                    t->shape[t->ndim++] = parse_json_int(p);
                    skip_ws(p);
                    if (**p == ',') (*p)++;
                    skip_ws(p);
                }
                if (**p == ']') (*p)++;
            }
        } else if (strcmp(key, "data_offsets") == 0) {
            /* Parse [start, end] */
            if (**p == '[') {
                (*p)++;
                size_t start = (size_t)parse_json_int(p);
                skip_ws(p);
                if (**p == ',') (*p)++;
                size_t end = (size_t)parse_json_int(p);
                skip_ws(p);
                if (**p == ']') (*p)++;
                t->data_offset = start;
                t->data_size = end - start;
            }
        } else {
            skip_json_value(p);
        }
        free(key);
        skip_ws(p);
        if (**p == ',') (*p)++;
    }
    return 0;
}

static safetensors_file_t *safetensors_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    size_t file_size = (size_t)st.st_size;

    void *addr = NULL;
    int is_heap_copy = 0;

    addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
#ifdef __EMSCRIPTEN__
        /* Some browser FS backends can fail mmap for large files. Fall back to heap copy. */
        addr = malloc(file_size);
        if (!addr) {
            close(fd);
            return NULL;
        }
        size_t off = 0;
        while (off < file_size) {
            ssize_t n = read(fd, (uint8_t *)addr + off, file_size - off);
            if (n <= 0) {
                free(addr);
                close(fd);
                return NULL;
            }
            off += (size_t)n;
        }
        close(fd);
        fd = -1;
        is_heap_copy = 1;
#else
        close(fd);
        return NULL;
#endif
    }

    /* Read header size (8 bytes LE) */
    uint64_t header_size;
    memcpy(&header_size, addr, 8);

    if (header_size + 8 > file_size) {
        if (is_heap_copy) free(addr);
        else munmap(addr, file_size);
        if (fd >= 0) close(fd);
        return NULL;
    }

    /* Parse JSON header */
    char *header_json = (char *)malloc(header_size + 1);
    memcpy(header_json, (uint8_t *)addr + 8, header_size);
    header_json[header_size] = '\0';

    /* Count tensors (top-level keys that aren't "__metadata__") */
    int n_tensors = 0;
    int capacity = 64;
    safetensor_t *tensors = (safetensor_t *)calloc(capacity, sizeof(safetensor_t));

    const char *p = header_json;
    skip_ws(&p);
    if (*p == '{') p++;

    while (1) {
        skip_ws(&p);
        if (*p == '}' || *p == '\0') break;

        char *name = parse_json_string(&p);
        skip_ws(&p);
        if (*p == ':') p++;
        skip_ws(&p);

        if (name && strcmp(name, "__metadata__") == 0) {
            skip_json_value(&p);
            free(name);
        } else {
            if (n_tensors >= capacity) {
                capacity *= 2;
                tensors = (safetensor_t *)realloc(tensors, capacity * sizeof(safetensor_t));
            }
            safetensor_t *t = &tensors[n_tensors];
            memset(t, 0, sizeof(*t));
            t->name = name;
            if (parse_tensor_entry(&p, t) == 0) {
                n_tensors++;
            } else {
                free(name);
            }
        }
        skip_ws(&p);
        if (*p == ',') p++;
    }

    free(header_json);

    safetensors_file_t *sf = (safetensors_file_t *)calloc(1, sizeof(safetensors_file_t));
    sf->path = strdup(path);
    sf->fd = fd;
    sf->mmap_addr = addr;
    sf->is_heap_copy = is_heap_copy;
    sf->mmap_size = file_size;
    sf->header_size = (size_t)header_size;
    sf->data_start = (uint8_t *)addr + 8 + header_size;
    sf->tensors = tensors;
    sf->n_tensors = n_tensors;

    return sf;
}

static void safetensors_close(safetensors_file_t *sf) {
    if (!sf) return;
    for (int i = 0; i < sf->n_tensors; i++) {
        free(sf->tensors[i].name);
        free(sf->tensors[i].dtype);
    }
    free(sf->tensors);
    if (sf->is_heap_copy) {
        free(sf->mmap_addr);
    } else if (sf->mmap_addr && sf->mmap_addr != MAP_FAILED) {
        munmap(sf->mmap_addr, sf->mmap_size);
    }
    if (sf->fd >= 0) close(sf->fd);
    free(sf->path);
    /* NOTE: do NOT free(sf) here -- the caller owns the storage.
       In multi_safetensors_open, the struct is shallow-copied into
       an array element, and the original is freed by the caller. */
}

/* Compare function for sorting file names */
static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

multi_safetensors_t *multi_safetensors_open(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "multi_safetensors_open: cannot open directory %s\n", dir);
        return NULL;
    }

    /* Collect .safetensors files */
    char **names = NULL;
    int n_names = 0;
    int names_cap = 0;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *dot = strrchr(ent->d_name, '.');
        if (dot && strcmp(dot, ".safetensors") == 0) {
            if (n_names >= names_cap) {
                names_cap = names_cap > 0 ? names_cap * 2 : 8;
                names = (char **)realloc(names, names_cap * sizeof(char *));
            }
            names[n_names++] = strdup(ent->d_name);
        }
    }
    closedir(d);

    if (n_names == 0) {
        fprintf(stderr, "multi_safetensors_open: no .safetensors files in %s\n", dir);
        free(names);
        return NULL;
    }

    /* Sort for deterministic ordering */
    qsort(names, n_names, sizeof(char *), cmp_str);

    multi_safetensors_t *ms = (multi_safetensors_t *)calloc(1, sizeof(multi_safetensors_t));
    snprintf(ms->base_dir, sizeof(ms->base_dir), "%s", dir);
    ms->files = (safetensors_file_t *)calloc(n_names, sizeof(safetensors_file_t));
    ms->n_files = 0;

    for (int i = 0; i < n_names; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        safetensors_file_t *sf = safetensors_open(path);
        if (sf) {
            ms->files[ms->n_files] = *sf;
            free(sf);  /* shallow copy was done */
            ms->n_files++;
            if (qwen_tts_verbose >= 2)
                fprintf(stderr, "  Opened: %s (%d tensors)\n", names[i], ms->files[ms->n_files - 1].n_tensors);
        } else {
            fprintf(stderr, "  Warning: failed to open %s\n", path);
        }
        free(names[i]);
    }
    free(names);

    if (qwen_tts_verbose >= 1) {
        int total = 0;
        for (int i = 0; i < ms->n_files; i++) total += ms->files[i].n_tensors;
        fprintf(stderr, "Loaded %d safetensors files (%d tensors total)\n", ms->n_files, total);
    }

    if (ms->n_files == 0) {
        fprintf(stderr, "multi_safetensors_open: failed to open any safetensors files in %s\n", dir);
        free(ms->files);
        free(ms);
        return NULL;
    }

    return ms;
}

void multi_safetensors_close(multi_safetensors_t *ms) {
    if (!ms) return;
    for (int i = 0; i < ms->n_files; i++) {
        safetensors_close(&ms->files[i]);
    }
    free(ms->files);
    free(ms);
}

const safetensor_t *multi_safetensors_find(const multi_safetensors_t *ms,
                                            const char *name,
                                            void **data) {
    for (int i = 0; i < ms->n_files; i++) {
        safetensors_file_t *sf = &ms->files[i];
        for (int j = 0; j < sf->n_tensors; j++) {
            if (strcmp(sf->tensors[j].name, name) == 0) {
                if (data) {
                    *data = sf->data_start + sf->tensors[j].data_offset;
                }
                return &sf->tensors[j];
            }
        }
    }
    return NULL;
}

const uint16_t *multi_safetensors_get_bf16(const multi_safetensors_t *ms,
                                            const char *name,
                                            int64_t *shape, int *ndim) {
    void *data = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &data);
    if (!t || !data) return NULL;

    if (shape && ndim) {
        *ndim = t->ndim;
        for (int i = 0; i < t->ndim; i++) shape[i] = t->shape[i];
    }
    return (const uint16_t *)data;
}

const float *multi_safetensors_get_f32(const multi_safetensors_t *ms,
                                        const char *name,
                                        int64_t *shape, int *ndim) {
    void *data = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &data);
    if (!t || !data) return NULL;

    if (shape && ndim) {
        *ndim = t->ndim;
        for (int i = 0; i < t->ndim; i++) shape[i] = t->shape[i];
    }
    return (const float *)data;
}

float *multi_safetensors_load_f32(const multi_safetensors_t *ms,
                                   const char *name,
                                   int64_t *shape, int *ndim) {
    void *data = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &data);
    if (!t || !data) return NULL;

    if (shape && ndim) {
        *ndim = t->ndim;
        for (int i = 0; i < t->ndim; i++) shape[i] = t->shape[i];
    }

    /* Calculate total elements */
    int64_t total = 1;
    for (int i = 0; i < t->ndim; i++) total *= t->shape[i];

    float *result = (float *)malloc((size_t)total * sizeof(float));
    if (!result) return NULL;

    if (strcmp(t->dtype, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)data;
        for (int64_t i = 0; i < total; i++) {
            uint32_t f32_bits = ((uint32_t)src[i]) << 16;
            memcpy(&result[i], &f32_bits, sizeof(float));
        }
    } else if (strcmp(t->dtype, "F32") == 0) {
        memcpy(result, data, (size_t)total * sizeof(float));
    } else if (strcmp(t->dtype, "F16") == 0) {
        /* F16 to F32 conversion (simplified) */
        const uint16_t *src = (const uint16_t *)data;
        for (int64_t i = 0; i < total; i++) {
            uint16_t h = src[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t f;
            if (exp == 0) {
                if (mant == 0) f = sign << 31;
                else {
                    exp = 1;
                    while (!(mant & 0x400)) { mant <<= 1; exp--; }
                    mant &= 0x3FF;
                    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                f = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            memcpy(&result[i], &f, sizeof(float));
        }
    } else {
        fprintf(stderr, "multi_safetensors_load_f32: unsupported dtype %s for %s\n",
                t->dtype, name);
        free(result);
        return NULL;
    }

    return result;
}
