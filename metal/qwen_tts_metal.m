/*
 * qwen_tts_metal.m - Metal GPU Backend Implementation
 *
 * Manages Metal device, command queues, buffer pool, and compute pipeline
 * states for all GPU kernels used by Qwen3-TTS inference.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "qwen_tts_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Internal State
 * ======================================================================== */

#define METAL_MAX_BUFFERS 4096

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLCommandBuffer> cmd_buf;
    id<MTLComputeCommandEncoder> encoder;  /* reused across dispatches */
    bool committed;
    int initialized;

    /* Buffer pool */
    id<MTLBuffer> buffers[METAL_MAX_BUFFERS];
    int buf_count;

    /* Pipeline states (one per kernel) */
    id<MTLComputePipelineState> ps_matvec_bf16;
    id<MTLComputePipelineState> ps_qkv_matvec_bf16;
    id<MTLComputePipelineState> ps_matvec_f32;
    id<MTLComputePipelineState> ps_matmul_f32;
    id<MTLComputePipelineState> ps_matmul_bf16;
    id<MTLComputePipelineState> ps_rms_norm;
    id<MTLComputePipelineState> ps_rms_norm_inplace;
    id<MTLComputePipelineState> ps_layer_norm;
    id<MTLComputePipelineState> ps_softmax;
    id<MTLComputePipelineState> ps_qk_norm_rope;
    id<MTLComputePipelineState> ps_kv_cache_store;
    id<MTLComputePipelineState> ps_attn_scores;
    id<MTLComputePipelineState> ps_attn_softmax_rows;
    id<MTLComputePipelineState> ps_attn_weighted_sum;
    id<MTLComputePipelineState> ps_attn_fused;
    id<MTLComputePipelineState> ps_silu_inplace;
    id<MTLComputePipelineState> ps_gelu_inplace;
    id<MTLComputePipelineState> ps_snake_beta;
    id<MTLComputePipelineState> ps_add_inplace;
    id<MTLComputePipelineState> ps_copy;
    id<MTLComputePipelineState> ps_mul_inplace;
    id<MTLComputePipelineState> ps_scale_inplace;
    id<MTLComputePipelineState> ps_clamp;
    id<MTLComputePipelineState> ps_rope_apply;
    id<MTLComputePipelineState> ps_causal_conv1d;
    id<MTLComputePipelineState> ps_transposed_conv1d;
    id<MTLComputePipelineState> ps_bf16_to_f32;
    id<MTLComputePipelineState> ps_argmax_i32;
    id<MTLComputePipelineState> ps_swiglu_matvec_bf16;
} metal_state_t;

static metal_state_t g_metal = {0};

/* ========================================================================
 * Pipeline Creation Helper
 * ======================================================================== */

static id<MTLComputePipelineState> create_pipeline(const char *name) {
    NSString *funcName = [NSString stringWithUTF8String:name];
    id<MTLFunction> func = [g_metal.library newFunctionWithName:funcName];
    if (!func) {
        fprintf(stderr, "Metal: kernel function '%s' not found in library\n", name);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> ps = [g_metal.device newComputePipelineStateWithFunction:func error:&error];
    if (!ps) {
        fprintf(stderr, "Metal: failed to create pipeline for '%s': %s\n",
                name, [[error localizedDescription] UTF8String]);
    }
    return ps;
}

/* ========================================================================
 * Initialization & Lifecycle
 * ======================================================================== */

int metal_init(void) {
    if (g_metal.initialized) return 0;

    @autoreleasepool {
        g_metal.device = MTLCreateSystemDefaultDevice();
        if (!g_metal.device) {
            fprintf(stderr, "Metal: no GPU device found\n");
            return -1;
        }

        g_metal.queue = [g_metal.device newCommandQueue];
        if (!g_metal.queue) {
            fprintf(stderr, "Metal: failed to create command queue\n");
            return -1;
        }

        /* Load Metal shader library.
         * Strategy:
         *   1. Try pre-compiled .metallib files in various locations
         *   2. Try compiling from .metal source at runtime
         *   3. Try default library (for Xcode builds)
         */
        NSError *error = nil;

        /* 1. Try pre-compiled metallib */
        NSArray *searchPaths = @[];
        NSString *execPath = [[NSBundle mainBundle] executablePath];
        if (execPath) {
            NSString *dir = [execPath stringByDeletingLastPathComponent];
            searchPaths = @[
                [dir stringByAppendingPathComponent:@"qwen_tts_metal_kernels.metallib"],
            ];
        }
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd))) {
            NSString *cwdStr = [NSString stringWithUTF8String:cwd];
            searchPaths = [searchPaths arrayByAddingObjectsFromArray:@[
                [cwdStr stringByAppendingPathComponent:@"qwen_tts_metal_kernels.metallib"],
                [cwdStr stringByAppendingPathComponent:@"metal/qwen_tts_metal_kernels.metallib"],
            ]];
        }

        for (NSString *path in searchPaths) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                NSURL *url = [NSURL fileURLWithPath:path];
                g_metal.library = [g_metal.device newLibraryWithURL:url error:&error];
                if (g_metal.library) {
                    fprintf(stderr, "Metal: loaded pre-compiled library from %s\n",
                            [path UTF8String]);
                    break;
                }
            }
        }

        /* 2. Try compiling from .metal source at runtime */
        if (!g_metal.library) {
            NSArray *sourcePaths = @[];
            if (getcwd(cwd, sizeof(cwd))) {
                NSString *cwdStr = [NSString stringWithUTF8String:cwd];
                sourcePaths = @[
                    [cwdStr stringByAppendingPathComponent:@"qwen_tts_metal_kernels.metal"],
                    [cwdStr stringByAppendingPathComponent:@"metal/qwen_tts_metal_kernels.metal"],
                ];
            }
            if (execPath) {
                NSString *dir = [execPath stringByDeletingLastPathComponent];
                sourcePaths = [sourcePaths arrayByAddingObject:
                    [dir stringByAppendingPathComponent:@"qwen_tts_metal_kernels.metal"]];
            }

            for (NSString *srcPath in sourcePaths) {
                if ([[NSFileManager defaultManager] fileExistsAtPath:srcPath]) {
                    NSString *source = [NSString stringWithContentsOfFile:srcPath
                                                                encoding:NSUTF8StringEncoding
                                                                   error:&error];
                    if (source) {
                        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
                        if (@available(macOS 15.0, iOS 18.0, *)) {
                            opts.mathMode = MTLMathModeFast;
                        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
                        }
                        g_metal.library = [g_metal.device newLibraryWithSource:source
                                                                      options:opts
                                                                        error:&error];
                        if (g_metal.library) {
                            fprintf(stderr, "Metal: compiled shaders from %s\n",
                                    [srcPath UTF8String]);
                            break;
                        } else {
                            fprintf(stderr, "Metal: shader compilation failed: %s\n",
                                    [[error localizedDescription] UTF8String]);
                        }
                    }
                }
            }
        }

        /* 3. Try default library (Xcode builds) */
        if (!g_metal.library) {
            g_metal.library = [g_metal.device newDefaultLibrary];
        }

        if (!g_metal.library) {
            fprintf(stderr, "Metal: failed to load shader library\n");
            return -1;
        }

        /* Create all pipeline states */
        g_metal.ps_matvec_bf16 = create_pipeline("kernel_matvec_bf16_metal");
        g_metal.ps_qkv_matvec_bf16 = create_pipeline("kernel_qkv_matvec_bf16_metal");
        g_metal.ps_matvec_f32 = create_pipeline("kernel_matvec_f32_metal");
        g_metal.ps_matmul_f32 = create_pipeline("kernel_matmul_f32_metal");
        g_metal.ps_matmul_bf16 = create_pipeline("kernel_matmul_bf16_metal");
        g_metal.ps_rms_norm = create_pipeline("kernel_rms_norm_metal");
        g_metal.ps_rms_norm_inplace = create_pipeline("kernel_rms_norm_inplace_metal");
        g_metal.ps_layer_norm = create_pipeline("kernel_layer_norm_metal");
        g_metal.ps_softmax = create_pipeline("kernel_softmax_metal");
        g_metal.ps_qk_norm_rope = create_pipeline("kernel_qk_norm_rope_metal");
        g_metal.ps_kv_cache_store = create_pipeline("kernel_kv_cache_store_metal");
        g_metal.ps_attn_scores = create_pipeline("kernel_attn_scores_metal");
        g_metal.ps_attn_softmax_rows = create_pipeline("kernel_attn_softmax_rows_metal");
        g_metal.ps_attn_weighted_sum = create_pipeline("kernel_attn_weighted_sum_metal");
        g_metal.ps_attn_fused = create_pipeline("kernel_attn_fused_metal");
        g_metal.ps_silu_inplace = create_pipeline("kernel_silu_inplace_metal");
        g_metal.ps_gelu_inplace = create_pipeline("kernel_gelu_inplace_metal");
        g_metal.ps_snake_beta = create_pipeline("kernel_snake_beta_metal");
        g_metal.ps_add_inplace = create_pipeline("kernel_add_inplace_metal");
        g_metal.ps_copy = create_pipeline("kernel_copy_metal");
        g_metal.ps_mul_inplace = create_pipeline("kernel_mul_inplace_metal");
        g_metal.ps_scale_inplace = create_pipeline("kernel_scale_inplace_metal");
        g_metal.ps_clamp = create_pipeline("kernel_clamp_metal");
        g_metal.ps_rope_apply = create_pipeline("kernel_rope_apply_metal");
        g_metal.ps_causal_conv1d = create_pipeline("kernel_causal_conv1d_metal");
        g_metal.ps_transposed_conv1d = create_pipeline("kernel_transposed_conv1d_metal");
        g_metal.ps_bf16_to_f32 = create_pipeline("kernel_bf16_to_f32_metal");
        g_metal.ps_argmax_i32 = create_pipeline("kernel_argmax_i32_metal");
        g_metal.ps_swiglu_matvec_bf16 = create_pipeline("kernel_swiglu_matvec_bf16_metal");

        g_metal.buf_count = 0;
        g_metal.cmd_buf = nil;
        g_metal.initialized = 1;
    }

    return 0;
}

int metal_is_available(void) {
    return g_metal.initialized;
}

void metal_shutdown(void) {
    if (!g_metal.initialized) return;
    @autoreleasepool {
        if (g_metal.encoder) {
            [g_metal.encoder endEncoding];
            g_metal.encoder = nil;
        }
        if (g_metal.cmd_buf) {
            [g_metal.cmd_buf waitUntilCompleted];
            g_metal.cmd_buf = nil;
        }
        for (int i = 0; i < g_metal.buf_count; i++) {
            g_metal.buffers[i] = nil;
        }
        g_metal.buf_count = 0;
        g_metal.queue = nil;
        g_metal.library = nil;
        g_metal.device = nil;
        g_metal.initialized = 0;
    }
}

void metal_print_info(void) {
    if (!g_metal.initialized) {
        fprintf(stderr, "Metal: not initialized\n");
        return;
    }
    fprintf(stderr, "Metal device: %s\n", [[g_metal.device name] UTF8String]);
    fprintf(stderr, "Metal max buffer length: %zu MB\n",
            (size_t)[g_metal.device maxBufferLength] / (1024 * 1024));
    fprintf(stderr, "Metal unified memory: %s\n",
            [g_metal.device hasUnifiedMemory] ? "yes" : "no");
}

/* ========================================================================
 * Buffer Management
 * ======================================================================== */

metal_buf_t metal_buf_create(const void *data, size_t size) {
    if (!g_metal.initialized || g_metal.buf_count >= METAL_MAX_BUFFERS) return METAL_BUF_INVALID;
    @autoreleasepool {
        id<MTLBuffer> buf = [g_metal.device newBufferWithBytes:data
                                                       length:size
                                                      options:MTLResourceStorageModeShared];
        if (!buf) return METAL_BUF_INVALID;
        int idx = g_metal.buf_count++;
        g_metal.buffers[idx] = buf;
        return idx;
    }
}

metal_buf_t metal_buf_from_ptr(void *ptr, size_t size) {
    if (!g_metal.initialized || g_metal.buf_count >= METAL_MAX_BUFFERS || !ptr || size == 0)
        return METAL_BUF_INVALID;
    @autoreleasepool {
        id<MTLBuffer> buf = nil;
        /* Check if pointer is page-aligned for zero-copy */
        uintptr_t addr = (uintptr_t)ptr;
        size_t page_size = (size_t)getpagesize();
        if ((addr % page_size) == 0) {
            /* Page-aligned: zero-copy wrap. Round size up to page boundary. */
            size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);
            buf = [g_metal.device newBufferWithBytesNoCopy:ptr
                                                   length:aligned_size
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        }
        if (!buf) {
            /* Not page-aligned or NoCopy failed: copy data */
            buf = [g_metal.device newBufferWithBytes:ptr
                                              length:size
                                             options:MTLResourceStorageModeShared];
        }
        if (!buf) return METAL_BUF_INVALID;
        int idx = g_metal.buf_count++;
        g_metal.buffers[idx] = buf;
        return idx;
    }
}

metal_buf_t metal_buf_create_empty(size_t size) {
    if (!g_metal.initialized || g_metal.buf_count >= METAL_MAX_BUFFERS) return METAL_BUF_INVALID;
    @autoreleasepool {
        id<MTLBuffer> buf = [g_metal.device newBufferWithLength:size
                                                       options:MTLResourceStorageModeShared];
        if (!buf) return METAL_BUF_INVALID;
        int idx = g_metal.buf_count++;
        g_metal.buffers[idx] = buf;
        return idx;
    }
}

metal_buf_t metal_buf_ensure(metal_buf_t buf, size_t size) {
    if (buf != METAL_BUF_INVALID && buf < g_metal.buf_count) {
        if ([g_metal.buffers[buf] length] >= size) return buf;
        /* Need to reallocate */
        g_metal.buffers[buf] = nil;
        @autoreleasepool {
            id<MTLBuffer> newbuf = [g_metal.device newBufferWithLength:size
                                                              options:MTLResourceStorageModeShared];
            if (!newbuf) return METAL_BUF_INVALID;
            g_metal.buffers[buf] = newbuf;
            return buf;
        }
    }
    return metal_buf_create_empty(size);
}

void metal_buf_release(metal_buf_t buf) {
    if (buf >= 0 && buf < g_metal.buf_count) {
        g_metal.buffers[buf] = nil;
    }
}

void *metal_buf_contents(metal_buf_t buf) {
    if (buf < 0 || buf >= g_metal.buf_count || !g_metal.buffers[buf]) return NULL;
    return [g_metal.buffers[buf] contents];
}

size_t metal_buf_size(metal_buf_t buf) {
    if (buf < 0 || buf >= g_metal.buf_count || !g_metal.buffers[buf]) return 0;
    return [g_metal.buffers[buf] length];
}

void metal_buf_write(metal_buf_t buf, const void *data, size_t size) {
    void *ptr = metal_buf_contents(buf);
    if (ptr) memcpy(ptr, data, size);
}

void metal_buf_read(metal_buf_t buf, void *dst, size_t size) {
    void *ptr = metal_buf_contents(buf);
    if (ptr) memcpy(dst, ptr, size);
}

/* ========================================================================
 * Command Buffer Management
 * ======================================================================== */

void metal_begin(void) {
    if (!g_metal.initialized) return;
    @autoreleasepool {
        if (g_metal.encoder) {
            [g_metal.encoder endEncoding];
            g_metal.encoder = nil;
        }
        if (g_metal.cmd_buf) {
            [g_metal.cmd_buf waitUntilCompleted];
        }
        g_metal.cmd_buf = [g_metal.queue commandBuffer];
        g_metal.committed = false;
    }
}

void metal_commit(void) {
    if (!g_metal.initialized || !g_metal.cmd_buf) return;
    if (g_metal.encoder) {
        [g_metal.encoder endEncoding];
        g_metal.encoder = nil;
    }
    [g_metal.cmd_buf commit];
    g_metal.committed = true;
}

void metal_sync(void) {
    if (!g_metal.initialized || !g_metal.cmd_buf) return;
    if (g_metal.encoder) {
        [g_metal.encoder endEncoding];
        g_metal.encoder = nil;
    }
    if (!g_metal.committed) {
        [g_metal.cmd_buf commit];
    }
    [g_metal.cmd_buf waitUntilCompleted];
    g_metal.cmd_buf = nil;
    g_metal.committed = false;
}

/* ========================================================================
 * Compute Dispatch Helpers
 * ======================================================================== */

/* Ensure we have an active command buffer */
static void ensure_cmd_buf(void) {
    if (!g_metal.cmd_buf) {
        g_metal.cmd_buf = [g_metal.queue commandBuffer];
    }
}

/* Get the shared compute encoder, set pipeline state.
 * Encoder is created once per command buffer and reused across all dispatches. */
static id<MTLComputeCommandEncoder> begin_compute(id<MTLComputePipelineState> ps) {
    ensure_cmd_buf();
    if (!g_metal.encoder) {
        g_metal.encoder = [g_metal.cmd_buf computeCommandEncoder];
    }
    [g_metal.encoder setComputePipelineState:ps];
    return g_metal.encoder;
}

static void dispatch_1d(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> ps, int count) {
    NSUInteger tw = [ps threadExecutionWidth];
    NSUInteger threads = (NSUInteger)count;
    MTLSize gridSize = MTLSizeMake(threads, 1, 1);
    MTLSize groupSize = MTLSizeMake(tw < threads ? tw : threads, 1, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
}

static void dispatch_2d(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> ps, int w, int h) {
    NSUInteger tw = [ps threadExecutionWidth];
    MTLSize gridSize = MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1);
    /* Use a square-ish thread group */
    NSUInteger gw = tw;
    NSUInteger gh = 1;
    if (w < (int)tw) { gw = (NSUInteger)w; }
    MTLSize groupSize = MTLSizeMake(gw, gh, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
}

/* ========================================================================
 * GPU Kernel Implementations
 * ======================================================================== */

/* Params struct matching Metal shader expectations */
typedef struct {
    int rows;
    int cols;
    int M;
    int N;
    int K;
    int dim;
    float eps;
    float scale;
    float min_val;
    float max_val;
    int num_heads;
    int head_dim;
    int in_channels;
    int out_channels;
    int kernel_size;
    int length;
    int dilation;
    int groups;
    int stride;
    int intermediate;
    int hidden;
    int channels;
    int n;
    int kv_heads;
    int groups_per_head;
    int kv_dim;
    int kv_max;
    int seq_len;
    int layer_idx;
    int pos;
    int row_idx;
    int row_size;
} metal_params_t;

void metal_matvec_bf16(metal_buf_t out, metal_buf_t A_bf16, metal_buf_t x,
                       int rows, int cols) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_matvec_bf16);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[A_bf16] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:2];
        metal_params_t p = {.rows = rows, .cols = cols};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        /* 4 SIMD groups (128 threads) per row for maximum memory bandwidth */
        {
            NSUInteger total = (NSUInteger)rows * 128;
            MTLSize grid = MTLSizeMake(total, 1, 1);
            MTLSize group = MTLSizeMake(128, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
        }
    }
}

void metal_qkv_matvec_bf16(metal_buf_t q, metal_buf_t k, metal_buf_t v,
                           metal_buf_t wq_bf16, metal_buf_t wk_bf16, metal_buf_t wv_bf16,
                           metal_buf_t x, int num_heads, int kv_heads, int head_dim, int hidden) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_qkv_matvec_bf16);
        [enc setBuffer:g_metal.buffers[q] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[k] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[v] offset:0 atIndex:2];
        [enc setBuffer:g_metal.buffers[wq_bf16] offset:0 atIndex:3];
        [enc setBuffer:g_metal.buffers[wk_bf16] offset:0 atIndex:4];
        [enc setBuffer:g_metal.buffers[wv_bf16] offset:0 atIndex:5];
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:6];
        metal_params_t p = {.num_heads = num_heads, .kv_heads = kv_heads, .head_dim = head_dim, .hidden = hidden};
        [enc setBytes:&p length:sizeof(p) atIndex:7];
        int q_rows = num_heads * head_dim;
        int kv_rows = kv_heads * head_dim;
        int total_rows = q_rows + kv_rows + kv_rows;
        /* 4 SIMD groups (128 threads) per row for maximum memory bandwidth */
        {
            NSUInteger total = (NSUInteger)total_rows * 128;
            MTLSize grid = MTLSizeMake(total, 1, 1);
            MTLSize group = MTLSizeMake(128, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
        }
    }
}

void metal_matvec_f32(metal_buf_t out, metal_buf_t A, metal_buf_t x,
                      int rows, int cols) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_matvec_f32);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[A] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:2];
        metal_params_t p = {.rows = rows, .cols = cols};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(enc, g_metal.ps_matvec_f32, rows);
    }
}

void metal_matmul_f32(metal_buf_t C, metal_buf_t A, metal_buf_t B,
                      int M, int N, int K) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_matmul_f32);
        [enc setBuffer:g_metal.buffers[C] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[A] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[B] offset:0 atIndex:2];
        metal_params_t p = {.M = M, .N = N, .K = K};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_2d(enc, g_metal.ps_matmul_f32, N, M);
    }
}

void metal_matmul_bf16(metal_buf_t C, metal_buf_t A, metal_buf_t B_bf16,
                       int M, int N, int K) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_matmul_bf16);
        [enc setBuffer:g_metal.buffers[C] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[A] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[B_bf16] offset:0 atIndex:2];
        metal_params_t p = {.M = M, .N = N, .K = K};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_2d(enc, g_metal.ps_matmul_bf16, N, M);
    }
}

void metal_rms_norm(metal_buf_t out, metal_buf_t x, metal_buf_t weight,
                    int dim, float eps) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_rms_norm);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[weight] offset:0 atIndex:2];
        metal_params_t p = {.dim = dim, .eps = eps};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        /* Single threadgroup for reduction */
        NSUInteger tw = [g_metal.ps_rms_norm threadExecutionWidth];
        NSUInteger tg = tw;
        while (tg < (NSUInteger)dim && tg < 1024) tg *= 2;
        if (tg > 1024) tg = 1024;
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void metal_rms_norm_inplace(metal_buf_t x, metal_buf_t weight,
                            int dim, float eps) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_rms_norm_inplace);
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[weight] offset:0 atIndex:1];
        metal_params_t p = {.dim = dim, .eps = eps};
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        NSUInteger tw = [g_metal.ps_rms_norm_inplace threadExecutionWidth];
        NSUInteger tg = tw;
        while (tg < (NSUInteger)dim && tg < 1024) tg *= 2;
        if (tg > 1024) tg = 1024;
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void metal_layer_norm(metal_buf_t out, metal_buf_t x, metal_buf_t weight,
                      metal_buf_t bias, int dim, float eps) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_layer_norm);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:1];
        [enc setBuffer:(weight >= 0 ? g_metal.buffers[weight] : nil) offset:0 atIndex:2];
        [enc setBuffer:(bias >= 0 ? g_metal.buffers[bias] : nil) offset:0 atIndex:3];
        metal_params_t p = {.dim = dim, .eps = eps};
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        NSUInteger tg = 256;
        if ((NSUInteger)dim < tg) tg = (NSUInteger)dim;
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void metal_softmax(metal_buf_t x, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_softmax);
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:0];
        metal_params_t p = {.n = n};
        [enc setBytes:&p length:sizeof(p) atIndex:1];
        NSUInteger tg = 256;
        if ((NSUInteger)n < tg) tg = (NSUInteger)n;
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void metal_qk_norm_rope(metal_buf_t q, metal_buf_t k,
                        metal_buf_t q_norm_weight, metal_buf_t k_norm_weight,
                        metal_buf_t cos_buf, metal_buf_t sin_buf,
                        int num_heads, int kv_heads, int head_dim, float eps) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_qk_norm_rope);
        [enc setBuffer:g_metal.buffers[q] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[k] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[q_norm_weight] offset:0 atIndex:2];
        [enc setBuffer:g_metal.buffers[k_norm_weight] offset:0 atIndex:3];
        [enc setBuffer:g_metal.buffers[cos_buf] offset:0 atIndex:4];
        [enc setBuffer:g_metal.buffers[sin_buf] offset:0 atIndex:5];
        metal_params_t p = {.num_heads = num_heads, .kv_heads = kv_heads, .head_dim = head_dim, .eps = eps};
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        NSUInteger tg = 128;
        if ((NSUInteger)head_dim < tg) tg = (NSUInteger)head_dim;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(num_heads + kv_heads), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void metal_kv_cache_store(metal_buf_t kv_k, metal_buf_t kv_v,
                          metal_buf_t k, metal_buf_t v,
                          int layer_idx, int pos, int kv_dim, int kv_max) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_kv_cache_store);
        [enc setBuffer:g_metal.buffers[kv_k] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[kv_v] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[k] offset:0 atIndex:2];
        [enc setBuffer:g_metal.buffers[v] offset:0 atIndex:3];
        metal_params_t p = {.layer_idx = layer_idx, .pos = pos, .kv_dim = kv_dim, .kv_max = kv_max};
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_1d(enc, g_metal.ps_kv_cache_store, kv_dim);
    }
}

void metal_attn_scores(metal_buf_t scores, metal_buf_t q, metal_buf_t kv_k,
                       int layer_idx, int seq_len, int kv_max,
                       int num_heads, int kv_heads, int head_dim, float scale) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_attn_scores);
        [enc setBuffer:g_metal.buffers[scores] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[q] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[kv_k] offset:0 atIndex:2];
        int groups_per_head = num_heads / kv_heads;
        metal_params_t p = {.layer_idx = layer_idx, .seq_len = seq_len, .kv_max = kv_max,
                            .num_heads = num_heads, .kv_heads = kv_heads, .head_dim = head_dim,
                            .groups_per_head = groups_per_head, .kv_dim = kv_heads * head_dim, .scale = scale};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_2d(enc, g_metal.ps_attn_scores, seq_len, num_heads);
    }
}

void metal_attn_softmax_rows(metal_buf_t scores, int num_heads, int seq_len, int kv_max) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_attn_softmax_rows);
        [enc setBuffer:g_metal.buffers[scores] offset:0 atIndex:0];
        metal_params_t p = {.num_heads = num_heads, .seq_len = seq_len, .kv_max = kv_max};
        [enc setBytes:&p length:sizeof(p) atIndex:1];
        NSUInteger tg = 256;
        if ((NSUInteger)seq_len < tg) {
            tg = 1;
            while (tg < (NSUInteger)seq_len) tg <<= 1;
            if (tg > 256) tg = 256;
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void metal_attn_weighted_sum(metal_buf_t out, metal_buf_t scores, metal_buf_t kv_v,
                             int layer_idx, int seq_len, int kv_max,
                             int num_heads, int kv_heads, int head_dim) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_attn_weighted_sum);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[scores] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[kv_v] offset:0 atIndex:2];
        int groups_per_head = num_heads / kv_heads;
        metal_params_t p = {.layer_idx = layer_idx, .seq_len = seq_len, .kv_max = kv_max,
                            .num_heads = num_heads, .kv_heads = kv_heads, .head_dim = head_dim,
                            .groups_per_head = groups_per_head, .kv_dim = kv_heads * head_dim};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_2d(enc, g_metal.ps_attn_weighted_sum, head_dim, num_heads);
    }
}

void metal_attn_fused(metal_buf_t out, metal_buf_t scores, metal_buf_t q,
                      metal_buf_t kv_k, metal_buf_t kv_v,
                      int layer_idx, int seq_len, int kv_max,
                      int num_heads, int kv_heads, int head_dim, float scale) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_attn_fused);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[scores] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[q] offset:0 atIndex:2];
        [enc setBuffer:g_metal.buffers[kv_k] offset:0 atIndex:3];
        [enc setBuffer:g_metal.buffers[kv_v] offset:0 atIndex:4];
        int groups_per_head = num_heads / kv_heads;
        metal_params_t p = {.layer_idx = layer_idx, .seq_len = seq_len, .kv_max = kv_max,
                            .num_heads = num_heads, .kv_heads = kv_heads, .head_dim = head_dim,
                            .groups_per_head = groups_per_head, .kv_dim = kv_heads * head_dim,
                            .scale = scale};
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        NSUInteger tg = 128;
        int max_work = seq_len > head_dim ? seq_len : head_dim;
        if (max_work < (int)tg) {
            tg = 1;
            while ((int)tg < max_work) tg <<= 1;
            if (tg > 128) tg = 128;
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void metal_silu_inplace(metal_buf_t x, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_silu_inplace);
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:0];
        metal_params_t p = {.n = n};
        [enc setBytes:&p length:sizeof(p) atIndex:1];
        dispatch_1d(enc, g_metal.ps_silu_inplace, n);
    }
}

void metal_gelu_inplace(metal_buf_t x, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_gelu_inplace);
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:0];
        metal_params_t p = {.n = n};
        [enc setBytes:&p length:sizeof(p) atIndex:1];
        dispatch_1d(enc, g_metal.ps_gelu_inplace, n);
    }
}

void metal_snake_beta(metal_buf_t out, metal_buf_t x, metal_buf_t alpha,
                      metal_buf_t beta, int channels, int length) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_snake_beta);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[alpha] offset:0 atIndex:2];
        [enc setBuffer:g_metal.buffers[beta] offset:0 atIndex:3];
        metal_params_t p = {.channels = channels, .length = length};
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_1d(enc, g_metal.ps_snake_beta, channels * length);
    }
}

void metal_add_inplace(metal_buf_t a, metal_buf_t b, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_add_inplace);
        [enc setBuffer:g_metal.buffers[a] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[b] offset:0 atIndex:1];
        metal_params_t p = {.n = n};
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(enc, g_metal.ps_add_inplace, n);
    }
}

void metal_mul_inplace(metal_buf_t a, metal_buf_t b, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_mul_inplace);
        [enc setBuffer:g_metal.buffers[a] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[b] offset:0 atIndex:1];
        metal_params_t p = {.n = n};
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(enc, g_metal.ps_mul_inplace, n);
    }
}

void metal_copy(metal_buf_t dst, metal_buf_t src, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_copy);
        [enc setBuffer:g_metal.buffers[dst] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[src] offset:0 atIndex:1];
        metal_params_t p = {.n = n};
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(enc, g_metal.ps_copy, n);
    }
}

void metal_scale_inplace(metal_buf_t x, float scale, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_scale_inplace);
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:0];
        metal_params_t p = {.n = n, .scale = scale};
        [enc setBytes:&p length:sizeof(p) atIndex:1];
        dispatch_1d(enc, g_metal.ps_scale_inplace, n);
    }
}

void metal_clamp(metal_buf_t x, int n, float min_val, float max_val) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_clamp);
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:0];
        metal_params_t p = {.n = n, .min_val = min_val, .max_val = max_val};
        [enc setBytes:&p length:sizeof(p) atIndex:1];
        dispatch_1d(enc, g_metal.ps_clamp, n);
    }
}

void metal_rope_apply(metal_buf_t q, metal_buf_t k, metal_buf_t cos_buf,
                      metal_buf_t sin_buf, int num_heads, int head_dim) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_rope_apply);
        [enc setBuffer:g_metal.buffers[q] offset:0 atIndex:0];
        [enc setBuffer:(k >= 0 ? g_metal.buffers[k] : nil) offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[cos_buf] offset:0 atIndex:2];
        [enc setBuffer:g_metal.buffers[sin_buf] offset:0 atIndex:3];
        metal_params_t p = {.num_heads = num_heads, .head_dim = head_dim};
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        int half = head_dim / 2;
        dispatch_1d(enc, g_metal.ps_rope_apply, num_heads * half);
    }
}

void metal_causal_conv1d(metal_buf_t out, metal_buf_t input, metal_buf_t weight,
                         metal_buf_t bias, int in_channels, int out_channels,
                         int kernel_size, int length, int dilation, int groups) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_causal_conv1d);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[input] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[weight] offset:0 atIndex:2];
        [enc setBuffer:(bias >= 0 ? g_metal.buffers[bias] : nil) offset:0 atIndex:3];
        metal_params_t p = {.in_channels = in_channels, .out_channels = out_channels,
                            .kernel_size = kernel_size, .length = length,
                            .dilation = dilation, .groups = groups};
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_2d(enc, g_metal.ps_causal_conv1d, length, out_channels);
    }
}

void metal_transposed_conv1d(metal_buf_t out, metal_buf_t input, metal_buf_t weight,
                             metal_buf_t bias, int in_channels, int out_channels,
                             int kernel_size, int stride, int length, int *out_length) {
    int raw_out_len = (length - 1) * stride + kernel_size;
    int right_pad = kernel_size - stride;
    int final_len = raw_out_len - right_pad;
    if (final_len < 0) final_len = 0;
    if (out_length) *out_length = final_len;

    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_transposed_conv1d);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[input] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[weight] offset:0 atIndex:2];
        [enc setBuffer:(bias >= 0 ? g_metal.buffers[bias] : nil) offset:0 atIndex:3];
        metal_params_t p = {.in_channels = in_channels, .out_channels = out_channels,
                            .kernel_size = kernel_size, .stride = stride,
                            .length = length, .n = final_len};
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        dispatch_2d(enc, g_metal.ps_transposed_conv1d, final_len, out_channels);
    }
}

void metal_bf16_to_f32(metal_buf_t out, metal_buf_t in_buf, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_bf16_to_f32);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[in_buf] offset:0 atIndex:1];
        metal_params_t p = {.n = n};
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(enc, g_metal.ps_bf16_to_f32, n);
    }
}

void metal_bf16_row_to_f32(metal_buf_t out, metal_buf_t in_buf, int row_idx, int row_size) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_bf16_to_f32);
        NSUInteger row_off = (NSUInteger)row_idx * (NSUInteger)row_size * sizeof(uint16_t);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[in_buf] offset:row_off atIndex:1];
        metal_params_t p = {.n = row_size};
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        dispatch_1d(enc, g_metal.ps_bf16_to_f32, row_size);
    }
}

void metal_argmax_i32(metal_buf_t out_idx, metal_buf_t x, int n) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_argmax_i32);
        [enc setBuffer:g_metal.buffers[out_idx] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:1];
        metal_params_t p = {.n = n};
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        NSUInteger tg = 256;
        if ((NSUInteger)n < tg) {
            tg = 1;
            while ((int)tg < n) tg <<= 1;
            if (tg > 256) tg = 256;
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
}

void metal_swiglu_matvec_bf16(metal_buf_t out, metal_buf_t gate_up_bf16,
                              metal_buf_t x, int intermediate, int hidden) {
    @autoreleasepool {
        id<MTLComputeCommandEncoder> enc = begin_compute(g_metal.ps_swiglu_matvec_bf16);
        [enc setBuffer:g_metal.buffers[out] offset:0 atIndex:0];
        [enc setBuffer:g_metal.buffers[gate_up_bf16] offset:0 atIndex:1];
        [enc setBuffer:g_metal.buffers[x] offset:0 atIndex:2];
        metal_params_t p = {.intermediate = intermediate, .hidden = hidden};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        /* 32 threads per row (SIMD group), hardware simd_sum reduction */
        {
            NSUInteger total = (NSUInteger)intermediate * 32;
            MTLSize grid = MTLSizeMake(total, 1, 1);
            MTLSize group = MTLSizeMake(32, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
        }
    }
}
