/*
 * qwen_tts_metal_kernels.metal - Metal GPU Compute Shaders for Qwen3-TTS
 *
 * All kernels operate on Metal buffers using Apple Silicon GPU.
 * BF16 is handled via ushort (uint16_t) with manual conversion.
 */

#include <metal_stdlib>
using namespace metal;

/* ========================================================================
 * Common Types & Helpers
 * ======================================================================== */

struct Params {
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
};

/* BF16 to F32 conversion */
inline float bf16_to_f32(ushort bf16) {
    uint bits = uint(bf16) << 16;
    return as_type<float>(bits);
}

/* ========================================================================
 * Matrix-Vector Multiply (BF16 weights) — Multi-SIMD-group optimized
 * out[row] = sum_c A_bf16[row, c] * x[c]
 * 4 SIMD groups (128 threads) per row for maximum memory bandwidth.
 * Each SIMD group reduces via simd_sum(), then threadgroup reduction.
 * Dispatch: rows * 128 total threads, threadgroup size = 128.
 * ======================================================================== */

kernel void kernel_matvec_bf16_metal(
    device float *out [[buffer(0)]],
    const device ushort *A_bf16 [[buffer(1)]],
    const device float *x [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    const int GROUPS_PER_ROW = 4;
    int row = (int)(gid / (32 * GROUPS_PER_ROW));
    if (row >= p.rows) return;

    const device ushort *a_row = A_bf16 + (long)row * p.cols;
    const device float4 *x4 = (const device float4 *)x;
    int lane = (int)tid_in_tg;  /* 0..127 */
    float sum = 0.0f;

    /* 128 threads each handle cols/128 elements (vectorized 4-wide) */
    for (int c = lane * 4; c < p.cols; c += 128 * 4) {
        const device ushort4 &a4 = ((const device ushort4 *)a_row)[c / 4];
        float4 af = float4(bf16_to_f32(a4.x), bf16_to_f32(a4.y),
                           bf16_to_f32(a4.z), bf16_to_f32(a4.w));
        sum += dot(af, x4[c / 4]);
    }

    /* Two-level reduction: simd_sum within each group, then threadgroup */
    sum = simd_sum(sum);

    threadgroup float partial[4];
    if (simd_lane == 0) partial[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_tg == 0) {
        out[row] = partial[0] + partial[1] + partial[2] + partial[3];
    }
}

/* ========================================================================
 * Matrix-Vector Multiply (F32 weights)
 * ======================================================================== */

kernel void kernel_matvec_f32_metal(
    device float *out [[buffer(0)]],
    const device float *A [[buffer(1)]],
    const device float *x [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.rows) return;

    int row = (int)gid;
    const device float *a_row = A + row * p.cols;
    float sum = 0.0f;

    for (int c = 0; c < p.cols; c++) {
        sum += a_row[c] * x[c];
    }

    out[row] = sum;
}

/* ========================================================================
 * Matrix-Matrix Multiply (F32): C[M,N] = A[M,K] @ B[N,K]^T
 * Thread (gid.x, gid.y) computes C[gid.y, gid.x]
 * ======================================================================== */

kernel void kernel_matmul_f32_metal(
    device float *C [[buffer(0)]],
    const device float *A [[buffer(1)]],
    const device float *B [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int n = (int)gid.x;
    int m = (int)gid.y;
    if (m >= p.M || n >= p.N) return;

    const device float *a_row = A + m * p.K;
    const device float *b_row = B + n * p.K;
    float sum = 0.0f;

    for (int k = 0; k < p.K; k++) {
        sum += a_row[k] * b_row[k];
    }

    C[m * p.N + n] = sum;
}

/* ========================================================================
 * Matrix-Matrix Multiply (BF16 B): C[M,N] = A[M,K] @ B_bf16[N,K]^T
 * ======================================================================== */

kernel void kernel_matmul_bf16_metal(
    device float *C [[buffer(0)]],
    const device float *A [[buffer(1)]],
    const device ushort *B_bf16 [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int n = (int)gid.x;
    int m = (int)gid.y;
    if (m >= p.M || n >= p.N) return;

    const device float *a_row = A + m * p.K;
    const device ushort *b_row = B_bf16 + n * p.K;
    float sum = 0.0f;

    for (int k = 0; k < p.K; k++) {
        sum += a_row[k] * bf16_to_f32(b_row[k]);
    }

    C[m * p.N + n] = sum;
}

/* ========================================================================
 * RMSNorm: out[i] = weight[i] * (x[i] / sqrt(mean(x^2) + eps))
 * Uses threadgroup reduction for sum of squares.
 * ======================================================================== */

kernel void kernel_rms_norm_metal(
    device float *out [[buffer(0)]],
    const device float *x [[buffer(1)]],
    const device float *weight [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_sum[1024];

    /* Compute partial sum of squares */
    float local_ss = 0.0f;
    for (int i = (int)tid; i < p.dim; i += (int)tg_size) {
        float v = x[i];
        local_ss += v * v;
    }
    shared_sum[tid] = local_ss;

    /* Reduce within threadgroup */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = 1.0f / sqrt(shared_sum[0] / float(p.dim) + p.eps);

    /* Apply norm */
    for (int i = (int)tid; i < p.dim; i += (int)tg_size) {
        out[i] = x[i] * inv * weight[i];
    }
}

/* RMSNorm in-place */
kernel void kernel_rms_norm_inplace_metal(
    device float *x [[buffer(0)]],
    const device float *weight [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_sum[1024];

    float local_ss = 0.0f;
    for (int i = (int)tid; i < p.dim; i += (int)tg_size) {
        float v = x[i];
        local_ss += v * v;
    }
    shared_sum[tid] = local_ss;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = 1.0f / sqrt(shared_sum[0] / float(p.dim) + p.eps);

    for (int i = (int)tid; i < p.dim; i += (int)tg_size) {
        x[i] = x[i] * inv * weight[i];
    }
}

/* ========================================================================
 * LayerNorm: out[i] = weight[i] * (x[i] - mean) / sqrt(var + eps) + bias[i]
 * ======================================================================== */

kernel void kernel_layer_norm_metal(
    device float *out [[buffer(0)]],
    const device float *x [[buffer(1)]],
    const device float *weight [[buffer(2)]],
    const device float *bias [[buffer(3)]],
    constant Params &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_data[1024];

    /* Compute mean */
    float local_sum = 0.0f;
    for (int i = (int)tid; i < p.dim; i += (int)tg_size) {
        local_sum += x[i];
    }
    shared_data[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_data[tid] += shared_data[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared_data[0] / float(p.dim);

    /* Compute variance */
    float local_var = 0.0f;
    for (int i = (int)tid; i < p.dim; i += (int)tg_size) {
        float d = x[i] - mean;
        local_var += d * d;
    }
    shared_data[tid] = local_var;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_data[tid] += shared_data[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(shared_data[0] / float(p.dim) + p.eps);

    /* Apply norm */
    for (int i = (int)tid; i < p.dim; i += (int)tg_size) {
        float v = (x[i] - mean) * inv;
        if (weight) v *= weight[i];
        if (bias) v += bias[i];
        out[i] = v;
    }
}

/* ========================================================================
 * Softmax (in-place)
 * ======================================================================== */

kernel void kernel_softmax_metal(
    device float *x [[buffer(0)]],
    constant Params &p [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_val[1024];

    /* Find max */
    float local_max = -INFINITY;
    for (int i = (int)tid; i < p.n; i += (int)tg_size) {
        local_max = max(local_max, x[i]);
    }
    shared_val[tid] = local_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_val[tid] = max(shared_val[tid], shared_val[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_val[0];

    /* Compute exp and sum */
    float local_sum = 0.0f;
    for (int i = (int)tid; i < p.n; i += (int)tg_size) {
        float e = exp(x[i] - max_val);
        x[i] = e;
        local_sum += e;
    }
    shared_val[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_val[tid] += shared_val[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_val[0];

    /* Normalize */
    for (int i = (int)tid; i < p.n; i += (int)tg_size) {
        x[i] *= inv_sum;
    }
}

/* ========================================================================
 * Fused QKV matvec (BF16): computes Q, K, V in one dispatch
 * 4 SIMD groups (128 threads) per row for maximum memory bandwidth.
 * Dispatch: total_rows * 128 threads, threadgroup size = 128.
 * ======================================================================== */

kernel void kernel_qkv_matvec_bf16_metal(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    const device ushort *wq_bf16 [[buffer(3)]],
    const device ushort *wk_bf16 [[buffer(4)]],
    const device ushort *wv_bf16 [[buffer(5)]],
    const device float *x [[buffer(6)]],
    constant Params &p [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    const int GROUPS_PER_ROW = 4;
    int q_rows = p.num_heads * p.head_dim;
    int kv_rows = p.kv_heads * p.head_dim;
    int total_rows = q_rows + kv_rows + kv_rows;
    int row = (int)(gid / (32 * GROUPS_PER_ROW));
    if (row >= total_rows) return;

    const device ushort *w_row = nullptr;
    device float *out_ptr = nullptr;
    int out_row = 0;
    if (row < q_rows) {
        w_row = wq_bf16 + (long)row * p.hidden;
        out_ptr = q;
        out_row = row;
    } else if (row < q_rows + kv_rows) {
        int r = row - q_rows;
        w_row = wk_bf16 + (long)r * p.hidden;
        out_ptr = k;
        out_row = r;
    } else {
        int r = row - q_rows - kv_rows;
        w_row = wv_bf16 + (long)r * p.hidden;
        out_ptr = v;
        out_row = r;
    }

    const device float4 *x4 = (const device float4 *)x;
    int lane = (int)tid_in_tg;
    float sum = 0.0f;

    for (int c = lane * 4; c < p.hidden; c += 128 * 4) {
        const device ushort4 &a4 = ((const device ushort4 *)w_row)[c / 4];
        float4 af = float4(bf16_to_f32(a4.x), bf16_to_f32(a4.y),
                           bf16_to_f32(a4.z), bf16_to_f32(a4.w));
        sum += dot(af, x4[c / 4]);
    }

    sum = simd_sum(sum);

    threadgroup float partial[4];
    if (simd_lane == 0) partial[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_tg == 0) out_ptr[out_row] = partial[0] + partial[1] + partial[2] + partial[3];
}

/* ========================================================================
 * Fused QK RMSNorm + RoPE (single token)
 * One threadgroup per head (Q heads first, then KV heads).
 * ======================================================================== */

kernel void kernel_qk_norm_rope_metal(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    const device float *q_norm_weight [[buffer(2)]],
    const device float *k_norm_weight [[buffer(3)]],
    const device float *cos_buf [[buffer(4)]],
    const device float *sin_buf [[buffer(5)]],
    constant Params &p [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    int head = (int)tg_id;
    int total_heads = p.num_heads + p.kv_heads;
    if (head >= total_heads) return;

    bool is_q = head < p.num_heads;
    int local_head = is_q ? head : (head - p.num_heads);
    device float *vec = is_q ? (q + (long)local_head * p.head_dim)
                             : (k + (long)local_head * p.head_dim);
    const device float *weight = is_q ? q_norm_weight : k_norm_weight;

    threadgroup float shared_sum[256];
    float local_ss = 0.0f;
    for (int i = (int)tid; i < p.head_dim; i += (int)tg_size) {
        float v0 = vec[i];
        local_ss += v0 * v0;
    }
    shared_sum[tid] = local_ss;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = 1.0f / sqrt(shared_sum[0] / float(p.head_dim) + p.eps);
    for (int i = (int)tid; i < p.head_dim; i += (int)tg_size) {
        vec[i] = vec[i] * inv * weight[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    int half_dim = p.head_dim / 2;
    for (int i = (int)tid; i < half_dim; i += (int)tg_size) {
        float c = cos_buf[i];
        float s = sin_buf[i];
        float v0 = vec[i];
        float v1 = vec[i + half_dim];
        vec[i] = v0 * c - v1 * s;
        vec[i + half_dim] = v1 * cos_buf[i + half_dim] + v0 * sin_buf[i + half_dim];
    }
}

/* ========================================================================
 * KV cache write for one layer/position
 * kv layout: [layers, kv_max, kv_dim]
 * ======================================================================== */

kernel void kernel_kv_cache_store_metal(
    device float *kv_k [[buffer(0)]],
    device float *kv_v [[buffer(1)]],
    const device float *k [[buffer(2)]],
    const device float *v [[buffer(3)]],
    constant Params &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= p.kv_dim) return;
    long base = ((long)p.layer_idx * p.kv_max + p.pos) * p.kv_dim;
    kv_k[base + i] = k[i];
    kv_v[base + i] = v[i];
}

/* ========================================================================
 * Attention score compute for one token
 * scores layout: [num_heads, kv_max], valid prefix [0, seq_len)
 * ======================================================================== */

kernel void kernel_attn_scores_metal(
    device float *scores [[buffer(0)]],
    const device float *q [[buffer(1)]],
    const device float *kv_k [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int t = (int)gid.x;
    int h = (int)gid.y;
    if (t >= p.seq_len || h >= p.num_heads) return;

    int kv_h = h / p.groups_per_head;
    const device float *qh = q + (long)h * p.head_dim;
    const device float *kh = kv_k + ((long)p.layer_idx * p.kv_max + t) * p.kv_dim +
                             (long)kv_h * p.head_dim;

    float sum = 0.0f;
    for (int i = 0; i < p.head_dim; i++) {
        sum += qh[i] * kh[i];
    }
    scores[(long)h * p.kv_max + t] = sum * p.scale;
}

/* Softmax over each head row in scores[num_heads, kv_max], prefix seq_len */
kernel void kernel_attn_softmax_rows_metal(
    device float *scores [[buffer(0)]],
    constant Params &p [[buffer(1)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if ((int)head >= p.num_heads) return;

    threadgroup float shared_val[256];
    long row_base = (long)head * p.kv_max;

    float local_max = -INFINITY;
    for (int t = (int)tid; t < p.seq_len; t += (int)tg_size) {
        local_max = max(local_max, scores[row_base + t]);
    }
    shared_val[tid] = local_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_val[tid] = max(shared_val[tid], shared_val[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_val[0];

    float local_sum = 0.0f;
    for (int t = (int)tid; t < p.seq_len; t += (int)tg_size) {
        float e = exp(scores[row_base + t] - max_val);
        scores[row_base + t] = e;
        local_sum += e;
    }
    shared_val[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_val[tid] += shared_val[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_val[0];

    for (int t = (int)tid; t < p.seq_len; t += (int)tg_size) {
        scores[row_base + t] *= inv_sum;
    }
}

/* Weighted sum: out[h, d] = sum_t scores[h, t] * v[layer, t, kv_h, d] */
kernel void kernel_attn_weighted_sum_metal(
    device float *out [[buffer(0)]],
    const device float *scores [[buffer(1)]],
    const device float *kv_v [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int d = (int)gid.x;
    int h = (int)gid.y;
    if (d >= p.head_dim || h >= p.num_heads) return;

    int kv_h = h / p.groups_per_head;
    float sum = 0.0f;
    long row_base = (long)h * p.kv_max;
    for (int t = 0; t < p.seq_len; t++) {
        float w = scores[row_base + t];
        const device float *vh = kv_v + ((long)p.layer_idx * p.kv_max + t) * p.kv_dim +
                                 (long)kv_h * p.head_dim;
        sum += w * vh[d];
    }
    out[(long)h * p.head_dim + d] = sum;
}

/* Fused attention for one token: scores + softmax + weighted sum */
kernel void kernel_attn_fused_metal(
    device float *out [[buffer(0)]],
    device float *scores [[buffer(1)]],
    const device float *q [[buffer(2)]],
    const device float *kv_k [[buffer(3)]],
    const device float *kv_v [[buffer(4)]],
    constant Params &p [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if ((int)head >= p.num_heads) return;

    int h = (int)head;
    int kv_h = h / p.groups_per_head;
    const device float *qh = q + (long)h * p.head_dim;
    long row_base = (long)h * p.kv_max;

    threadgroup float shared_val[256];

    /* 1) Scores + max */
    float local_max = -INFINITY;
    for (int t = (int)tid; t < p.seq_len; t += (int)tg_size) {
        const device float *kh = kv_k + ((long)p.layer_idx * p.kv_max + t) * p.kv_dim +
                                 (long)kv_h * p.head_dim;
        float score = 0.0f;
        for (int i = 0; i < p.head_dim; i++) {
            score += qh[i] * kh[i];
        }
        score *= p.scale;
        scores[row_base + t] = score;
        local_max = max(local_max, score);
    }
    shared_val[tid] = local_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_val[tid] = max(shared_val[tid], shared_val[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_val[0];

    /* 2) exp + sum */
    float local_sum = 0.0f;
    for (int t = (int)tid; t < p.seq_len; t += (int)tg_size) {
        float e = exp(scores[row_base + t] - max_val);
        scores[row_base + t] = e;
        local_sum += e;
    }
    shared_val[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_val[tid] += shared_val[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_val[0];

    /* 3) normalize */
    for (int t = (int)tid; t < p.seq_len; t += (int)tg_size) {
        scores[row_base + t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* 4) weighted sum */
    for (int d = (int)tid; d < p.head_dim; d += (int)tg_size) {
        float acc = 0.0f;
        for (int t = 0; t < p.seq_len; t++) {
            float w = scores[row_base + t];
            const device float *vh = kv_v + ((long)p.layer_idx * p.kv_max + t) * p.kv_dim +
                                     (long)kv_h * p.head_dim;
            acc += w * vh[d];
        }
        out[(long)h * p.head_dim + d] = acc;
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

kernel void kernel_silu_inplace_metal(
    device float *x [[buffer(0)]],
    constant Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    float v = x[gid];
    x[gid] = v / (1.0f + exp(-v));
}

kernel void kernel_gelu_inplace_metal(
    device float *x [[buffer(0)]],
    constant Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    float v = x[gid];
    x[gid] = 0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

/* SnakeBeta: out = x + inv_beta * sin^2(alpha * x) */
kernel void kernel_snake_beta_metal(
    device float *out [[buffer(0)]],
    const device float *x [[buffer(1)]],
    const device float *alpha [[buffer(2)]],
    const device float *beta [[buffer(3)]],
    constant Params &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int total = p.channels * p.length;
    if ((int)gid >= total) return;

    int c = (int)gid / p.length;
    float a = alpha[c];
    float inv_b = beta[c];
    float s = sin(x[gid] * a);
    out[gid] = x[gid] + inv_b * s * s;
}

/* ========================================================================
 * Element-wise Operations
 * ======================================================================== */

kernel void kernel_add_inplace_metal(
    device float *a [[buffer(0)]],
    const device float *b [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    a[gid] += b[gid];
}

kernel void kernel_copy_metal(
    device float *dst [[buffer(0)]],
    const device float *src [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    dst[gid] = src[gid];
}

kernel void kernel_mul_inplace_metal(
    device float *a [[buffer(0)]],
    const device float *b [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    a[gid] *= b[gid];
}

kernel void kernel_scale_inplace_metal(
    device float *x [[buffer(0)]],
    constant Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    x[gid] *= p.scale;
}

kernel void kernel_clamp_metal(
    device float *x [[buffer(0)]],
    constant Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    x[gid] = clamp(x[gid], p.min_val, p.max_val);
}

/* ========================================================================
 * RoPE (Rotary Position Embedding)
 * q/k: [num_heads * head_dim], cos/sin: [head_dim]
 * Thread gid maps to (head, dim_pair)
 * ======================================================================== */

kernel void kernel_rope_apply_metal(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    const device float *cos_buf [[buffer(2)]],
    const device float *sin_buf [[buffer(3)]],
    constant Params &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int half_dim = p.head_dim / 2;
    int total = p.num_heads * half_dim;
    if ((int)gid >= total) return;

    int h = (int)gid / half_dim;
    int i = (int)gid % half_dim;

    float c = cos_buf[i];
    float s = sin_buf[i];

    /* Apply to Q */
    int qi = h * p.head_dim + i;
    int qi2 = h * p.head_dim + i + half_dim;
    float q0 = q[qi], q1 = q[qi2];
    q[qi]  = q0 * c - q1 * s;
    q[qi2] = q1 * c + q0 * s;

    /* Apply to K if provided (k buffer might be nil/empty) */
    if (k) {
        int ki = h * p.head_dim + i;
        int ki2 = h * p.head_dim + i + half_dim;
        float k0 = k[ki], k1 = k[ki2];
        k[ki]  = k0 * c - k1 * s;
        k[ki2] = k1 * c + k0 * s;
    }
}

/* ========================================================================
 * Causal Conv1d
 * Input: [in_channels, length], Weight: [out_channels, in_channels/groups, kernel_size]
 * Output: [out_channels, length]
 * Thread (gid.x, gid.y) = (time, out_channel)
 * ======================================================================== */

kernel void kernel_causal_conv1d_metal(
    device float *out [[buffer(0)]],
    const device float *input [[buffer(1)]],
    const device float *weight [[buffer(2)]],
    const device float *bias [[buffer(3)]],
    constant Params &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int t = (int)gid.x;
    int oc = (int)gid.y;
    if (t >= p.length || oc >= p.out_channels) return;

    int eff_kernel = (p.kernel_size - 1) * p.dilation + 1;
    int pad = eff_kernel - 1;
    int ch_per_group = p.in_channels / p.groups;
    int out_per_group = p.out_channels / p.groups;
    int g = oc / out_per_group;
    int ic_base = g * ch_per_group;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < ch_per_group; ic++) {
        const device float *w = weight + ((long)oc * ch_per_group + ic) * p.kernel_size;
        const device float *in_ch = input + (long)(ic_base + ic) * p.length;

        for (int k = 0; k < p.kernel_size; k++) {
            int in_t = t - pad + k * p.dilation;
            if (in_t >= 0 && in_t < p.length) {
                sum += w[k] * in_ch[in_t];
            }
        }
    }

    out[(long)oc * p.length + t] = sum;
}

/* ========================================================================
 * Transposed Conv1d (for upsampling)
 * Input: [in_channels, length], Weight: [in_channels, out_channels, kernel_size]
 * Output: [out_channels, final_len]
 * Thread (gid.x, gid.y) = (out_time, out_channel)
 * p.n = final_len
 * ======================================================================== */

kernel void kernel_transposed_conv1d_metal(
    device float *out [[buffer(0)]],
    const device float *input [[buffer(1)]],
    const device float *weight [[buffer(2)]],
    const device float *bias [[buffer(3)]],
    constant Params &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int ot = (int)gid.x;
    int oc = (int)gid.y;
    int final_len = p.n;
    if (ot >= final_len || oc >= p.out_channels) return;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < p.in_channels; ic++) {
        const device float *in_ch = input + (long)ic * p.length;
        const device float *w = weight + (long)ic * p.out_channels * p.kernel_size + (long)oc * p.kernel_size;

        for (int k = 0; k < p.kernel_size; k++) {
            /* For transposed conv: out[ot] gets contribution from in[t] where t*stride + k == ot */
            int rem = ot - k;
            if (rem >= 0 && rem % p.stride == 0) {
                int t = rem / p.stride;
                if (t < p.length) {
                    sum += in_ch[t] * w[k];
                }
            }
        }
    }

    out[(long)oc * final_len + ot] = sum;
}

/* ========================================================================
 * BF16 to F32 Conversion
 * ======================================================================== */

kernel void kernel_bf16_to_f32_metal(
    device float *out [[buffer(0)]],
    const device ushort *in [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    out[gid] = bf16_to_f32(in[gid]);
}

/* ========================================================================
 * Argmax over float vector
 * ======================================================================== */

kernel void kernel_argmax_i32_metal(
    device int *out_idx [[buffer(0)]],
    const device float *x [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_val[256];
    threadgroup int shared_idx[256];

    if (p.n <= 0) {
        if (tid == 0) out_idx[0] = 0;
        return;
    }

    float best_val = -INFINITY;
    int best_idx = 0;
    for (int i = (int)tid; i < p.n; i += (int)tg_size) {
        float v = x[i];
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }
    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v0 = shared_val[tid];
            int i0 = shared_idx[tid];
            float v1 = shared_val[tid + s];
            int i1 = shared_idx[tid + s];
            if (v1 > v0 || (v1 == v0 && i1 < i0)) {
                shared_val[tid] = v1;
                shared_idx[tid] = i1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_idx[0] = shared_idx[0];
    }
}

/* ========================================================================
 * Embed from Argmax (BF16)
 * Reads an argmax token from a GPU buffer and uses it to index a BF16
 * embedding table, producing a F32 embedding vector. Enables zero-sync
 * chaining of autoregressive subtalker steps on GPU.
 * Dispatch: dim threads.
 * ======================================================================== */

kernel void kernel_embed_from_argmax_bf16_metal(
    device float *out [[buffer(0)]],
    const device ushort *embeddings [[buffer(1)]],
    const device int *token_buf [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.n) return;
    int token = token_buf[0];
    out[gid] = bf16_to_f32(embeddings[(uint)token * (uint)p.n + gid]);
}

/* ========================================================================
 * Scatter Int: copy a single int from src[0] to dst[idx].
 * Used to store argmax results into a codes buffer on GPU.
 * Dispatch: 1 thread.
 * ======================================================================== */

kernel void kernel_scatter_int_metal(
    device int *dst [[buffer(0)]],
    const device int *src [[buffer(1)]],
    constant Params &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        dst[p.n] = src[0];  /* p.n = destination index */
    }
}

/* ========================================================================
 * SwiGLU Fused Matvec (BF16)
 * gate_up_fused: [2*intermediate, hidden] BF16
 * out[i] = silu(gate[i]) * up[i]
 * 4 SIMD groups (128 threads) per output row for maximum memory bandwidth.
 * Dispatch: intermediate * 128 threads, threadgroup size = 128.
 * ======================================================================== */

kernel void kernel_swiglu_matvec_bf16_metal(
    device float *out [[buffer(0)]],
    const device ushort *gate_up_bf16 [[buffer(1)]],
    const device float *x [[buffer(2)]],
    constant Params &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    const int GROUPS_PER_ROW = 4;
    int i = (int)(gid / (32 * GROUPS_PER_ROW));
    if (i >= p.intermediate) return;

    int lane = (int)tid_in_tg;
    const device float4 *x4 = (const device float4 *)x;

    /* Compute gate = gate_weights[i] @ x */
    const device ushort *gate_row = gate_up_bf16 + (long)i * p.hidden;
    float gate_sum = 0.0f;
    for (int c = lane * 4; c < p.hidden; c += 128 * 4) {
        const device ushort4 &a4 = ((const device ushort4 *)gate_row)[c / 4];
        float4 af = float4(bf16_to_f32(a4.x), bf16_to_f32(a4.y),
                           bf16_to_f32(a4.z), bf16_to_f32(a4.w));
        gate_sum += dot(af, x4[c / 4]);
    }
    gate_sum = simd_sum(gate_sum);

    /* Compute up = up_weights[i] @ x */
    const device ushort *up_row = gate_up_bf16 + ((long)p.intermediate + i) * p.hidden;
    float up_sum = 0.0f;
    for (int c = lane * 4; c < p.hidden; c += 128 * 4) {
        const device ushort4 &a4 = ((const device ushort4 *)up_row)[c / 4];
        float4 af = float4(bf16_to_f32(a4.x), bf16_to_f32(a4.y),
                           bf16_to_f32(a4.z), bf16_to_f32(a4.w));
        up_sum += dot(af, x4[c / 4]);
    }
    up_sum = simd_sum(up_sum);

    /* Two-level reduction: simd_sum done, now threadgroup reduction */
    threadgroup float partial_gate[4];
    threadgroup float partial_up[4];
    if (simd_lane == 0) {
        partial_gate[simd_id] = gate_sum;
        partial_up[simd_id] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_tg == 0) {
        float gate_val = partial_gate[0] + partial_gate[1] + partial_gate[2] + partial_gate[3];
        float up_val = partial_up[0] + partial_up[1] + partial_up[2] + partial_up[3];
        out[i] = (gate_val / (1.0f + exp(-gate_val))) * up_val;
    }
}
