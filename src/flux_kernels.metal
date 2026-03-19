#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// System-wide saturation for FP16 stability
inline half saturate_cast(float v) {
    return (half)clamp(v, -65504.0f, 65504.0f);
}

struct LinearParams { uint32_t M; uint32_t N; uint32_t K; uint32_t out_offset; uint32_t accumulate; };

// -----------------------------------------------------------------------------
// SDPA INTERLEAVED [S, H, D]
// -----------------------------------------------------------------------------

// Tiling: 8 queries per threadgroup (32 threads / 1 SIMD group)

template <typename T>
inline void sdpa_v3_impl(device const T *q_ptr,
                         device const T *k_ptr,
                         device const T *v_ptr,
                         device T *out_ptr, 
                         constant uint32_t *params, 
                         uint3 gid, uint tid,
                         threadgroup half q_s[4][8][136],
                         threadgroup half k_s[32][136],
                         threadgroup half v_s[32][136],
                         threadgroup float scratch_f[1024],
                         threadgroup half scratch_h[1024]) {
    const uint32_t p_seq_len = params[0], p_num_heads = params[1], p_head_dim = params[2];
    const float p_scale = as_type<float>(params[4]);
    const uint h_idx = gid.z, q_base = gid.y * 32;
    const uint S_stride = p_num_heads * p_head_dim;
    const uint s_idx = tid / 32;
    const uint lane = tid % 32;

    // Load Q tile (32 queries across 4 SIMD groups)
    for (uint i = tid; i < 4096; i += 128) {
        uint g = i / 1024;
        uint r = (i % 1024) / 128;
        uint c = i % 128;
        uint q_idx = q_base + g * 8 + r;
        if (q_idx < p_seq_len) q_s[g][r][c] = (half)q_ptr[q_idx * S_stride + h_idx * p_head_dim + c];
        else q_s[g][r][c] = 0.0h;
    }

    simdgroup_float8x8 acc[16];
    for(int i=0; i<16; i++) acc[i] = simdgroup_float8x8(0.0f);
    float l[8], m[8];
    for (int i=0; i<8; i++) { l[i] = 0.0f; m[i] = -1e30f; }

    for (uint kv_base = 0; kv_base < p_seq_len; kv_base += 32) {
        // Load K, V tiles (32 tokens) - 128 threads load 4096 elements
        for (uint i = tid; i < 4096; i += 128) {
            uint r = i / 128, c = i % 128;
            uint s_idx_kv = kv_base + r;
            if (s_idx_kv < p_seq_len) {
                uint off = s_idx_kv * S_stride + h_idx * p_head_dim + c;
                k_s[r][c] = (half)k_ptr[off];
                v_s[r][c] = (half)v_ptr[off];
            } else { k_s[r][c] = 0.0h; v_s[r][c] = 0.0h; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Q * K^T
        for (int kb=0; kb<32; kb+=8) {
            simdgroup_float8x8 s = simdgroup_float8x8(0.0f);
            for (uint d=0; d<128; d+=8) {
                simdgroup_half8x8 qm, km;
                simdgroup_load(qm, &q_s[s_idx][0][d], 136, ulong2(0), false);
                simdgroup_load(km, &k_s[kb][d], 136, ulong2(0), true);
                simdgroup_multiply_accumulate(s, qm, km, s);
            }
            simdgroup_store(s, &scratch_f[s_idx * 256 + kb * 8], 8, ulong2(0), false);
        }

        // Online Softmax
        float alpha_reg[8], m_old[8];
        for (int r=0; r<8; r++) {
            float row_max = scratch_f[s_idx*256 + r*8 + (lane/8)*64 + (lane%8)] * p_scale;
            row_max = (kv_base + lane < p_seq_len) ? row_max : -1e30f;
            simdgroup_barrier(mem_flags::mem_none);
            row_max = max(row_max, simd_shuffle_down(row_max, 16));
            row_max = max(row_max, simd_shuffle_down(row_max, 8));
            row_max = max(row_max, simd_shuffle_down(row_max, 4));
            row_max = max(row_max, simd_shuffle_down(row_max, 2));
            row_max = max(row_max, simd_shuffle_down(row_max, 1));
            row_max = simd_broadcast(row_max, 0);
            
            m_old[r] = m[r];
            m[r] = max(m_old[r], row_max);
            alpha_reg[r] = exp(m_old[r] - m[r]);
            
            float val = (kv_base + lane < p_seq_len) ? exp(clamp(scratch_f[s_idx*256 + r*8 + (lane/8)*64 + (lane%8)] * p_scale - m[r], -80.0f, 0.0f)) : 0.0f;
            scratch_h[s_idx*256 + r*8 + (lane/8)*64 + (lane%8)] = (half)val;
            
            float row_sum = val;
            simdgroup_barrier(mem_flags::mem_none);
            row_sum += simd_shuffle_down(row_sum, 16);
            row_sum += simd_shuffle_down(row_sum, 8);
            row_sum += simd_shuffle_down(row_sum, 4);
            row_sum += simd_shuffle_down(row_sum, 2);
            row_sum += simd_shuffle_down(row_sum, 1);
            row_sum = simd_broadcast(row_sum, 0);
            
            l[r] = l[r] * alpha_reg[r] + row_sum;
        }

        // Scale Accumulator
        for (int h=0; h<16; h++) {
            simdgroup_store(acc[h], &scratch_f[s_idx * 256], 8, ulong2(0,0), false);
            simdgroup_barrier(mem_flags::mem_none);
            if (lane < 32) {
                #pragma unroll
                for (int i=0; i<2; i++) {
                    uint l_off = lane + i*32;
                    uint rr = l_off / 8;
                    scratch_f[s_idx * 256 + l_off] *= alpha_reg[rr];
                }
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(acc[h], &scratch_f[s_idx * 256], 8, ulong2(0,0), false);
        }

        // P * V
        for (int kb=0; kb<32; kb+=8) {
            simdgroup_half8x8 p_mat;
            simdgroup_load(p_mat, &scratch_h[s_idx * 256 + kb * 8], 8, ulong2(0), false);
            #pragma unroll
            for (int h_chunk=0; h_chunk<16; h_chunk++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, &v_s[kb][h_chunk * 8], 136, ulong2(0), false);
                simdgroup_multiply_accumulate(acc[h_chunk], p_mat, vm, acc[h_chunk]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store Output
    if (q_base + s_idx * 8 < p_seq_len) {
        #pragma unroll
        for (int h_chunk=0; h_chunk<16; h_chunk++) {
            simdgroup_store(acc[h_chunk], &scratch_f[s_idx * 256], 8, ulong2(0), false);
            simdgroup_barrier(mem_flags::mem_none);
            for (int r=0; r<8; r++) {
                if (q_base + s_idx * 8 + r < p_seq_len) {
                    float inv_l = 1.0f / (l[r] + 1e-10f);
                    if (lane < 8) {
                        out_ptr[(q_base + s_idx * 8 + r) * S_stride + h_idx * p_head_dim + h_chunk * 8 + lane] = (T)(scratch_f[s_idx * 256 + r*8 + lane] * inv_l);
                    }
                }
            }
        }
    }
}
kernel void sdpa_h16_kernel(device const half *q [[buffer(0)]], 
                             device const half *k [[buffer(1)]],
                             device const half *v [[buffer(2)]],
                             device half *ot [[buffer(3)]], 
                             constant uint32_t *p [[buffer(4)]], 
                             uint3 gid [[threadgroup_position_in_grid]], 
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half q_s[4][8][136], k_s[32][136], v_s[32][136];
    threadgroup float scratch_f[1024];
    threadgroup half scratch_h[1024];
    sdpa_v3_impl<half>(q, k, v, ot, p, gid, tid, q_s, k_s, v_s, scratch_f, scratch_h);
}

kernel void sdpa_bf16_kernel(device const bfloat *q [[buffer(0)]], 
                             device const bfloat *k [[buffer(1)]],
                             device const bfloat *v [[buffer(2)]],
                             device bfloat *ot [[buffer(3)]], 
                             constant uint32_t *p [[buffer(4)]], 
                             uint3 gid [[threadgroup_position_in_grid]], 
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half q_s[4][8][136], k_s[32][136], v_s[32][136];
    threadgroup float scratch_f[1024];
    threadgroup half scratch_h[1024];
    sdpa_v3_impl<bfloat>(q, k, v, ot, p, gid, tid, q_s, k_s, v_s, scratch_f, scratch_h);
}

kernel void rope_h16_kernel(device half *x [[buffer(0)]], device const float *freqs [[buffer(1)]], constant uint &rope_dim [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    // Tiling: x is [S, 3, H, D] or [S, H, D]
    // Assume r = seq_idx, head_idx = gid.y, d = gid.x
    // gid.x = d_idx / 2 (0..rope_dim/2)
    // gid.y = token_idx * num_heads + head_idx
    const uint num_heads = 32; // Inferred from graph
    const uint head_dim = 128;
    const uint total_heads = (uint)gid.y;
    const uint r = total_heads / num_heads; // seq_idx
    const uint c = gid.x; // pair_idx
    
    if (c >= rope_dim / 2) return;
    
    float f = freqs[r * (rope_dim / 2) + c];
    float cos_f = cos(f), sin_f = sin(f);
    
    // x layout is [S, 3, H, D]. Q is at j=0, K at j=1.
    // In our Graph, tmp_h + oc is [S, 3, H, D] interleaved?
    // Let's assume the layout seen in SDPA: stride L3 = 3 * H * D
    const uint L3 = 3 * num_heads * head_dim;
    const uint head_off = (total_heads % num_heads) * head_dim;
    
    // Apply to Q
    device half *q = &x[r * L3 + head_off];
    float2 vq = float2(q[c], q[c + rope_dim / 2]);
    q[c] = (half)(vq.x * cos_f - vq.y * sin_f);
    q[c + rope_dim / 2] = (half)(vq.x * sin_f + vq.y * cos_f);
    
    // Apply to K
    device half *k = &x[r * L3 + num_heads * head_dim + head_off];
    float2 vk = float2(k[c], k[c + rope_dim / 2]);
    k[c] = (half)(vk.x * cos_f - vk.y * sin_f);
    k[c + rope_dim / 2] = (half)(vk.x * sin_f + vk.y * cos_f);
}

kernel void linear_h16_kernel(device const half *x [[buffer(0)]], device const half *w [[buffer(1)]], device half *ot [[buffer(2)]], constant LinearParams &p [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint MT = gid.y * 32, NT = gid.x * 64; if (MT >= p.M || NT >= p.N) return;
    const uint sgi = tid / 32, sgr = sgi * 8; 
    simdgroup_float8x8 acc[8]; for(int i=0; i<8; i++) acc[i] = simdgroup_float8x8(0.0f);
    
    threadgroup union {
        struct { half xs[32][33]; half ws[32][65]; } load;
        float rg[32][65];
    } scratch;

    for (uint k = 0; k < p.K; k += 32) {
        // Load xs[32][32] and ws[32][64]
        // xs: 32*32 = 1024. ws: 32*64 = 2048. Total 3072. 24/thread.
        for (int i = 0; i < 24; i++) {
            uint idx = i * 128 + tid;
            if (idx < 1024) { // xs
                uint r = idx / 32, c = idx % 32;
                if (MT + r < p.M && k + c < p.K) scratch.load.xs[r][c] = x[(MT + r) * p.K + k + c];
                else scratch.load.xs[r][c] = 0.0h;
            } else if (idx < 3072) { // ws: transposed load — coalesced along K
                uint tidx = idx - 1024;
                uint r = tidx % 32, c = tidx / 32; // adjacent threads read consecutive K addresses
                if (NT + c < p.N && k + r < p.K) scratch.load.ws[r][c] = w[(NT + c) * p.K + k + r];
                else scratch.load.ws[r][c] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int ki=0; ki<32; ki+=8) {
            simdgroup_half8x8 xm; simdgroup_load(xm, &scratch.load.xs[sgr][ki], 33, ulong2(0, 0), false);
            #pragma unroll
            for (int i=0; i<8; i++) {
                simdgroup_half8x8 wm; simdgroup_load(wm, &scratch.load.ws[ki][i*8], 65, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(acc[i], xm, wm, acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (int i=0; i<8; i++) simdgroup_store(acc[i], &scratch.rg[sgr][i*8], 65, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint i = tid; i < 2048; i += 128) {
        uint rr = i / 64, rc = i % 64; 
        if (MT + rr < p.M && NT + rc < p.N) {
            float v = scratch.rg[rr][rc];
            if (p.accumulate) v += (float)ot[(MT + rr) * p.N + NT + rc];
            ot[(MT + rr) * p.N + NT + rc] = saturate_cast(v);
        }
    }
}


kernel void linear_bf16_kernel(device const bfloat *x [[buffer(0)]], device const bfloat *w [[buffer(1)]], device bfloat *ot [[buffer(2)]], constant LinearParams &p [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint MT = gid.y * 32, NT = gid.x * 64; if (MT >= p.M || NT >= p.N) return;
    const uint sgi = tid / 32, sgr = sgi * 8; 
    simdgroup_float8x8 acc[8]; for(int i=0; i<8; i++) acc[i] = simdgroup_float8x8(0.0f);
    
    threadgroup union {
        struct { bfloat xs[32][33]; bfloat ws[32][65]; } load;
        float rg[32][65];
    } scratch;

    for (uint k = 0; k < p.K; k += 32) {
        // Load xs[32][32] and ws[32][64]
        // xs: 32*32 = 1024. ws: 32*64 = 2048. Total 3072. 24/thread.
        for (int i = 0; i < 24; i++) {
            uint idx = i * 128 + tid;
            if (idx < 1024) { 
                uint r = idx / 32, c = idx % 32;
                if (MT + r < p.M && k + c < p.K) scratch.load.xs[r][c] = x[(MT + r) * p.K + k + c];
                else scratch.load.xs[r][c] = 0.0bf;
            } else if (idx < 3072) { // ws: transposed load — coalesced along K
                uint tidx = idx - 1024; 
                uint r = tidx % 32, c = tidx / 32; // adjacent threads read consecutive K addresses
                if (NT + c < p.N && k + r < p.K) scratch.load.ws[r][c] = w[(NT + c) * p.K + k + r];
                else scratch.load.ws[r][c] = 0.0bf;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int ki=0; ki<32; ki+=8) {
            simdgroup_bfloat8x8 xm; simdgroup_load(xm, &scratch.load.xs[sgr][ki], 33, ulong2(0, 0), false);
            #pragma unroll
            for (int i=0; i<8; i++) {
                simdgroup_bfloat8x8 wm; simdgroup_load(wm, &scratch.load.ws[ki][i*8], 65, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(acc[i], xm, wm, acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (int i=0; i<8; i++) simdgroup_store(acc[i], &scratch.rg[sgr][i*8], 65, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint i = tid; i < 2048; i += 128) {
        uint rr = i / 64, rc = i % 64; 
        if (MT + rr < p.M && NT + rc < p.N) {
            float v = scratch.rg[rr][rc];
            if (p.accumulate) v += (float)ot[(MT + rr) * p.N + NT + rc];
            ot[(MT + rr) * p.N + NT + rc] = (bfloat)v;
        }
    }
}

kernel void copy_buffer_f32_kernel(device const half *sr [[buffer(0)]], device half *ds [[buffer(1)]], constant uint &l [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < l) ds[gid] = sr[gid]; }

kernel void rm_h16_kernel(device const half *x [[buffer(0)]], device half *ot [[buffer(1)]], device const half *w [[buffer(2)]], constant uint &dim [[buffer(3)]], uint gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint r = gid; const uint ltid = tid % 256; threadgroup float ssq[256]; float ls = 0.0f; for (uint i = ltid; i < dim; i += 256) { float v = (float)x[r * dim + i]; ls += v * v; } ssq[ltid] = ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) { if (ltid < s) ssq[ltid] += ssq[ltid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float ir = rsqrt(ssq[0] / dim + 1e-6f); for (uint i = ltid; i < dim; i += 256) ot[r * dim + i] = saturate_cast((float)x[r * dim + i] * ir * (w ? (float)w[i] : 1.0f));
}

kernel void silu_h16_kernel(device half *x [[buffer(0)]], constant uint &l [[buffer(1)]], uint gid [[thread_position_in_grid]]) { if (gid < l) { float v = (float)x[gid]; x[gid] = saturate_cast(v / (1.0f + exp(-v))); } }

kernel void modulation_h16_kernel(device half *x [[buffer(0)]], device const half *m [[buffer(1)]], constant uint &dim [[buffer(2)]], constant bool &isg [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]]) {
    uint r = gid.y; uint c = gid.x; if (c < dim) { if (isg) x[r * dim + c] = saturate_cast((float)x[r * dim + c] * tanh((float)m[2 * dim + c])); else { float sh = (float)m[0 * dim + c]; float sl = (float)m[1 * dim + c]; x[r * dim + c] = saturate_cast((float)x[r * dim + c] * (1.0f + sl) + sh); } }
}

kernel void add_h16_kernel(device half *ot [[buffer(0)]], device const half *x [[buffer(1)]], uint gid [[thread_position_in_grid]]) { ot[gid] = saturate_cast((float)ot[gid] + (float)x[gid]); }

kernel void gated_add_h16_kernel(device half *x [[buffer(0)]], device const half *d [[buffer(1)]], device const half *m [[buffer(2)]], constant uint &K [[buffer(3)]], constant uint &gi [[buffer(4)]], uint2 gid [[threadgroup_position_in_grid]]) {
    uint r = gid.y, c = gid.x; if (c < K) x[r * K + c] = saturate_cast((float)x[r * K + c] + (float)d[r * K + c] * tanh((float)m[gi * K + c]));
}

kernel void gsilu_h16_kernel(device half *ot [[buffer(0)]], device const half *x [[buffer(1)]], constant uint &K [[buffer(2)]], uint2 gid [[threadgroup_position_in_grid]]) {
    uint r = gid.y, c = gid.x; if (c < K) { float v = (float)x[r * 2 * K + c]; ot[r * K + c] = saturate_cast(v / (1.0f + exp(-v)) * (float)x[r * 2 * K + K + c]); }
}

kernel void euler_h16_kernel(device half *x [[buffer(0)]], device const half *v [[buffer(1)]], constant float &dt [[buffer(2)]], uint gid [[thread_position_in_grid]]) { x[gid] = saturate_cast((float)x[gid] + dt * (float)v[gid]); }

kernel void empty_kernel() {}
