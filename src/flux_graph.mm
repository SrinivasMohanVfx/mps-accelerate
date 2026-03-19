#include <Metal/Metal.h>
#include <iostream>
#include "flux_graph.h"

// Production build: Simple heartbeat for load confirmation
__attribute__((constructor))
static void metalflux_core_init() {
    std::cout << ">> [MetalFlux] metalflux_core library loaded into process." << std::endl;
}

extern "C" {

void amx_sdpa_v2(void *enc_ptr, void *q_ptr, uint32_t o_q, void *k_ptr,
                 uint32_t o_k, void *v_ptr, uint32_t o_v, void *out_ptr,
                 uint32_t o_o, uint32_t seq_len, uint32_t num_heads,
                 uint32_t head_dim, float scale, const char *lib_p, uint32_t is_bfloat) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)enc_ptr;
    if (!enc) return;

    id<MTLBuffer> qb = (id<MTLBuffer>)q_ptr;
    id<MTLBuffer> kb = (id<MTLBuffer>)k_ptr;
    id<MTLBuffer> vb = (id<MTLBuffer>)v_ptr;
    id<MTLBuffer> out_b = (id<MTLBuffer>)out_ptr;

    static id<MTLComputePipelineState> pso_h16 = nil;
    static id<MTLComputePipelineState> pso_bf16 = nil;
    
    id<MTLComputePipelineState> &pso = is_bfloat ? pso_bf16 : pso_h16;
    NSString *kernel_name = is_bfloat ? @"sdpa_bf16_kernel" : @"sdpa_h16_kernel";

    if (!pso) {
        NSError *error = nil;
        id<MTLDevice> device = enc.device;
        if (!device) return;

        id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:lib_p]] error:&error];
        if (!lib) return;

        id<MTLFunction> func = [lib newFunctionWithName:kernel_name];
        if (!func) return;

        pso = [device newComputePipelineStateWithFunction:func error:&error];
        if (!pso) return;
    }

    [enc setComputePipelineState:pso];
    [enc setBuffer:qb offset:o_q atIndex:0];
    [enc setBuffer:kb offset:o_k atIndex:1];
    [enc setBuffer:vb offset:o_v atIndex:2];
    [enc setBuffer:out_b offset:o_o atIndex:3];

    uint32_t p[8] = {seq_len, num_heads, head_dim, seq_len * head_dim, 0, 0, 0, 0};
    memcpy(&p[4], &scale, 4);
    [enc setBytes:p length:sizeof(p) atIndex:4];

    MTLSize grid = MTLSizeMake(1, (seq_len + 31) / 32, num_heads);
    MTLSize threadgroup = MTLSizeMake(128, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
}

// Stub for legacy API compatibility
void amx_linear(void *enc_ptr, void *x_ptr, uint64_t ox, void *w_ptr,
                uint64_t ow, void *o_ptr, uint64_t oo, uint32_t M, uint32_t N,
                uint32_t K, uint32_t acc, uint32_t is_bfloat, const char *lib_p) {
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)enc_ptr;
    if (!enc) return;

    id<MTLBuffer> xb = (id<MTLBuffer>)x_ptr;
    id<MTLBuffer> wb = (id<MTLBuffer>)w_ptr;
    id<MTLBuffer> out_b = (id<MTLBuffer>)o_ptr;

    static id<MTLComputePipelineState> pso_h16 = nil;
    static id<MTLComputePipelineState> pso_bf16 = nil;
    
    id<MTLComputePipelineState> &pso = is_bfloat ? pso_bf16 : pso_h16;
    NSString *kernel_name = is_bfloat ? @"linear_bf16_kernel" : @"linear_h16_kernel";

    if (!pso) {
        NSError *error = nil;
        id<MTLDevice> device = enc.device;
        if (!device) return;
        
        // Use the lib_path passed from Python (resolves to the custom node directory)
        NSString *libPath = [NSString stringWithUTF8String:lib_p];
        id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        if (!lib) {
            std::cout << ">> [MetalFlux] ERROR: Failed to load metallib from: " << lib_p << std::endl;
            if (error) std::cout << ">> [MetalFlux] Error: " << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }
        std::cout << ">> [MetalFlux] Loaded metallib from: " << lib_p << std::endl;
        
        id<MTLFunction> func = [lib newFunctionWithName:kernel_name];
        if (!func) {
            std::cout << ">> [MetalFlux] ERROR: Kernel function not found: " << [kernel_name UTF8String] << std::endl;
            return;
        }
        pso = [device newComputePipelineStateWithFunction:func error:&error];
        if (!pso) return;
    }

    [enc setComputePipelineState:pso];
    [enc setBuffer:xb offset:ox atIndex:0];
    [enc setBuffer:wb offset:ow atIndex:1];
    [enc setBuffer:out_b offset:oo atIndex:2];

    uint32_t p[5] = {M, N, K, 0, acc};
    [enc setBytes:p length:sizeof(p) atIndex:3];

    MTLSize grid = MTLSizeMake((N + 63) / 64, (M + 31) / 32, 1);
    MTLSize threadgroup = MTLSizeMake(128, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
}

void amx_sdpa(void *enc_ptr, void *qkv_ptr, uint32_t o_q, void *out_ptr,
              uint32_t o_o, uint32_t seq_len, uint32_t num_heads,
              uint32_t head_dim, float scale, const char *lib_p) {
}

} // extern "C"

