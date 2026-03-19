#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declare metal-cpp types to avoid header collision
namespace MTL {
class Device;
class Buffer;
class CommandQueue;
} // namespace MTL

class SafeTensorsLoader;

class FluxGraph {
private:
  struct Impl;
  std::unique_ptr<Impl> pImpl;

public:
  FluxGraph(MTL::Device *mtl_dev, SafeTensorsLoader &w,
            const char *mlib_path = nullptr);
  ~FluxGraph();

  bool build_double_block(int i);
  bool build_single_block(int i);

  // Execution Methods
  bool compile();
  void execute(MTL::CommandQueue *queue, void *input_img, void *input_txt,
               void *output_img, float timestep, float guidance, int i_len,
               int t_len);
  void generate(MTL::CommandQueue *queue, void *input_img, void *input_txt,
                void *output_img, float guidance, int steps, int i_len,
                int t_len);

  bool build_full_graph();
  bool build_layer(int i);
};

#ifdef __cplusplus
extern "C" {
#endif

/// Production Shared Library API
/// These functions are intended for use via dlopen / ctypes / Swift

typedef void *FluxEngineHandle;

FluxEngineHandle flux_engine_create(void *mtl_device, const char *model_path);
bool flux_engine_compile(FluxEngineHandle handle);
void flux_engine_execute(FluxEngineHandle handle, void *mtl_queue, void *img_in,
                         void *txt_in, void *out, float timestep,
                         float guidance, int i_len, int t_len);
void flux_engine_generate(FluxEngineHandle handle, void *mtl_queue,
                          void *img_in, void *txt_in, void *out, float guidance,
                          int steps, int i_len, int t_len);
/// Standalone AMX Operations
void amx_linear(void *cb_ptr, void *x_ptr, uint64_t ox, void *w_ptr,
                uint64_t ow, void *o_ptr, uint64_t oo, uint32_t M, uint32_t N,
                uint32_t K, uint32_t acc, uint32_t is_bfloat,
                const char *lib_p = "default.metallib");

void amx_sdpa(void *enc_ptr, void *qkv_ptr, uint32_t o_q, void *out_ptr,
              uint32_t o_o, uint32_t seq_len, uint32_t num_heads,
              uint32_t head_dim, float scale,
              const char *lib_p = "default.metallib");

void amx_sdpa_v2(void *enc_ptr, void *q_ptr, uint32_t o_q, void *k_ptr,
                 uint32_t o_k, void *v_ptr, uint32_t o_v, void *out_ptr,
                 uint32_t o_o, uint32_t seq_len, uint32_t num_heads,
                 uint32_t head_dim, float scale,
                 const char *lib_p = "default.metallib", uint32_t is_bfloat = 0);

#ifdef __cplusplus
}
#endif
