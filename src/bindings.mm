#include <torch/extension.h>
#include <iostream>
#include <chrono>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "flux_graph.h"

namespace py = pybind11;

// Helper to get MTLBuffer from torch::Tensor
inline id<MTLBuffer> get_buffer(const torch::Tensor &t) {
  return (id<MTLBuffer>)t.storage().data();
}

inline uint64_t get_offset(const torch::Tensor &t) {
  return (uint64_t)t.storage_offset() * t.element_size();
}

void py_mps_sdpa(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
                 float scale, std::string lib_path) {
  if (!q.is_mps() || !k.is_mps() || !v.is_mps() || !out.is_mps()) {
    throw std::runtime_error("All tensors must be on MPS device.");
  }
  
  id<MTLComputeCommandEncoder> enc =
      at::mps::getCurrentMPSStream()->commandEncoder();

  id<MTLBuffer> mtl_q = get_buffer(q);
  id<MTLBuffer> mtl_k = get_buffer(k);
  id<MTLBuffer> mtl_v = get_buffer(v);
  id<MTLBuffer> mtl_out = get_buffer(out);

  uint32_t off_q = get_offset(q);
  uint32_t off_k = get_offset(k);
  uint32_t off_v = get_offset(v);
  uint32_t off_o = get_offset(out);

  uint32_t heads, seq, dim;
  if (q.dim() == 4) {
      heads = q.size(1);
      seq = q.size(2);
      dim = q.size(3);
  } else {
      seq = q.size(0);
      heads = q.size(1);
      dim = q.size(2);
  }

  uint32_t is_bfloat = q.dtype() == torch::kBFloat16 ? 1 : 0;

  amx_sdpa_v2((void *)enc, (void *)mtl_q, off_q, (void *)mtl_k, off_k,
              (void *)mtl_v, off_v, (void *)mtl_out, off_o, seq, heads, dim,
              scale, lib_path.empty() ? "default.metallib" : lib_path.c_str(), is_bfloat);
}

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

void py_mps_linear(torch::Tensor x, torch::Tensor w, torch::Tensor out,
                   bool accumulate, std::string lib_path) {
  if (!x.is_mps() || !w.is_mps() || !out.is_mps()) {
    throw std::runtime_error("All tensors must be on MPS device.");
  }
  
  // Get MPS stream
  auto stream = at::mps::getCurrentMPSStream();
  
  // Flush the current compute encoder so we can use MPSMatrixMultiplication
  // which encodes directly to the command buffer (not a compute encoder)
  stream->endKernelCoalescing();
  
  id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  
  // Extract buffers and offsets
  id<MTLBuffer> xb = get_buffer(x);
  id<MTLBuffer> wb = get_buffer(w);
  id<MTLBuffer> ob = get_buffer(out);
  
  uint64_t ox = get_offset(x);
  uint64_t ow = get_offset(w);
  uint64_t oo = get_offset(out);
  
  // Compute dimensions
  uint32_t K = x.size(x.dim() - 1);
  uint32_t M = 1;
  for (int i = 0; i < x.dim() - 1; ++i) {
      M *= x.size(i);
  }
  uint32_t N = w.size(0); // weight is [N, K]
  
  // Determine data type and element size
  MPSDataType mpsType;
  NSUInteger elemSize;
  if (x.dtype() == torch::kFloat32) {
    mpsType = MPSDataTypeFloat32;
    elemSize = 4;
  } else if (x.dtype() == torch::kFloat16) {
    mpsType = MPSDataTypeFloat16;
    elemSize = 2;
  } else {
    throw std::runtime_error("linear_mps: unsupported dtype. Use float32 or float16.");
  }
  
  // Create MPS matrix descriptors
  MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:M columns:K
      rowBytes:K * elemSize dataType:mpsType];
  
  MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:N columns:K
      rowBytes:K * elemSize dataType:mpsType];
  
  MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
      matrixDescriptorWithRows:M columns:N
      rowBytes:N * elemSize dataType:mpsType];
  
  // Wrap MTLBuffers as MPS matrices (with tensor storage offsets)
  MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:xb offset:ox descriptor:descA];
  MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:wb offset:ow descriptor:descB];
  MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:ob offset:oo descriptor:descC];
  
  // C = alpha * A * B^T + beta * C  (F.linear computes X @ W^T)
  double beta = accumulate ? 1.0 : 0.0;
  MPSMatrixMultiplication *gemm = [[MPSMatrixMultiplication alloc]
      initWithDevice:device
      transposeLeft:NO
      transposeRight:YES
      resultRows:M
      resultColumns:N
      interiorColumns:K
      alpha:1.0
      beta:beta];
  
  // Encode directly onto PyTorch's command buffer
  [gemm encodeToCommandBuffer:cmdBuf
                   leftMatrix:matA
                  rightMatrix:matB
                 resultMatrix:matC];
}

PYBIND11_MODULE(mps_accel_core, m) {
  m.doc() = "MPS-Accelerate: Direct MPSMatrixMultiplication for PyTorch on Apple Silicon";
  m.def("sdpa_mps", &py_mps_sdpa, "Run MPS-optimized SDPA", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("out"), py::arg("scale"),
        py::arg("lib_path") = "");
  m.def("linear_mps", &py_mps_linear, "Run MPS-optimized Linear (MPSMatrixMultiplication)");
}
