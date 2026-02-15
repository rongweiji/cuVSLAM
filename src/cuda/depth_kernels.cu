#include "cuvslam/cuda_depth_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace cuvslam {
namespace {

__global__ void convertDepthKernel(const uint16_t* depth_u16,
                                   float* depth_m,
                                   int count,
                                   float scale,
                                   float min_depth,
                                   float max_depth) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const uint16_t raw = depth_u16[idx];
  float z = static_cast<float>(raw) * scale;
  if (raw == 0 || z < min_depth || z > max_depth) {
    z = 0.0f;
  }
  depth_m[idx] = z;
}

class CudaBuffers {
 public:
  ~CudaBuffers() {
    if (d_in_) {
      cudaFree(d_in_);
      d_in_ = nullptr;
    }
    if (d_out_) {
      cudaFree(d_out_);
      d_out_ = nullptr;
    }
  }

  bool ensureCapacity(size_t count) {
    if (count <= capacity_) {
      return true;
    }

    if (d_in_) {
      cudaFree(d_in_);
      d_in_ = nullptr;
    }
    if (d_out_) {
      cudaFree(d_out_);
      d_out_ = nullptr;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_in_), count * sizeof(uint16_t)) != cudaSuccess) {
      d_in_ = nullptr;
      return false;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_out_), count * sizeof(float)) != cudaSuccess) {
      cudaFree(d_in_);
      d_in_ = nullptr;
      d_out_ = nullptr;
      return false;
    }

    capacity_ = count;
    return true;
  }

  uint16_t* input() { return d_in_; }
  float* output() { return d_out_; }

 private:
  uint16_t* d_in_ = nullptr;
  float* d_out_ = nullptr;
  size_t capacity_ = 0;
};

CudaBuffers& buffers() {
  static CudaBuffers instance;
  return instance;
}

}  // namespace

bool cudaConvertDepthU16ToMeters(const cv::Mat& depth_u16,
                                 cv::Mat& depth_m,
                                 float depth_scale_m_per_unit,
                                 float min_depth_m,
                                 float max_depth_m) {
  if (depth_u16.empty() || depth_u16.type() != CV_16UC1) {
    return false;
  }

  cv::Mat input = depth_u16;
  if (!input.isContinuous()) {
    input = depth_u16.clone();
  }

  depth_m.create(input.rows, input.cols, CV_32FC1);
  if (!depth_m.isContinuous()) {
    depth_m = depth_m.clone();
  }

  const int count = input.rows * input.cols;
  CudaBuffers& buf = buffers();
  if (!buf.ensureCapacity(static_cast<size_t>(count))) {
    return false;
  }

  if (cudaMemcpy(buf.input(), input.ptr<uint16_t>(), count * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = (count + kBlockSize - 1) / kBlockSize;
  convertDepthKernel<<<grid_size, kBlockSize>>>(
      buf.input(), buf.output(), count, depth_scale_m_per_unit, min_depth_m, max_depth_m);

  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  if (cudaMemcpy(depth_m.ptr<float>(), buf.output(), count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return false;
  }

  return true;
}

}  // namespace cuvslam

