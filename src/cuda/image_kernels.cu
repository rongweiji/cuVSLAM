#include "cuvslam/cuda_image_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace cuvslam {
namespace {

__global__ void bgrToGrayKernel(const uint8_t* bgr, uint8_t* gray, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const int base = idx * 3;
  const uint32_t b = bgr[base + 0];
  const uint32_t g = bgr[base + 1];
  const uint32_t r = bgr[base + 2];

  // Integer approximation of ITU-R BT.601 luma conversion with rounding.
  gray[idx] = static_cast<uint8_t>((29u * b + 150u * g + 77u * r + 128u) >> 8);
}

class CudaBuffers {
 public:
  ~CudaBuffers() {
    if (d_bgr_) {
      cudaFree(d_bgr_);
      d_bgr_ = nullptr;
    }
    if (d_gray_) {
      cudaFree(d_gray_);
      d_gray_ = nullptr;
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  bool ensureCapacity(size_t pixel_count) {
    if (pixel_count <= capacity_) {
      return stream_ != nullptr;
    }

    if (d_bgr_) {
      cudaFree(d_bgr_);
      d_bgr_ = nullptr;
    }
    if (d_gray_) {
      cudaFree(d_gray_);
      d_gray_ = nullptr;
    }
    if (!stream_) {
      if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
        stream_ = nullptr;
        return false;
      }
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_bgr_), pixel_count * 3U * sizeof(uint8_t)) != cudaSuccess) {
      d_bgr_ = nullptr;
      return false;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_gray_), pixel_count * sizeof(uint8_t)) != cudaSuccess) {
      cudaFree(d_bgr_);
      d_bgr_ = nullptr;
      d_gray_ = nullptr;
      return false;
    }

    capacity_ = pixel_count;
    return true;
  }

  uint8_t* bgr() { return d_bgr_; }
  uint8_t* gray() { return d_gray_; }
  cudaStream_t stream() const { return stream_; }

 private:
  uint8_t* d_bgr_ = nullptr;
  uint8_t* d_gray_ = nullptr;
  size_t capacity_ = 0;
  cudaStream_t stream_ = nullptr;
};

CudaBuffers& buffers() {
  static CudaBuffers instance;
  return instance;
}

}  // namespace

bool cudaConvertBgrToGray(const cv::Mat& bgr, cv::Mat& gray) {
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    return false;
  }

  cv::Mat input = bgr;
  if (!input.isContinuous()) {
    input = bgr.clone();
  }

  gray.create(input.rows, input.cols, CV_8UC1);
  if (!gray.isContinuous()) {
    gray = gray.clone();
  }

  const int count = input.rows * input.cols;
  CudaBuffers& buf = buffers();
  if (!buf.ensureCapacity(static_cast<size_t>(count))) {
    return false;
  }

  cudaStream_t stream = buf.stream();
  if (stream == nullptr) {
    return false;
  }

  if (cudaMemcpyAsync(buf.bgr(),
                      input.ptr<uint8_t>(),
                      static_cast<size_t>(count) * 3U * sizeof(uint8_t),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = (count + kBlockSize - 1) / kBlockSize;
  bgrToGrayKernel<<<grid_size, kBlockSize, 0, stream>>>(buf.bgr(), buf.gray(), count);

  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  if (cudaMemcpyAsync(gray.ptr<uint8_t>(),
                      buf.gray(),
                      static_cast<size_t>(count) * sizeof(uint8_t),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess) {
    return false;
  }

  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }

  return true;
}

}  // namespace cuvslam

