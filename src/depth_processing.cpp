#include "cuvslam/depth_processing.hpp"

#include "cuvslam/cuda_depth_kernels.hpp"

#include <opencv2/core/utility.hpp>

#include <chrono>

namespace cuvslam {

DepthProcessor::DepthProcessor(bool prefer_cuda)
    : use_cuda_(false) {
#ifdef CUVSLAM_WITH_CUDA
  use_cuda_ = prefer_cuda;
#else
  (void)prefer_cuda;
#endif
}

bool DepthProcessor::convertToMeters(const cv::Mat& depth_u16, cv::Mat& depth_m) {
  if (depth_u16.empty() || depth_u16.type() != CV_16UC1) {
    return false;
  }

#ifdef CUVSLAM_WITH_CUDA
  if (use_cuda_ && !backend_decided_) {
    cv::Mat depth_gpu;
    const auto gpu_start = std::chrono::steady_clock::now();
    const bool gpu_ok = cudaConvertDepthU16ToMeters(depth_u16,
                                                    depth_gpu,
                                                    depth_scale_m_per_unit_,
                                                    min_depth_m_,
                                                    max_depth_m_);
    const auto gpu_end = std::chrono::steady_clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    cv::Mat depth_cpu;
    const auto cpu_start = std::chrono::steady_clock::now();
    const bool cpu_ok = convertToMetersCpu(depth_u16, depth_cpu);
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    if (!cpu_ok) {
      return false;
    }

    backend_decided_ = true;
    if (gpu_ok && gpu_ms < cpu_ms) {
      depth_m = std::move(depth_gpu);
      use_cuda_ = true;
    } else {
      depth_m = std::move(depth_cpu);
      use_cuda_ = false;
    }
    return true;
  }

  if (use_cuda_) {
    if (cudaConvertDepthU16ToMeters(depth_u16,
                                    depth_m,
                                    depth_scale_m_per_unit_,
                                    min_depth_m_,
                                    max_depth_m_)) {
      return true;
    }
    // GPU path failed, transparently fallback.
    backend_decided_ = true;
    use_cuda_ = false;
  }
#endif

  backend_decided_ = true;
  return convertToMetersCpu(depth_u16, depth_m);
}

bool DepthProcessor::convertToMetersCpu(const cv::Mat& depth_u16, cv::Mat& depth_m) const {
  depth_m.create(depth_u16.rows, depth_u16.cols, CV_32FC1);

  const int rows = depth_u16.rows;
  const int cols = depth_u16.cols;
  const float scale = depth_scale_m_per_unit_;
  const float min_depth = min_depth_m_;
  const float max_depth = max_depth_m_;

  cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
    for (int r = range.start; r < range.end; ++r) {
      const uint16_t* src = depth_u16.ptr<uint16_t>(r);
      float* dst = depth_m.ptr<float>(r);
      for (int c = 0; c < cols; ++c) {
        const uint16_t raw = src[c];
        float meters = static_cast<float>(raw) * scale;
        if (raw == 0 || meters < min_depth || meters > max_depth) {
          meters = 0.0f;
        }
        dst[c] = meters;
      }
    }
  });

  return true;
}

}  // namespace cuvslam
