#include "cuvslam/image_processing.hpp"

#include "cuvslam/cuda_image_kernels.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>

namespace cuvslam {

ImageProcessor::ImageProcessor(bool prefer_cuda)
    : use_cuda_(false) {
#ifdef CUVSLAM_WITH_CUDA
  use_cuda_ = prefer_cuda;
#else
  (void)prefer_cuda;
#endif
}

bool ImageProcessor::convertBgrToGray(const cv::Mat& bgr, cv::Mat& gray) {
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    return false;
  }

#ifdef CUVSLAM_WITH_CUDA
  if (use_cuda_ && !backend_decided_) {
    cv::Mat gray_gpu;
    const auto gpu_start = std::chrono::steady_clock::now();
    const bool gpu_ok = cudaConvertBgrToGray(bgr, gray_gpu);
    const auto gpu_end = std::chrono::steady_clock::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    cv::Mat gray_cpu;
    const auto cpu_start = std::chrono::steady_clock::now();
    const bool cpu_ok = convertBgrToGrayCpu(bgr, gray_cpu);
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    if (!cpu_ok) {
      return false;
    }

    backend_decided_ = true;
    if (gpu_ok && gpu_ms < cpu_ms) {
      gray = std::move(gray_gpu);
      use_cuda_ = true;
    } else {
      gray = std::move(gray_cpu);
      use_cuda_ = false;
    }
    return true;
  }

  if (use_cuda_) {
    if (cudaConvertBgrToGray(bgr, gray)) {
      return true;
    }
    // GPU path failed, transparently fallback.
    backend_decided_ = true;
    use_cuda_ = false;
  }
#endif

  backend_decided_ = true;
  return convertBgrToGrayCpu(bgr, gray);
}

bool ImageProcessor::convertBgrToGrayCpu(const cv::Mat& bgr, cv::Mat& gray) const {
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  return true;
}

}  // namespace cuvslam
