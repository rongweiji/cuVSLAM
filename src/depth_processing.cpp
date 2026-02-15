#include "cuvslam/depth_processing.hpp"

#include "cuvslam/cuda_depth_kernels.hpp"

#include <opencv2/core/utility.hpp>

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
  if (use_cuda_) {
    if (cudaConvertDepthU16ToMeters(depth_u16,
                                    depth_m,
                                    depth_scale_m_per_unit_,
                                    min_depth_m_,
                                    max_depth_m_)) {
      return true;
    }
    // GPU path failed, transparently fallback.
    use_cuda_ = false;
  }
#endif

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

