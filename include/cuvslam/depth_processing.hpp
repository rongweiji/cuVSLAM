#pragma once

#include <opencv2/core.hpp>

namespace cuvslam {

class DepthProcessor {
 public:
  explicit DepthProcessor(bool prefer_cuda = true);

  void setDepthScale(float meters_per_unit) { depth_scale_m_per_unit_ = meters_per_unit; }
  void setDepthLimits(float min_depth_m, float max_depth_m) {
    min_depth_m_ = min_depth_m;
    max_depth_m_ = max_depth_m;
  }

  bool usingCuda() const { return use_cuda_; }

  bool convertToMeters(const cv::Mat& depth_u16, cv::Mat& depth_m);

 private:
  bool convertToMetersCpu(const cv::Mat& depth_u16, cv::Mat& depth_m) const;

  bool use_cuda_ = false;
  float depth_scale_m_per_unit_ = 0.001f;
  float min_depth_m_ = 0.1f;
  float max_depth_m_ = 8.0f;
};

}  // namespace cuvslam

