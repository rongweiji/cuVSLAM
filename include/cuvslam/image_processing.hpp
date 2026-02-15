#pragma once

#include <opencv2/core.hpp>

namespace cuvslam {

class ImageProcessor {
 public:
  explicit ImageProcessor(bool prefer_cuda = true);

  bool usingCuda() const { return use_cuda_; }
  bool convertBgrToGray(const cv::Mat& bgr, cv::Mat& gray);

 private:
  bool convertBgrToGrayCpu(const cv::Mat& bgr, cv::Mat& gray) const;

  bool use_cuda_ = false;
  bool backend_decided_ = false;
};

}  // namespace cuvslam
