#pragma once

#include <opencv2/core.hpp>

namespace cuvslam {

bool cudaConvertBgrToGray(const cv::Mat& bgr, cv::Mat& gray);

}  // namespace cuvslam

