#pragma once

#include <opencv2/core.hpp>

namespace cuvslam {

bool cudaConvertDepthU16ToMeters(const cv::Mat& depth_u16,
                                 cv::Mat& depth_m,
                                 float depth_scale_m_per_unit,
                                 float min_depth_m,
                                 float max_depth_m);

}  // namespace cuvslam

