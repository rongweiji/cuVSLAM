#pragma once

#include "cuvslam/types.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace cuvslam {

struct PoseEstimatorParams {
  int max_features = 1400;
  bool use_clahe_preprocessing = true;
  float ratio_test = 0.75f;
  int min_correspondences = 40;
  int min_inliers = 28;
  int pnp_iterations = 120;
  float pnp_reprojection_error_px = 3.0f;
  double max_translation_per_frame_m = 0.6;
  double max_rotation_per_frame_deg = 35.0;
  bool enable_svd_fallback = true;
  int ransac_iterations = 160;
  double ransac_inlier_threshold_m = 0.05;
  float min_depth_m = 0.15f;
  float max_depth_m = 8.0f;
};

class DepthPoseEstimator {
 public:
  explicit DepthPoseEstimator(PoseEstimatorParams params = {});

  TrackingStats estimate(const FrameData& prev,
                         const FrameData& curr,
                         const CameraIntrinsics& intrinsics,
                         StageTimings* timings = nullptr);

 private:
  bool backproject(const cv::Point2f& pixel,
                   const cv::Mat& depth_m,
                   const CameraIntrinsics& intrinsics,
                   Eigen::Vector3d& out) const;

  Pose estimateRigidTransformSvd(const std::vector<Eigen::Vector3d>& src,
                                 const std::vector<Eigen::Vector3d>& dst) const;

  int countInliers(const Pose& transform_curr_from_prev,
                   const std::vector<Eigen::Vector3d>& src,
                   const std::vector<Eigen::Vector3d>& dst,
                   std::vector<int>* inlier_indices = nullptr) const;

  bool isMotionReasonable(const Pose& transform_curr_from_prev) const;

  PoseEstimatorParams params_;
  cv::Ptr<cv::ORB> orb_;
  cv::Ptr<cv::CLAHE> clahe_;
  cv::BFMatcher matcher_;
};

}  // namespace cuvslam
