#include "cuvslam/pose_estimator.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

namespace cuvslam {
namespace {

template <typename Clock = std::chrono::steady_clock>
double elapsedMs(const typename Clock::time_point& start,
                 const typename Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

DepthPoseEstimator::DepthPoseEstimator(PoseEstimatorParams params)
    : params_(params),
      orb_(cv::ORB::create(params_.max_features)),
      clahe_(cv::createCLAHE(2.0, cv::Size(8, 8))),
      matcher_(cv::NORM_HAMMING, false) {}

TrackingStats DepthPoseEstimator::estimate(const FrameData& prev,
                                           const FrameData& curr,
                                           const CameraIntrinsics& intrinsics,
                                           StageTimings* timings) {
  TrackingStats stats;

  std::vector<cv::KeyPoint> kp_prev;
  std::vector<cv::KeyPoint> kp_curr;
  cv::Mat desc_prev;
  cv::Mat desc_curr;

  cv::Mat prev_for_features = prev.gray;
  cv::Mat curr_for_features = curr.gray;
  if (params_.use_clahe_preprocessing) {
    clahe_->apply(prev.gray, prev_for_features);
    clahe_->apply(curr.gray, curr_for_features);
  }

  const auto detect_start = std::chrono::steady_clock::now();
  orb_->detectAndCompute(prev_for_features, cv::noArray(), kp_prev, desc_prev);
  orb_->detectAndCompute(curr_for_features, cv::noArray(), kp_curr, desc_curr);
  const auto detect_end = std::chrono::steady_clock::now();

  if (timings) {
    timings->detect_describe_ms = elapsedMs(detect_start, detect_end);
  }

  stats.keypoints_prev = static_cast<int>(kp_prev.size());
  stats.keypoints_curr = static_cast<int>(kp_curr.size());

  if (desc_prev.empty() || desc_curr.empty()) {
    return stats;
  }

  const auto match_start = std::chrono::steady_clock::now();

  std::vector<std::vector<cv::DMatch>> knn_forward;
  std::vector<std::vector<cv::DMatch>> knn_reverse;
  matcher_.knnMatch(desc_prev, desc_curr, knn_forward, 2);
  matcher_.knnMatch(desc_curr, desc_prev, knn_reverse, 2);

  std::vector<int> best_forward(static_cast<size_t>(desc_prev.rows), -1);
  std::vector<float> best_forward_dist(static_cast<size_t>(desc_prev.rows), 0.0f);
  for (size_t i = 0; i < knn_forward.size(); ++i) {
    const auto& candidates = knn_forward[i];
    if (candidates.size() < 2) {
      continue;
    }
    if (candidates[0].distance < params_.ratio_test * candidates[1].distance) {
      best_forward[i] = candidates[0].trainIdx;
      best_forward_dist[i] = candidates[0].distance;
    }
  }

  std::vector<int> best_reverse(static_cast<size_t>(desc_curr.rows), -1);
  for (size_t i = 0; i < knn_reverse.size(); ++i) {
    const auto& candidates = knn_reverse[i];
    if (candidates.size() < 2) {
      continue;
    }
    if (candidates[0].distance < params_.ratio_test * candidates[1].distance) {
      best_reverse[i] = candidates[0].trainIdx;
    }
  }

  std::vector<cv::DMatch> good_matches;
  good_matches.reserve(knn_forward.size());
  for (size_t i = 0; i < best_forward.size(); ++i) {
    const int train_idx = best_forward[i];
    if (train_idx < 0) {
      continue;
    }
    if (train_idx >= static_cast<int>(best_reverse.size())) {
      continue;
    }
    if (best_reverse[static_cast<size_t>(train_idx)] == static_cast<int>(i)) {
      good_matches.emplace_back(static_cast<int>(i), train_idx, best_forward_dist[i]);
    }
  }

  const auto match_end = std::chrono::steady_clock::now();
  if (timings) {
    timings->match_ms = elapsedMs(match_start, match_end);
  }

  stats.tentative_matches = static_cast<int>(good_matches.size());
  if (good_matches.empty()) {
    return stats;
  }

  std::vector<cv::Point3f> object_points_prev;
  std::vector<cv::Point2f> image_points_curr;
  object_points_prev.reserve(good_matches.size());
  image_points_curr.reserve(good_matches.size());

  std::vector<Eigen::Vector3d> src_prev;
  std::vector<Eigen::Vector3d> dst_curr;
  src_prev.reserve(good_matches.size());
  dst_curr.reserve(good_matches.size());

  for (const auto& match : good_matches) {
    Eigen::Vector3d p_prev;
    if (!backproject(kp_prev[match.queryIdx].pt, prev.depth_m, intrinsics, p_prev)) {
      continue;
    }

    object_points_prev.emplace_back(static_cast<float>(p_prev.x()),
                                    static_cast<float>(p_prev.y()),
                                    static_cast<float>(p_prev.z()));
    image_points_curr.push_back(kp_curr[match.trainIdx].pt);

    Eigen::Vector3d p_curr;
    if (backproject(kp_curr[match.trainIdx].pt, curr.depth_m, intrinsics, p_curr)) {
      src_prev.push_back(p_prev);
      dst_curr.push_back(p_curr);
    }
  }

  if (object_points_prev.size() < static_cast<size_t>(params_.min_correspondences)) {
    return stats;
  }

  const auto pose_start = std::chrono::steady_clock::now();

  bool pnp_success = false;
  Pose pose_from_pnp = Pose::Identity();
  int pnp_inliers = 0;

  {
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << intrinsics.fx,
                             0.0,
                             intrinsics.cx,
                             0.0,
                             intrinsics.fy,
                             intrinsics.cy,
                             0.0,
                             0.0,
                             1.0);

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;

    pnp_success = cv::solvePnPRansac(object_points_prev,
                                     image_points_curr,
                                     camera_matrix,
                                     cv::noArray(),
                                     rvec,
                                     tvec,
                                     false,
                                     params_.pnp_iterations,
                                     params_.pnp_reprojection_error_px,
                                     0.99,
                                     inliers,
                                     cv::SOLVEPNP_EPNP);

    if (pnp_success && inliers.rows >= params_.min_inliers) {
      std::vector<cv::Point3f> inlier_object_points;
      std::vector<cv::Point2f> inlier_image_points;
      inlier_object_points.reserve(static_cast<size_t>(inliers.rows));
      inlier_image_points.reserve(static_cast<size_t>(inliers.rows));

      for (int r = 0; r < inliers.rows; ++r) {
        const int idx = inliers.at<int>(r, 0);
        inlier_object_points.push_back(object_points_prev[static_cast<size_t>(idx)]);
        inlier_image_points.push_back(image_points_curr[static_cast<size_t>(idx)]);
      }

      if (inlier_object_points.size() >= 6U) {
        cv::solvePnPRefineLM(inlier_object_points,
                             inlier_image_points,
                             camera_matrix,
                             cv::noArray(),
                             rvec,
                             tvec);
      }

      cv::Mat R_cv;
      cv::Rodrigues(rvec, R_cv);

      Pose candidate;
      candidate.R << R_cv.at<double>(0, 0), R_cv.at<double>(0, 1), R_cv.at<double>(0, 2),
          R_cv.at<double>(1, 0), R_cv.at<double>(1, 1), R_cv.at<double>(1, 2),
          R_cv.at<double>(2, 0), R_cv.at<double>(2, 1), R_cv.at<double>(2, 2);
      candidate.t = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

      if (isFinitePose(candidate) && isMotionReasonable(candidate)) {
        pose_from_pnp = candidate;
        pnp_inliers = inliers.rows;
      } else {
        pnp_success = false;
      }
    } else {
      pnp_success = false;
    }
  }

  if (pnp_success) {
    stats.pose_valid = true;
    stats.relative_curr_from_prev = pose_from_pnp;
    stats.inlier_matches = pnp_inliers;
  } else if (params_.enable_svd_fallback &&
             src_prev.size() >= static_cast<size_t>(params_.min_correspondences) &&
             src_prev.size() >= 3U) {
    Pose best_pose = Pose::Identity();
    int best_inliers = 0;
    std::vector<int> best_inlier_indices;

    std::mt19937 rng(static_cast<uint32_t>(prev.meta.index * 73856093U + curr.meta.index * 19349663U + 0x1234U));
    std::uniform_int_distribution<int> dist(0, static_cast<int>(src_prev.size() - 1));

    for (int iter = 0; iter < params_.ransac_iterations; ++iter) {
      int i0 = dist(rng);
      int i1 = dist(rng);
      int i2 = dist(rng);
      if (i0 == i1 || i0 == i2 || i1 == i2) {
        continue;
      }

      std::vector<Eigen::Vector3d> sample_src = {src_prev[static_cast<size_t>(i0)],
                                                  src_prev[static_cast<size_t>(i1)],
                                                  src_prev[static_cast<size_t>(i2)]};
      std::vector<Eigen::Vector3d> sample_dst = {dst_curr[static_cast<size_t>(i0)],
                                                  dst_curr[static_cast<size_t>(i1)],
                                                  dst_curr[static_cast<size_t>(i2)]};

      const Pose candidate = estimateRigidTransformSvd(sample_src, sample_dst);
      if (!isFinitePose(candidate) || !isMotionReasonable(candidate)) {
        continue;
      }

      std::vector<int> inlier_indices;
      const int inlier_count = countInliers(candidate, src_prev, dst_curr, &inlier_indices);
      if (inlier_count > best_inliers) {
        best_inliers = inlier_count;
        best_pose = candidate;
        best_inlier_indices = std::move(inlier_indices);
      }
    }

    if (best_inliers >= params_.min_inliers && !best_inlier_indices.empty()) {
      std::vector<Eigen::Vector3d> inlier_src;
      std::vector<Eigen::Vector3d> inlier_dst;
      inlier_src.reserve(best_inlier_indices.size());
      inlier_dst.reserve(best_inlier_indices.size());

      for (int idx : best_inlier_indices) {
        inlier_src.push_back(src_prev[static_cast<size_t>(idx)]);
        inlier_dst.push_back(dst_curr[static_cast<size_t>(idx)]);
      }

      best_pose = estimateRigidTransformSvd(inlier_src, inlier_dst);
      if (isFinitePose(best_pose) && isMotionReasonable(best_pose)) {
        stats.pose_valid = true;
        stats.relative_curr_from_prev = best_pose;
        stats.inlier_matches = best_inliers;
      }
    }
  }

  const auto pose_end = std::chrono::steady_clock::now();
  if (timings) {
    timings->pose_ms = elapsedMs(pose_start, pose_end);
  }

  return stats;
}

bool DepthPoseEstimator::backproject(const cv::Point2f& pixel,
                                     const cv::Mat& depth_m,
                                     const CameraIntrinsics& intrinsics,
                                     Eigen::Vector3d& out) const {
  const int u = static_cast<int>(std::lround(pixel.x));
  const int v = static_cast<int>(std::lround(pixel.y));

  if (u < 0 || u >= depth_m.cols || v < 0 || v >= depth_m.rows) {
    return false;
  }

  const float z = depth_m.at<float>(v, u);
  if (!std::isfinite(z) || z < params_.min_depth_m || z > params_.max_depth_m) {
    return false;
  }

  const double x = (static_cast<double>(u) - intrinsics.cx) * static_cast<double>(z) / intrinsics.fx;
  const double y = (static_cast<double>(v) - intrinsics.cy) * static_cast<double>(z) / intrinsics.fy;
  out = Eigen::Vector3d(x, y, z);
  return true;
}

Pose DepthPoseEstimator::estimateRigidTransformSvd(const std::vector<Eigen::Vector3d>& src,
                                                   const std::vector<Eigen::Vector3d>& dst) const {
  Pose out = Pose::Identity();
  if (src.size() != dst.size() || src.size() < 3U) {
    return out;
  }

  Eigen::Vector3d src_centroid = Eigen::Vector3d::Zero();
  Eigen::Vector3d dst_centroid = Eigen::Vector3d::Zero();

  for (size_t i = 0; i < src.size(); ++i) {
    src_centroid += src[i];
    dst_centroid += dst[i];
  }
  src_centroid /= static_cast<double>(src.size());
  dst_centroid /= static_cast<double>(dst.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    covariance += (src[i] - src_centroid) * (dst[i] - dst_centroid).transpose();
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d rotation = svd.matrixV() * svd.matrixU().transpose();

  if (rotation.determinant() < 0.0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) *= -1.0;
    rotation = V * svd.matrixU().transpose();
  }

  out.R = rotation;
  out.t = dst_centroid - out.R * src_centroid;
  return out;
}

int DepthPoseEstimator::countInliers(const Pose& transform_curr_from_prev,
                                     const std::vector<Eigen::Vector3d>& src,
                                     const std::vector<Eigen::Vector3d>& dst,
                                     std::vector<int>* inlier_indices) const {
  const double threshold2 = params_.ransac_inlier_threshold_m * params_.ransac_inlier_threshold_m;
  int inliers = 0;

  if (inlier_indices) {
    inlier_indices->clear();
  }

  for (size_t i = 0; i < src.size(); ++i) {
    const Eigen::Vector3d projected = transform_curr_from_prev.transform(src[i]);
    const double error2 = (projected - dst[i]).squaredNorm();
    if (error2 <= threshold2) {
      ++inliers;
      if (inlier_indices) {
        inlier_indices->push_back(static_cast<int>(i));
      }
    }
  }

  return inliers;
}

bool DepthPoseEstimator::isMotionReasonable(const Pose& transform_curr_from_prev) const {
  if (!isFinitePose(transform_curr_from_prev)) {
    return false;
  }

  const double translation_norm = transform_curr_from_prev.t.norm();
  if (translation_norm > params_.max_translation_per_frame_m) {
    return false;
  }

  Eigen::AngleAxisd angle_axis(transform_curr_from_prev.R);
  const double angle_deg = std::abs(angle_axis.angle()) * 180.0 / M_PI;
  return angle_deg <= params_.max_rotation_per_frame_deg;
}

}  // namespace cuvslam

