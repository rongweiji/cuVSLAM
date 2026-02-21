#pragma once

#include <opencv2/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <string>
#include <vector>

namespace cuvslam {

enum class DatasetFormat {
  kAuto = 0,
  kCustomIphone = 1,
  kTumRgbd = 2,
};

inline const char* datasetFormatToString(DatasetFormat format) {
  switch (format) {
    case DatasetFormat::kCustomIphone:
      return "custom_iphone";
    case DatasetFormat::kTumRgbd:
      return "tum_rgbd";
    case DatasetFormat::kAuto:
    default:
      return "auto";
  }
}

struct CameraIntrinsics {
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;

  bool isValid() const { return fx > 0.0 && fy > 0.0; }
};

struct Pose {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  static Pose Identity() { return {}; }

  Pose inverse() const {
    Pose out;
    out.R = R.transpose();
    out.t = -out.R * t;
    return out;
  }

  Pose operator*(const Pose& rhs) const {
    Pose out;
    out.R = R * rhs.R;
    out.t = R * rhs.t + t;
    return out;
  }

  Eigen::Vector3d transform(const Eigen::Vector3d& p) const { return R * p + t; }

  Eigen::Quaterniond quaternion() const {
    Eigen::Quaterniond q(R);
    q.normalize();
    return q;
  }
};

struct FramePaths {
  int index = -1;
  std::string frame_id;
  double timestamp_s = 0.0;
  std::string rgb_path;
  std::string depth_path;
};

struct FrameData {
  FramePaths meta;
  cv::Mat rgb;
  cv::Mat gray;
  cv::Mat depth_u16;
};

struct TrackingStats {
  bool pose_valid = false;
};

struct StageTimings {
  double load_ms = 0.0;
  double preprocess_ms = 0.0;
  double pose_ms = 0.0;
  double total_ms = 0.0;
};

struct FrameResult {
  int index = -1;
  double timestamp_s = 0.0;
  Pose world_from_cam = Pose::Identity();
  TrackingStats tracking;
  StageTimings timings;
};

struct TrajectorySample {
  double timestamp_s = 0.0;
  Pose world_from_cam = Pose::Identity();
};

struct RunSummary {
  size_t total_frames = 0;
  size_t tracked_frames = 0;
  double total_time_ms = 0.0;
  double average_fps = 0.0;
};

inline bool isFinitePose(const Pose& pose) {
  return pose.R.allFinite() && pose.t.allFinite() && std::abs(pose.R.determinant()) > 1e-9;
}

}  // namespace cuvslam
