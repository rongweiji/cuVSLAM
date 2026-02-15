#include "cuvslam/evaluation.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <tuple>

namespace cuvslam {
namespace {

struct MatchedSample {
  TrajectorySample est;
  TrajectorySample ref;
};

Pose rigidAlignmentFromTranslations(const std::vector<Eigen::Vector3d>& est,
                                    const std::vector<Eigen::Vector3d>& ref) {
  Pose alignment = Pose::Identity();
  if (est.size() != ref.size() || est.size() < 3U) {
    return alignment;
  }

  Eigen::Vector3d est_centroid = Eigen::Vector3d::Zero();
  Eigen::Vector3d ref_centroid = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < est.size(); ++i) {
    est_centroid += est[i];
    ref_centroid += ref[i];
  }
  est_centroid /= static_cast<double>(est.size());
  ref_centroid /= static_cast<double>(ref.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < est.size(); ++i) {
    covariance += (est[i] - est_centroid) * (ref[i] - ref_centroid).transpose();
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d rotation = svd.matrixV() * svd.matrixU().transpose();
  if (rotation.determinant() < 0.0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) *= -1.0;
    rotation = V * svd.matrixU().transpose();
  }

  alignment.R = rotation;
  alignment.t = ref_centroid - alignment.R * est_centroid;
  return alignment;
}

TrajectorySample applyAlignment(const TrajectorySample& in, const Pose& alignment) {
  TrajectorySample out;
  out.timestamp_s = in.timestamp_s;
  out.world_from_cam.R = alignment.R * in.world_from_cam.R;
  out.world_from_cam.t = alignment.R * in.world_from_cam.t + alignment.t;
  return out;
}

double computeRmse(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum_sq = 0.0;
  for (double v : values) {
    sum_sq += v * v;
  }
  return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

double computeMean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (double v : values) {
    sum += v;
  }
  return sum / static_cast<double>(values.size());
}

double computePercentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double clamped = std::min(1.0, std::max(0.0, percentile));
  const double idx = clamped * static_cast<double>(values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) {
    return values[lo];
  }
  const double alpha = idx - static_cast<double>(lo);
  return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

std::vector<MatchedSample> matchByTimestamp(const std::vector<TrajectorySample>& est,
                                            const std::vector<TrajectorySample>& ref,
                                            double tolerance_s) {
  std::vector<MatchedSample> matches;
  matches.reserve(std::min(est.size(), ref.size()));

  size_t i = 0;
  size_t j = 0;

  while (i < est.size() && j < ref.size()) {
    const double dt = est[i].timestamp_s - ref[j].timestamp_s;
    if (std::abs(dt) <= tolerance_s) {
      matches.push_back({est[i], ref[j]});
      ++i;
      ++j;
    } else if (dt < 0.0) {
      ++i;
    } else {
      ++j;
    }
  }

  return matches;
}

}  // namespace

std::vector<TrajectorySample> readTumTrajectory(const std::string& tum_path, std::string* error) {
  std::ifstream in(tum_path);
  if (!in.is_open()) {
    if (error) {
      *error = "Failed to open TUM trajectory: " + tum_path;
    }
    return {};
  }

  std::vector<TrajectorySample> trajectory;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream ss(line);
    TrajectorySample sample;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 1.0;

    if (!(ss >> sample.timestamp_s >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
      continue;
    }

    sample.world_from_cam.t = Eigen::Vector3d(tx, ty, tz);
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    sample.world_from_cam.R = q.toRotationMatrix();
    trajectory.push_back(sample);
  }

  return trajectory;
}

EvaluationMetrics evaluateTrajectory(const std::vector<TrajectorySample>& estimated,
                                     const std::vector<TrajectorySample>& reference,
                                     double timestamp_tolerance_s) {
  EvaluationMetrics metrics;
  metrics.has_reference = !reference.empty();

  if (estimated.empty() || reference.empty()) {
    return metrics;
  }

  const std::vector<MatchedSample> matches = matchByTimestamp(estimated, reference, timestamp_tolerance_s);
  metrics.matched_samples = matches.size();
  if (matches.size() < 3U) {
    return metrics;
  }

  std::vector<Eigen::Vector3d> est_points;
  std::vector<Eigen::Vector3d> ref_points;
  est_points.reserve(matches.size());
  ref_points.reserve(matches.size());

  for (const auto& match : matches) {
    est_points.push_back(match.est.world_from_cam.t);
    ref_points.push_back(match.ref.world_from_cam.t);
  }

  const Pose alignment = rigidAlignmentFromTranslations(est_points, ref_points);

  std::vector<double> ate_errors;
  ate_errors.reserve(matches.size());

  std::vector<TrajectorySample> aligned_est;
  aligned_est.reserve(matches.size());
  std::vector<TrajectorySample> ref_matched;
  ref_matched.reserve(matches.size());

  for (const auto& match : matches) {
    const TrajectorySample aligned = applyAlignment(match.est, alignment);
    aligned_est.push_back(aligned);
    ref_matched.push_back(match.ref);

    const double error = (aligned.world_from_cam.t - match.ref.world_from_cam.t).norm();
    ate_errors.push_back(error);
  }

  metrics.ate_rmse_m = computeRmse(ate_errors);
  metrics.ate_mean_m = computeMean(ate_errors);
  metrics.ate_median_m = computePercentile(ate_errors, 0.5);
  metrics.ate_p95_m = computePercentile(ate_errors, 0.95);

  std::vector<double> rpe_errors;
  if (aligned_est.size() >= 2U) {
    rpe_errors.reserve(aligned_est.size() - 1U);
    for (size_t i = 1; i < aligned_est.size(); ++i) {
      const Pose delta_est = aligned_est[i - 1].world_from_cam.inverse() * aligned_est[i].world_from_cam;
      const Pose delta_ref = ref_matched[i - 1].world_from_cam.inverse() * ref_matched[i].world_from_cam;
      rpe_errors.push_back((delta_est.t - delta_ref.t).norm());
    }
  }

  metrics.rpe_rmse_m = computeRmse(rpe_errors);
  return metrics;
}

bool writeTumTrajectory(const std::string& path,
                        const std::vector<TrajectorySample>& trajectory,
                        std::string* error) {
  std::ofstream out(path);
  if (!out.is_open()) {
    if (error) {
      *error = "Failed to write trajectory file: " + path;
    }
    return false;
  }

  out << std::fixed << std::setprecision(9);
  for (const auto& sample : trajectory) {
    const Eigen::Quaterniond q = sample.world_from_cam.quaternion();
    out << sample.timestamp_s << " "
        << sample.world_from_cam.t.x() << " "
        << sample.world_from_cam.t.y() << " "
        << sample.world_from_cam.t.z() << " "
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
  }

  return true;
}

}  // namespace cuvslam
