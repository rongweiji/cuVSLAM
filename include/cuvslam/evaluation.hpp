#pragma once

#include "cuvslam/types.hpp"

#include <string>
#include <vector>

namespace cuvslam {

struct EvaluationMetrics {
  bool has_reference = false;
  size_t matched_samples = 0;
  double ate_rmse_m = 0.0;
  double ate_mean_m = 0.0;
  double ate_median_m = 0.0;
  double ate_p95_m = 0.0;
  double rpe_rmse_m = 0.0;
};

std::vector<TrajectorySample> readTumTrajectory(const std::string& tum_path, std::string* error = nullptr);

EvaluationMetrics evaluateTrajectory(const std::vector<TrajectorySample>& estimated,
                                     const std::vector<TrajectorySample>& reference,
                                     double timestamp_tolerance_s = 0.005);

bool writeTumTrajectory(const std::string& path,
                        const std::vector<TrajectorySample>& trajectory,
                        std::string* error = nullptr);

}  // namespace cuvslam

