#pragma once

#include "cuvslam/evaluation.hpp"
#include "cuvslam/types.hpp"

#include <string>
#include <vector>

namespace cuvslam {

struct PipelineOptions {
  std::string dataset_root = "data_sample";
  std::string output_dir = "outputs";
  DatasetFormat dataset_format = DatasetFormat::kAuto;
  std::string reference_tum_path;
  size_t max_frames = 0;
  bool override_intrinsics = false;
  CameraIntrinsics intrinsics_override;
  bool use_cuda = true;
  float depth_scale_m_per_unit = 0.0f;
  float min_depth_m = 0.15f;
  float max_depth_m = 8.0f;
  double evaluation_timestamp_tolerance_s = 0.02;

  bool enable_rerun = false;
  bool rerun_spawn = false;
  bool rerun_log_images = true;
  size_t rerun_log_every_n_frames = 3;
  std::string rerun_recording_id = "cuvslam";
  std::string rerun_save_path;
};

struct PipelineResult {
  bool success = false;
  std::string message;

  DatasetFormat dataset_format = DatasetFormat::kAuto;
  CameraIntrinsics intrinsics;
  std::vector<FrameResult> frames;
  std::vector<TrajectorySample> trajectory;
  RunSummary summary;
  EvaluationMetrics evaluation;

  std::string trajectory_path;
  std::string frame_metrics_path;
  std::string report_path;
};

PipelineResult runPipeline(const PipelineOptions& options);

}  // namespace cuvslam
