#include "cuvslam/pipeline.hpp"

#include "cuvslam/dataset_loader.hpp"
#include "cuvslam/depth_processing.hpp"
#include "cuvslam/image_processing.hpp"
#include "cuvslam/pose_estimator.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef CUVSLAM_WITH_RERUN
#include <rerun.hpp>
#endif

namespace cuvslam {
namespace fs = std::filesystem;

namespace {

template <typename Clock = std::chrono::steady_clock>
double elapsedMs(const typename Clock::time_point& start,
                 const typename Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool writeFrameMetricsCsv(const std::string& path,
                          const std::vector<FrameResult>& frames,
                          std::string* error) {
  std::ofstream out(path);
  if (!out.is_open()) {
    if (error) {
      *error = "Failed to write frame metrics CSV: " + path;
    }
    return false;
  }

  out << "index,timestamp_s,tx,ty,tz,tracked,inliers,keypoints_prev,keypoints_curr,tentative_matches,"
         "load_ms,preprocess_ms,detect_describe_ms,match_ms,pose_ms,total_ms\n";

  out << std::fixed << std::setprecision(6);
  for (const auto& frame : frames) {
    out << frame.index << ","
        << frame.timestamp_s << ","
        << frame.world_from_cam.t.x() << ","
        << frame.world_from_cam.t.y() << ","
        << frame.world_from_cam.t.z() << ","
        << (frame.tracking.pose_valid ? 1 : 0) << ","
        << frame.tracking.inlier_matches << ","
        << frame.tracking.keypoints_prev << ","
        << frame.tracking.keypoints_curr << ","
        << frame.tracking.tentative_matches << ","
        << frame.timings.load_ms << ","
        << frame.timings.preprocess_ms << ","
        << frame.timings.detect_describe_ms << ","
        << frame.timings.match_ms << ","
        << frame.timings.pose_ms << ","
        << frame.timings.total_ms << "\n";
  }

  return true;
}

bool writePerformanceReport(const std::string& path,
                            const PipelineResult& result,
                            float depth_scale_m_per_unit,
                            std::string* error) {
  std::ofstream out(path);
  if (!out.is_open()) {
    if (error) {
      *error = "Failed to write performance report: " + path;
    }
    return false;
  }

  double load_total = 0.0;
  double preprocess_total = 0.0;
  double detect_total = 0.0;
  double match_total = 0.0;
  double pose_total = 0.0;
  double frame_total = 0.0;

  for (const auto& frame : result.frames) {
    load_total += frame.timings.load_ms;
    preprocess_total += frame.timings.preprocess_ms;
    detect_total += frame.timings.detect_describe_ms;
    match_total += frame.timings.match_ms;
    pose_total += frame.timings.pose_ms;
    frame_total += frame.timings.total_ms;
  }

  const double n = result.frames.empty() ? 1.0 : static_cast<double>(result.frames.size());

  out << "# cuVSLAM Performance Report\n\n";
  out << "## Dataset\n\n";
  out << "- Dataset format: " << datasetFormatToString(result.dataset_format) << "\n";
  out << "- Depth scale (m/unit): " << std::fixed << std::setprecision(6) << depth_scale_m_per_unit << "\n";
  out << "- Intrinsics (fx, fy, cx, cy): " << std::fixed << std::setprecision(3)
      << result.intrinsics.fx << ", " << result.intrinsics.fy << ", "
      << result.intrinsics.cx << ", " << result.intrinsics.cy << "\n\n";

  out << "## Summary\n\n";
  out << "- Total frames: " << result.summary.total_frames << "\n";
  out << "- Successfully tracked frames: " << result.summary.tracked_frames << "\n";
  out << "- Total runtime (ms): " << std::fixed << std::setprecision(3) << result.summary.total_time_ms << "\n";
  out << "- Average FPS: " << std::fixed << std::setprecision(3) << result.summary.average_fps << "\n";
  out << "- Average inlier matches: " << std::fixed << std::setprecision(3) << result.summary.avg_inliers << "\n";
  out << "- ATE RMSE (m): " << std::fixed << std::setprecision(4) << result.evaluation.ate_rmse_m << "\n";
  out << "- RPE RMSE (m): " << std::fixed << std::setprecision(4) << result.evaluation.rpe_rmse_m << "\n\n";

  out << "## Stage Timing (per frame average)\n\n";
  out << "| Stage | Mean ms |\n";
  out << "|---|---:|\n";
  out << "| Load images | " << std::fixed << std::setprecision(3) << (load_total / n) << " |\n";
  out << "| Depth preprocessing | " << std::fixed << std::setprecision(3) << (preprocess_total / n) << " |\n";
  out << "| Feature detect+describe | " << std::fixed << std::setprecision(3) << (detect_total / n) << " |\n";
  out << "| Feature match | " << std::fixed << std::setprecision(3) << (match_total / n) << " |\n";
  out << "| Relative pose estimate | " << std::fixed << std::setprecision(3) << (pose_total / n) << " |\n";
  out << "| Full frame | " << std::fixed << std::setprecision(3) << (frame_total / n) << " |\n\n";

  if (result.evaluation.has_reference) {
    out << "## Accuracy vs Reference\n\n";
    out << "- Matched trajectory samples: " << result.evaluation.matched_samples << "\n";
    out << "- ATE mean/median/p95 (m): "
        << std::fixed << std::setprecision(4)
        << result.evaluation.ate_mean_m << " / "
        << result.evaluation.ate_median_m << " / "
        << result.evaluation.ate_p95_m << "\n";
    out << "- ATE RMSE (m): " << std::fixed << std::setprecision(4) << result.evaluation.ate_rmse_m << "\n";
    out << "- RPE RMSE (m): " << std::fixed << std::setprecision(4) << result.evaluation.rpe_rmse_m << "\n";
  }

  return true;
}

#ifdef CUVSLAM_WITH_RERUN
class RerunLogger {
 public:
  RerunLogger(const PipelineOptions& options,
              const CameraIntrinsics& intrinsics,
              int width,
              int height,
              float depth_scale_m_per_unit,
              std::string* warning)
      : options_(options),
        rec_(options.rerun_recording_id.empty() ? "cuvslam" : options.rerun_recording_id),
        depth_scale_m_per_unit_(depth_scale_m_per_unit) {
    if (!options_.rerun_save_path.empty()) {
      const auto err = rec_.save(options_.rerun_save_path);
      if (err.is_err()) {
        enabled_ = false;
        if (warning) {
          *warning = "Failed to open Rerun save path: " + options_.rerun_save_path;
        }
        return;
      }
    }

    if (options_.rerun_spawn) {
      const auto err = rec_.spawn();
      if (err.is_err() && warning) {
        *warning = "Failed to spawn Rerun viewer executable from PATH. Continuing without live viewer.";
      }
    }

    rec_.log_static("world/camera",
                    rerun::Pinhole::from_focal_length_and_resolution(
                        {static_cast<float>(intrinsics.fx), static_cast<float>(intrinsics.fy)},
                        {static_cast<float>(width), static_cast<float>(height)})
                        .with_camera_xyz(rerun::components::ViewCoordinates::RDF));
  }

  bool enabled() const { return enabled_; }

  void logFrame(const FrameData& frame, const FrameResult& frame_result) {
    if (!enabled_) {
      return;
    }

    rec_.set_time_timestamp_secs_since_epoch("capture_time", frame_result.timestamp_s);
    rec_.set_time_sequence("frame", frame_result.index);

    const Eigen::Quaterniond q = frame_result.world_from_cam.quaternion();
    rec_.log("world/camera",
             rerun::Transform3D(
                 rerun::components::Translation3D(static_cast<float>(frame_result.world_from_cam.t.x()),
                                                  static_cast<float>(frame_result.world_from_cam.t.y()),
                                                  static_cast<float>(frame_result.world_from_cam.t.z())),
                 rerun::Rotation3D(rerun::datatypes::Quaternion::from_wxyz(
                     static_cast<float>(q.w()),
                     static_cast<float>(q.x()),
                     static_cast<float>(q.y()),
                     static_cast<float>(q.z())))));

    trajectory_positions_.emplace_back(static_cast<float>(frame_result.world_from_cam.t.x()),
                                       static_cast<float>(frame_result.world_from_cam.t.y()),
                                       static_cast<float>(frame_result.world_from_cam.t.z()));
    rec_.log("world/trajectory", rerun::Points3D(trajectory_positions_));

    const size_t stride = std::max<size_t>(1, options_.rerun_log_every_n_frames);
    if (options_.rerun_log_images && (static_cast<size_t>(frame_result.index) % stride == 0U)) {
      rec_.log(
          "world/camera/rgb",
          rerun::Image(frame.rgb.ptr<uint8_t>(),
                       {frame.rgb.cols, frame.rgb.rows},
                       rerun::datatypes::ColorModel::BGR));

      rec_.log(
          "world/camera/depth",
          rerun::DepthImage(frame.depth_u16.ptr<uint16_t>(),
                            {frame.depth_u16.cols, frame.depth_u16.rows})
              .with_meter(depth_scale_m_per_unit_));
    }
  }

 private:
  PipelineOptions options_;
  rerun::RecordingStream rec_;
  float depth_scale_m_per_unit_ = 0.0f;
  bool enabled_ = true;
  std::vector<rerun::components::Position3D> trajectory_positions_;
};
#endif

}  // namespace

PipelineResult runPipeline(const PipelineOptions& options) {
  PipelineResult result;

  DatasetLoader loader(options.dataset_root, options.dataset_format);
  std::string error;
  if (!loader.loadMetadata(&error)) {
    result.message = error;
    return result;
  }

  result.dataset_format = loader.datasetFormat();

  CameraIntrinsics intrinsics = loader.intrinsics();
  if (options.override_intrinsics) {
    if (!options.intrinsics_override.isValid()) {
      result.message = "Invalid intrinsics override provided.";
      return result;
    }
    intrinsics = options.intrinsics_override;
  }
  result.intrinsics = intrinsics;

  const float depth_scale_m_per_unit = options.depth_scale_m_per_unit > 0.0f
                                           ? options.depth_scale_m_per_unit
                                           : loader.recommendedDepthScaleMPerUnit();

  const size_t total_available = loader.frames().size();
  const size_t frame_limit = options.max_frames > 0 ? std::min(options.max_frames, total_available) : total_available;

  if (frame_limit == 0) {
    result.message = "No frames available to process.";
    return result;
  }

  PoseEstimatorParams pose_params;
  pose_params.min_depth_m = options.min_depth_m;
  pose_params.max_depth_m = options.max_depth_m;

  DepthPoseEstimator estimator(pose_params);
  ImageProcessor image_processor(options.use_cuda);
  DepthProcessor depth_processor(options.use_cuda);
  depth_processor.setDepthScale(depth_scale_m_per_unit);
  depth_processor.setDepthLimits(options.min_depth_m, options.max_depth_m);

  result.frames.reserve(frame_limit);
  result.trajectory.reserve(frame_limit);

  Pose world_from_cam = Pose::Identity();
  FrameData prev_frame;

  double inlier_sum = 0.0;
  size_t tracked_count = 0;

#ifdef CUVSLAM_WITH_RERUN
  std::unique_ptr<RerunLogger> rerun_logger;
#endif
  std::string rerun_warning;

  const auto run_start = std::chrono::steady_clock::now();
#ifdef CUVSLAM_WITH_RERUN
  const bool need_rgb_color = options.enable_rerun && options.rerun_log_images;
#else
  const bool need_rgb_color = false;
#endif

  struct LoadedFrame {
    FrameData frame;
    std::string error;
    bool ok = false;
    double load_ms = 0.0;
  };

  auto load_frame = [&loader, need_rgb_color](size_t index) {
    LoadedFrame loaded;
    const auto load_start = std::chrono::steady_clock::now();
    loaded.ok = loader.loadFrame(index, loaded.frame, need_rgb_color, &loaded.error);
    const auto load_end = std::chrono::steady_clock::now();
    loaded.load_ms = elapsedMs(load_start, load_end);
    return loaded;
  };

  const unsigned hw_threads = std::thread::hardware_concurrency();
  const size_t prefetch_window = std::max<size_t>(
      2,
      std::min<size_t>(8, hw_threads > 0 ? static_cast<size_t>(hw_threads / 2) : 2));

  std::unordered_map<size_t, std::future<LoadedFrame>> prefetch_futures;
  prefetch_futures.reserve(prefetch_window + 1);

  auto schedule_prefetch = [&](size_t index) {
    if (index >= frame_limit) {
      return;
    }
    if (prefetch_futures.find(index) != prefetch_futures.end()) {
      return;
    }
    prefetch_futures.emplace(index, std::async(std::launch::async, load_frame, index));
  };

  const size_t initial_prefetch = std::min(frame_limit, prefetch_window);
  for (size_t i = 0; i < initial_prefetch; ++i) {
    schedule_prefetch(i);
  }

  for (size_t i = 0; i < frame_limit; ++i) {
    const auto frame_start = std::chrono::steady_clock::now();
    StageTimings timings;

    schedule_prefetch(i);
    auto future_it = prefetch_futures.find(i);
    if (future_it == prefetch_futures.end()) {
      result.message = "Internal prefetch error: missing frame future for index " + std::to_string(i);
      return result;
    }
    LoadedFrame loaded_frame = future_it->second.get();
    prefetch_futures.erase(future_it);

    schedule_prefetch(i + prefetch_window);

    if (!loaded_frame.ok) {
      result.message = "Frame load failed: " + loaded_frame.error;
      return result;
    }

    FrameData frame = std::move(loaded_frame.frame);
    timings.load_ms = loaded_frame.load_ms;

    const auto pre_start = std::chrono::steady_clock::now();
    if (frame.gray.empty()) {
      if (!image_processor.convertBgrToGray(frame.rgb, frame.gray)) {
        result.message = "Gray conversion failed for frame " + frame.meta.frame_id;
        return result;
      }
    }
    if (!depth_processor.convertToMeters(frame.depth_u16, frame.depth_m)) {
      result.message = "Depth conversion failed for frame " + frame.meta.frame_id;
      return result;
    }
    const auto pre_end = std::chrono::steady_clock::now();
    timings.preprocess_ms = elapsedMs(pre_start, pre_end);

    FrameResult frame_result;
    frame_result.index = frame.meta.index;
    frame_result.timestamp_s = frame.meta.timestamp_s;

    if (i == 0) {
      frame_result.world_from_cam = world_from_cam;
      frame_result.tracking.pose_valid = true;
    } else {
      TrackingStats stats = estimator.estimate(prev_frame, frame, intrinsics, &timings);
      frame_result.tracking = stats;

      if (stats.pose_valid) {
        const Pose prev_from_curr = stats.relative_curr_from_prev.inverse();
        world_from_cam = world_from_cam * prev_from_curr;
        ++tracked_count;
        inlier_sum += static_cast<double>(stats.inlier_matches);
      }

      frame_result.world_from_cam = world_from_cam;
    }

    const auto frame_end = std::chrono::steady_clock::now();
    timings.total_ms = elapsedMs(frame_start, frame_end);
    frame_result.timings = timings;

    result.frames.push_back(frame_result);
    result.trajectory.push_back({frame_result.timestamp_s, frame_result.world_from_cam});

#ifdef CUVSLAM_WITH_RERUN
    if (options.enable_rerun && !rerun_logger) {
      const int frame_width = !frame.rgb.empty() ? frame.rgb.cols : frame.gray.cols;
      const int frame_height = !frame.rgb.empty() ? frame.rgb.rows : frame.gray.rows;
      rerun_logger = std::make_unique<RerunLogger>(
          options,
          intrinsics,
          frame_width,
          frame_height,
          depth_scale_m_per_unit,
          &rerun_warning);
    }
    if (rerun_logger && rerun_logger->enabled()) {
      rerun_logger->logFrame(frame, frame_result);
    }
#else
    if (options.enable_rerun && rerun_warning.empty()) {
      rerun_warning = "Rerun visualization requested but binary was built without CUVSLAM_ENABLE_RERUN=ON.";
    }
#endif

    prev_frame = std::move(frame);
  }

  const auto run_end = std::chrono::steady_clock::now();
  result.summary.total_frames = result.frames.size();
  result.summary.tracked_frames = tracked_count;
  result.summary.total_time_ms = elapsedMs(run_start, run_end);
  result.summary.average_fps = result.summary.total_time_ms > 0.0
                                   ? static_cast<double>(result.summary.total_frames) * 1000.0 / result.summary.total_time_ms
                                   : 0.0;
  result.summary.avg_inliers = tracked_count > 0 ? inlier_sum / static_cast<double>(tracked_count) : 0.0;

  const fs::path out_dir = options.output_dir.empty() ? fs::path("outputs") : fs::path(options.output_dir);
  fs::create_directories(out_dir);

  result.trajectory_path = (out_dir / "estimated_trajectory.tum").string();
  result.frame_metrics_path = (out_dir / "frame_metrics.csv").string();
  result.report_path = (out_dir / "performance_report.md").string();

  if (!writeTumTrajectory(result.trajectory_path, result.trajectory, &error)) {
    result.message = error;
    return result;
  }

  if (!writeFrameMetricsCsv(result.frame_metrics_path, result.frames, &error)) {
    result.message = error;
    return result;
  }

  std::string reference_path = options.reference_tum_path;
  if (reference_path.empty()) {
    const fs::path custom_ref = fs::path(options.dataset_root) / "orbslam3_poses.tum";
    const fs::path tum_ref = fs::path(options.dataset_root) / "groundtruth.txt";

    if (fs::exists(custom_ref)) {
      reference_path = custom_ref.string();
    } else if (fs::exists(tum_ref)) {
      reference_path = tum_ref.string();
    }
  }

  if (!reference_path.empty()) {
    std::string ref_error;
    const std::vector<TrajectorySample> reference = readTumTrajectory(reference_path, &ref_error);
    if (!reference.empty()) {
      result.evaluation = evaluateTrajectory(result.trajectory, reference, options.evaluation_timestamp_tolerance_s);
    }
  }

  if (!writePerformanceReport(result.report_path, result, depth_scale_m_per_unit, &error)) {
    result.message = error;
    return result;
  }

  result.success = true;
  result.message = "Pipeline finished successfully.";
  if (!rerun_warning.empty()) {
    result.message += " " + rerun_warning;
  }
  return result;
}

}  // namespace cuvslam
