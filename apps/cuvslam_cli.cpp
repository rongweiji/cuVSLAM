#include "cuvslam/pipeline.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printUsage() {
  std::cout << "Usage: cuvslam_cli [options]\n\n"
            << "Options:\n"
            << "  --dataset_root <path>         Dataset root path\n"
            << "  --dataset_format <fmt>        auto|custom|tum (default: auto)\n"
            << "  --libcuvslam_path <path>      Optional explicit path to libcuvslam.so\n"
            << "  --libcuvslam_verbosity <N>    libcuvslam verbosity (default: 0)\n"
            << "  --output_dir <path>           Output directory for trajectory and reports\n"
            << "  --reference_tum <path>        Optional reference trajectory in TUM format\n"
            << "  --max_frames <N>              Process only first N frames (default: all)\n"
            << "  --depth_scale <value>         Depth meters per raw unit (default: auto per dataset)\n"
            << "  --eval_tolerance_s <value>    Timestamp tolerance for eval matching (default: 0.02)\n"
            << "  --fx --fy --cx --cy <value>   Override camera intrinsics\n"
            << "  --no_cuda                     Disable CUDA grayscale preprocessing\n"
            << "  --enable_rerun                Enable Rerun logging support\n"
            << "  --rerun_spawn                 Spawn Rerun viewer (requires rerun executable in PATH)\n"
            << "  --rerun_save <file.rrd>       Save Rerun stream to an .rrd file\n"
            << "  --rerun_no_images             Disable image/depth logging to Rerun\n"
            << "  --rerun_log_every_n <N>       Log images every N frames (default: 3)\n"
            << "  --realtime                    Pace processing to dataset timestamps (real-time playback)\n"
            << "  --realtime_speed <value>      Real-time playback speed multiplier (default: 1.0)\n"
            << "  --help                        Show this help\n";
}

bool readValue(int argc, char** argv, int& i, std::string& out) {
  if (i + 1 >= argc) {
    return false;
  }
  out = argv[++i];
  return true;
}

bool parseDatasetFormat(const std::string& value, cuvslam::DatasetFormat& format) {
  if (value == "auto") {
    format = cuvslam::DatasetFormat::kAuto;
    return true;
  }
  if (value == "custom" || value == "custom_iphone") {
    format = cuvslam::DatasetFormat::kCustomIphone;
    return true;
  }
  if (value == "tum" || value == "tum_rgbd") {
    format = cuvslam::DatasetFormat::kTumRgbd;
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  cuvslam::PipelineOptions options;

  bool fx_set = false;
  bool fy_set = false;
  bool cx_set = false;
  bool cy_set = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--help") {
      printUsage();
      return 0;
    }

    if (arg == "--no_cuda") {
      options.use_cuda = false;
      continue;
    }

    if (arg == "--enable_rerun") {
      options.enable_rerun = true;
      continue;
    }

    if (arg == "--rerun_spawn") {
      options.enable_rerun = true;
      options.rerun_spawn = true;
      continue;
    }

    if (arg == "--rerun_no_images") {
      options.enable_rerun = true;
      options.rerun_log_images = false;
      continue;
    }

    if (arg == "--realtime") {
      options.realtime_playback = true;
      continue;
    }

    std::string value;
    if (arg == "--dataset_root") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --dataset_root\n";
        return 1;
      }
      options.dataset_root = value;
    } else if (arg == "--libcuvslam_path") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --libcuvslam_path\n";
        return 1;
      }
      options.libcuvslam_library_path = value;
    } else if (arg == "--libcuvslam_verbosity") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --libcuvslam_verbosity\n";
        return 1;
      }
      options.libcuvslam_verbosity = std::stoi(value);
    } else if (arg == "--dataset_format") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --dataset_format\n";
        return 1;
      }
      if (!parseDatasetFormat(value, options.dataset_format)) {
        std::cerr << "Unsupported dataset format: " << value << "\n";
        return 1;
      }
    } else if (arg == "--output_dir") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --output_dir\n";
        return 1;
      }
      options.output_dir = value;
    } else if (arg == "--reference_tum") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --reference_tum\n";
        return 1;
      }
      options.reference_tum_path = value;
    } else if (arg == "--max_frames") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --max_frames\n";
        return 1;
      }
      options.max_frames = static_cast<size_t>(std::stoull(value));
    } else if (arg == "--depth_scale") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --depth_scale\n";
        return 1;
      }
      options.depth_scale_m_per_unit = std::stof(value);
    } else if (arg == "--eval_tolerance_s") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --eval_tolerance_s\n";
        return 1;
      }
      options.evaluation_timestamp_tolerance_s = std::stod(value);
    } else if (arg == "--fx") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --fx\n";
        return 1;
      }
      options.intrinsics_override.fx = std::stod(value);
      fx_set = true;
    } else if (arg == "--fy") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --fy\n";
        return 1;
      }
      options.intrinsics_override.fy = std::stod(value);
      fy_set = true;
    } else if (arg == "--cx") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --cx\n";
        return 1;
      }
      options.intrinsics_override.cx = std::stod(value);
      cx_set = true;
    } else if (arg == "--cy") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --cy\n";
        return 1;
      }
      options.intrinsics_override.cy = std::stod(value);
      cy_set = true;
    } else if (arg == "--rerun_save") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --rerun_save\n";
        return 1;
      }
      options.enable_rerun = true;
      options.rerun_save_path = value;
    } else if (arg == "--rerun_log_every_n") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --rerun_log_every_n\n";
        return 1;
      }
      options.enable_rerun = true;
      options.rerun_log_every_n_frames = static_cast<size_t>(std::stoull(value));
    } else if (arg == "--realtime_speed") {
      if (!readValue(argc, argv, i, value)) {
        std::cerr << "Missing value for --realtime_speed\n";
        return 1;
      }
      options.realtime_playback = true;
      options.realtime_playback_speed = std::stod(value);
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      printUsage();
      return 1;
    }
  }

  const bool any_intrinsic_set = fx_set || fy_set || cx_set || cy_set;
  if (any_intrinsic_set) {
    if (!(fx_set && fy_set && cx_set && cy_set)) {
      std::cerr << "If overriding intrinsics, all of --fx --fy --cx --cy must be provided.\n";
      return 1;
    }
    options.override_intrinsics = true;
  }

  if (options.realtime_playback_speed <= 0.0) {
    std::cerr << "--realtime_speed must be > 0\n";
    return 1;
  }

  if (options.reference_tum_path.empty()) {
    const std::filesystem::path tum_ref = std::filesystem::path(options.dataset_root) / "groundtruth.txt";
    if (std::filesystem::exists(tum_ref)) {
      options.reference_tum_path = tum_ref.string();
    }
  }

  const cuvslam::PipelineResult result = cuvslam::runPipeline(options);
  if (!result.success) {
    std::cerr << "Pipeline failed: " << result.message << "\n";
    return 2;
  }

  std::cout << "Pipeline completed.\n";
  std::cout << "Dataset format: " << cuvslam::datasetFormatToString(result.dataset_format) << "\n";
  std::cout << "Backend used: " << cuvslam::kTrackingBackendName << "\n";
  if (!result.backend_details.empty()) {
    std::cout << "Backend details: " << result.backend_details << "\n";
  }
  std::cout << "Frames: " << result.summary.total_frames << "\n";
  std::cout << "Tracked frames: " << result.summary.tracked_frames << "\n";
  std::cout << "Average FPS: " << result.summary.average_fps << "\n";
  if (result.evaluation.has_reference) {
    std::cout << "ATE RMSE (m): " << result.evaluation.ate_rmse_m << "\n";
    std::cout << "RPE RMSE (m): " << result.evaluation.rpe_rmse_m << "\n";
  }
  std::cout << "Trajectory: " << result.trajectory_path << "\n";
  std::cout << "Frame metrics: " << result.frame_metrics_path << "\n";
  std::cout << "Performance report: " << result.report_path << "\n";
  std::cout << result.message << "\n";

  return 0;
}
