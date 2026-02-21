#include "cuvslam/libcuvslam_backend.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace cuvslam {
namespace fs = std::filesystem;

namespace {

namespace capi {

enum CUVSLAM_ImageEncoding { MONO8, RGB8 };
enum CUVSLAM_OdometryMode { Multicamera, Inertial, RGBD };

struct CUVSLAM_Pose {
  float r[9];
  float t[3];
};

struct CUVSLAM_ImuCalibration {
  struct CUVSLAM_Pose rig_from_imu;
  float gyroscope_noise_density;
  float gyroscope_random_walk;
  float accelerometer_noise_density;
  float accelerometer_random_walk;
  float frequency;
};

struct CUVSLAM_Camera {
  const char* distortion_model;
  const float* parameters;
  int32_t num_parameters;
  int32_t width;
  int32_t height;
  int32_t border_top;
  int32_t border_bottom;
  int32_t border_left;
  int32_t border_right;
  struct CUVSLAM_Pose pose;
};

struct CUVSLAM_CameraRig {
  const struct CUVSLAM_Camera* cameras;
  int32_t num_cameras;
};

struct CUVSLAM_RGBDOdometrySettings {
  float depth_scale_factor;
  int32_t depth_camera_id;
  int32_t enable_depth_stereo_tracking;
};

struct CUVSLAM_SlamCameras {
  uint32_t num;
  int32_t* camera_list;
};

struct CUVSLAM_LocalizationSettings {
  float horizontal_search_radius;
  float vertical_search_radius;
  float horizontal_step;
  float vertical_step;
  float angular_step_rads;
};

struct CUVSLAM_Configuration {
  int32_t use_motion_model;
  int32_t use_denoising;
  int32_t use_gpu;
  int32_t horizontal_stereo_camera;
  int32_t enable_observations_export;
  int32_t enable_landmarks_export;
  int32_t enable_localization_n_mapping;
  float map_cell_size;
  int32_t slam_sync_mode;
  struct CUVSLAM_SlamCameras slam_cameras;
  int32_t enable_reading_slam_internals;
  const char* debug_dump_directory;
  float max_frame_delta_s;
  struct CUVSLAM_ImuCalibration imu_calibration;
  enum CUVSLAM_OdometryMode odometry_mode;
  struct CUVSLAM_RGBDOdometrySettings rgbd_settings;
  int32_t planar_constraints;
  int32_t debug_imu_mode;
  int32_t slam_gt_align_mode;
  int32_t slam_max_map_size;
  uint64_t slam_throttling_time_ms;
  int32_t multicam_mode;
  struct CUVSLAM_LocalizationSettings localization_settings;
};

// Legacy API v12.x configuration layout (no RGBD settings in C API).
struct CUVSLAM_ConfigurationLegacy {
  int32_t use_motion_model;
  int32_t use_denoising;
  int32_t use_gpu;
  int32_t horizontal_stereo_camera;
  int32_t enable_observations_export;
  int32_t enable_landmarks_export;
  int32_t enable_localization_n_mapping;
  float map_cell_size;
  int32_t slam_sync_mode;
  int32_t enable_reading_slam_internals;
  const char* debug_dump_directory;
  float max_frame_delta_ms;
  struct CUVSLAM_ImuCalibration imu_calibration;
  int32_t enable_imu_fusion;
  int32_t planar_constraints;
  int32_t debug_imu_mode;
  int32_t slam_gt_align_mode;
  int32_t slam_max_map_size;
  int32_t multicam_mode;
};

struct CUVSLAM_Tracker;
typedef struct CUVSLAM_Tracker* CUVSLAM_TrackerHandle;

struct CUVSLAM_Image {
  const uint8_t* pixels;
  int64_t timestamp_ns;
  int32_t width;
  int32_t height;
  int32_t camera_index;
  int32_t pitch;
  enum CUVSLAM_ImageEncoding image_encoding;
  const uint8_t* input_mask;
  int32_t mask_width;
  int32_t mask_height;
  int32_t mask_pitch;
};

// Legacy API v12.x image layout (no input mask fields).
struct CUVSLAM_ImageLegacy {
  const uint8_t* pixels;
  int64_t timestamp_ns;
  int32_t width;
  int32_t height;
  int32_t camera_index;
  int32_t pitch;
  enum CUVSLAM_ImageEncoding image_encoding;
};

struct CUVSLAM_DepthImage {
  const uint16_t* pixels;
  int64_t timestamp_ns;
  int32_t width;
  int32_t height;
  int32_t camera_index;
  int32_t pitch;
};

struct CUVSLAM_PoseEstimate {
  struct CUVSLAM_Pose pose;
  int64_t timestamp_ns;
  float covariance[6 * 6];
};

typedef uint32_t CUVSLAM_Status;

using FnGetVersion = void (*)(int32_t*, int32_t*, const char**);
using FnSetVerbosity = void (*)(int);
using FnWarmUpGPU = void (*)();
using FnInitDefaultConfiguration = void (*)(void*);
using FnCreateTracker = CUVSLAM_Status (*)(CUVSLAM_TrackerHandle*,
                                           const CUVSLAM_CameraRig*,
                                           const void*);
using FnDestroyTracker = void (*)(CUVSLAM_TrackerHandle);
using FnTrackModern = CUVSLAM_Status (*)(CUVSLAM_TrackerHandle,
                                         const CUVSLAM_Image*,
                                         size_t,
                                         const CUVSLAM_DepthImage*,
                                         const CUVSLAM_Pose*,
                                         CUVSLAM_PoseEstimate*);
using FnTrackLegacy = CUVSLAM_Status (*)(CUVSLAM_TrackerHandle,
                                         const CUVSLAM_ImageLegacy*,
                                         size_t,
                                         const CUVSLAM_Pose*,
                                         CUVSLAM_PoseEstimate*);

}  // namespace capi

constexpr capi::CUVSLAM_Status kCuvslamSuccess = 0U;
constexpr capi::CUVSLAM_Status kCuvslamTrackingLost = 1U;
constexpr capi::CUVSLAM_Status kCuvslamGenericError = 4U;

std::string statusToString(capi::CUVSLAM_Status status) {
  switch (status) {
    case 0:
      return "CUVSLAM_SUCCESS";
    case 1:
      return "CUVSLAM_TRACKING_LOST";
    case 2:
      return "CUVSLAM_INVALID_ARG";
    case 3:
      return "CUVSLAM_CAN_NOT_LOCALIZE";
    case 4:
      return "CUVSLAM_GENERIC_ERROR";
    case 5:
      return "CUVSLAM_UNSUPPORTED_NUMBER_OF_CAMERAS";
    case 6:
      return "CUVSLAM_SLAM_IS_NOT_INITIALIZED";
    case 7:
      return "CUVSLAM_NOT_IMPLEMENTED";
    case 8:
      return "CUVSLAM_READING_SLAM_INTERNALS_DISABLED";
    default:
      return "CUVSLAM_STATUS_" + std::to_string(status);
  }
}

void setIdentityPose(capi::CUVSLAM_Pose* pose) {
  std::memset(pose, 0, sizeof(*pose));
  pose->r[0] = 1.0f;
  pose->r[4] = 1.0f;
  pose->r[8] = 1.0f;
}

Pose toPose(const capi::CUVSLAM_Pose& in) {
  Pose out;
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      out.R(row, col) = static_cast<double>(in.r[col * 3 + row]);
    }
  }
  out.t = Eigen::Vector3d(static_cast<double>(in.t[0]),
                          static_cast<double>(in.t[1]),
                          static_cast<double>(in.t[2]));
  return out;
}

template <typename FnType>
bool loadSymbol(void* handle, const char* name, FnType* fn, std::string* error) {
#if defined(__linux__)
  dlerror();
  void* raw = dlsym(handle, name);
  const char* dl_err = dlerror();
  if (dl_err != nullptr || raw == nullptr) {
    if (error) {
      std::ostringstream oss;
      oss << "Failed to load symbol '" << name << "'";
      if (dl_err != nullptr) {
        oss << ": " << dl_err;
      }
      *error = oss.str();
    }
    return false;
  }
  *fn = reinterpret_cast<FnType>(raw);
  return true;
#else
  (void)handle;
  (void)name;
  (void)fn;
  if (error) {
    *error = "Dynamic loading of libcuvslam is only supported on Linux.";
  }
  return false;
#endif
}

std::vector<std::string> candidateLibraryPaths(const std::string& requested_path) {
  std::vector<std::string> candidates;
  if (!requested_path.empty()) {
    candidates.push_back(requested_path);
  }

#ifdef CUVSLAM_DEFAULT_LIBRARY_PATH
  candidates.emplace_back(CUVSLAM_DEFAULT_LIBRARY_PATH);
#endif

  if (const char* env_path = std::getenv("CUVSLAM_LIB_PATH")) {
    if (*env_path != '\0') {
      candidates.emplace_back(env_path);
    }
  }
  if (const char* env_path = std::getenv("CUVSLAM_LIBRARY_PATH")) {
    if (*env_path != '\0') {
      candidates.emplace_back(env_path);
    }
  }

  candidates.emplace_back("libcuvslam.so");

  std::vector<std::string> unique;
  std::unordered_set<std::string> seen;
  unique.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (candidate.empty()) {
      continue;
    }
    if (seen.insert(candidate).second) {
      unique.push_back(candidate);
    }
  }
  return unique;
}

bool looksLikeFilesystemPath(const std::string& path) {
  return path.find('/') != std::string::npos;
}

}  // namespace

struct LibCuVSLAMBackend::Impl {
#if defined(__linux__)
  void* library_handle = nullptr;
#endif
  capi::CUVSLAM_TrackerHandle tracker = nullptr;

  capi::FnGetVersion get_version = nullptr;
  capi::FnSetVerbosity set_verbosity = nullptr;
  capi::FnWarmUpGPU warm_up_gpu = nullptr;
  capi::FnInitDefaultConfiguration init_default_config = nullptr;
  capi::FnCreateTracker create_tracker = nullptr;
  capi::FnDestroyTracker destroy_tracker = nullptr;
  void* track_symbol = nullptr;
  capi::FnTrackModern track_modern = nullptr;
  capi::FnTrackLegacy track_legacy = nullptr;

  std::array<float, 4> pinhole_parameters = {0.0f, 0.0f, 0.0f, 0.0f};
  capi::CUVSLAM_Camera camera = {};
  capi::CUVSLAM_CameraRig rig = {};

  std::string loaded_library_path;
  std::string version_string;
  int api_major = 0;
  bool use_modern_rgbd = true;
  bool initialized = false;
  int width = 0;
  int height = 0;

  void shutdown() {
    if (tracker != nullptr && destroy_tracker != nullptr) {
      destroy_tracker(tracker);
    }
    tracker = nullptr;
    initialized = false;

#if defined(__linux__)
    if (library_handle != nullptr) {
      dlclose(library_handle);
      library_handle = nullptr;
    }
#endif
    get_version = nullptr;
    set_verbosity = nullptr;
    warm_up_gpu = nullptr;
    init_default_config = nullptr;
    create_tracker = nullptr;
    destroy_tracker = nullptr;
    track_symbol = nullptr;
    track_modern = nullptr;
    track_legacy = nullptr;

    loaded_library_path.clear();
    version_string.clear();
    api_major = 0;
    use_modern_rgbd = true;
  }

  ~Impl() { shutdown(); }
};

LibCuVSLAMBackend::LibCuVSLAMBackend()
    : impl_(std::make_unique<Impl>()) {}

LibCuVSLAMBackend::~LibCuVSLAMBackend() = default;

bool LibCuVSLAMBackend::initialize(const CameraIntrinsics& intrinsics,
                                   int width,
                                   int height,
                                   const LibCuVSLAMOptions& options,
                                   std::string* error) {
  if (impl_->initialized) {
    return true;
  }

#if !defined(__linux__)
  if (error) {
    *error = "libcuvslam backend is only supported on Linux.";
  }
  return false;
#else
  if (!intrinsics.isValid()) {
    if (error) {
      *error = "Invalid intrinsics for libcuvslam backend.";
    }
    return false;
  }
  if (width <= 0 || height <= 0) {
    if (error) {
      *error = "Invalid frame size for libcuvslam backend initialization.";
    }
    return false;
  }

  const std::vector<std::string> candidates = candidateLibraryPaths(options.library_path);
  std::ostringstream attempts;
  for (const auto& candidate : candidates) {
    if (looksLikeFilesystemPath(candidate) && !fs::exists(candidate)) {
      attempts << "  - " << candidate << ": not found\n";
      continue;
    }

    void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      attempts << "  - " << candidate << ": " << dlerror() << "\n";
      continue;
    }

    impl_->library_handle = handle;
    impl_->loaded_library_path = candidate;
    break;
  }

  if (impl_->library_handle == nullptr) {
    if (error) {
      std::ostringstream oss;
      oss << "Unable to load libcuvslam.so. Tried:\n" << attempts.str();
      *error = oss.str();
    }
    return false;
  }

  std::string sym_error;
  if (!loadSymbol(impl_->library_handle, "CUVSLAM_GetVersion", &impl_->get_version, &sym_error) ||
      !loadSymbol(impl_->library_handle, "CUVSLAM_SetVerbosity", &impl_->set_verbosity, &sym_error) ||
      !loadSymbol(impl_->library_handle, "CUVSLAM_WarmUpGPU", &impl_->warm_up_gpu, &sym_error) ||
      !loadSymbol(impl_->library_handle, "CUVSLAM_InitDefaultConfiguration", &impl_->init_default_config, &sym_error) ||
      !loadSymbol(impl_->library_handle, "CUVSLAM_CreateTracker", &impl_->create_tracker, &sym_error) ||
      !loadSymbol(impl_->library_handle, "CUVSLAM_DestroyTracker", &impl_->destroy_tracker, &sym_error) ||
      !loadSymbol(impl_->library_handle, "CUVSLAM_Track", &impl_->track_symbol, &sym_error)) {
    impl_->shutdown();
    if (error) {
      *error = sym_error;
    }
    return false;
  }

  int32_t version_major = 0;
  int32_t version_minor = 0;
  const char* version_text = nullptr;
  impl_->get_version(&version_major, &version_minor, &version_text);
  if (version_text != nullptr) {
    impl_->version_string = version_text;
  } else {
    impl_->version_string = std::to_string(version_major) + "." + std::to_string(version_minor);
  }
  impl_->api_major = version_major;
  impl_->use_modern_rgbd = version_major >= 14;
  if (impl_->use_modern_rgbd) {
    impl_->track_modern = reinterpret_cast<capi::FnTrackModern>(impl_->track_symbol);
  } else {
    impl_->track_legacy = reinterpret_cast<capi::FnTrackLegacy>(impl_->track_symbol);
  }
  if ((impl_->use_modern_rgbd && impl_->track_modern == nullptr) ||
      (!impl_->use_modern_rgbd && impl_->track_legacy == nullptr)) {
    impl_->shutdown();
    if (error) {
      *error = "Failed to resolve CUVSLAM_Track entry point.";
    }
    return false;
  }

  if (version_major <= 0) {
    impl_->shutdown();
    if (error) {
      *error = "libcuvslam returned invalid API version.";
    }
    return false;
  }

  impl_->set_verbosity(options.verbosity);
  const bool backend_use_gpu = true;
  if (backend_use_gpu) {
    impl_->warm_up_gpu();
  }

  impl_->pinhole_parameters[0] = static_cast<float>(intrinsics.cx);
  impl_->pinhole_parameters[1] = static_cast<float>(intrinsics.cy);
  impl_->pinhole_parameters[2] = static_cast<float>(intrinsics.fx);
  impl_->pinhole_parameters[3] = static_cast<float>(intrinsics.fy);

  impl_->camera = {};
  impl_->camera.distortion_model = "pinhole";
  impl_->camera.parameters = impl_->pinhole_parameters.data();
  impl_->camera.num_parameters = static_cast<int32_t>(impl_->pinhole_parameters.size());
  impl_->camera.width = width;
  impl_->camera.height = height;
  impl_->camera.border_top = 0;
  impl_->camera.border_bottom = 0;
  impl_->camera.border_left = 0;
  impl_->camera.border_right = 0;
  setIdentityPose(&impl_->camera.pose);

  impl_->rig = {};
  impl_->rig.cameras = &impl_->camera;
  impl_->rig.num_cameras = 1;

  capi::CUVSLAM_Status status = kCuvslamGenericError;
  if (impl_->use_modern_rgbd) {
    capi::CUVSLAM_Configuration cfg = {};
    impl_->init_default_config(&cfg);
    cfg.use_motion_model = 1;
    cfg.use_denoising = 0;
    cfg.use_gpu = backend_use_gpu ? 1 : 0;
    cfg.horizontal_stereo_camera = 0;
    cfg.enable_observations_export = 0;
    cfg.enable_landmarks_export = 0;
    cfg.enable_localization_n_mapping = 0;
    cfg.debug_dump_directory = nullptr;
    cfg.odometry_mode = capi::CUVSLAM_OdometryMode::RGBD;
    cfg.rgbd_settings.depth_scale_factor =
        options.depth_scale_m_per_unit > 0.0f ? (1.0f / options.depth_scale_m_per_unit) : 1.0f;
    cfg.rgbd_settings.depth_camera_id = 0;
    cfg.rgbd_settings.enable_depth_stereo_tracking = 0;
    status = impl_->create_tracker(&impl_->tracker, &impl_->rig, &cfg);
  } else {
    capi::CUVSLAM_ConfigurationLegacy cfg = {};
    impl_->init_default_config(&cfg);
    cfg.use_motion_model = 1;
    cfg.use_denoising = 0;
    cfg.use_gpu = backend_use_gpu ? 1 : 0;
    cfg.horizontal_stereo_camera = 0;
    cfg.enable_observations_export = 0;
    cfg.enable_landmarks_export = 0;
    cfg.enable_localization_n_mapping = 0;
    cfg.debug_dump_directory = nullptr;
    cfg.enable_imu_fusion = 0;
    status = impl_->create_tracker(&impl_->tracker, &impl_->rig, &cfg);
  }

  if (status != kCuvslamSuccess || impl_->tracker == nullptr) {
    std::string msg = "CUVSLAM_CreateTracker failed with status " + statusToString(status);
    if (status == kCuvslamGenericError) {
      msg += ". Check NVIDIA driver/CUDA compatibility for this cuVSLAM build.";
    } else if (status == 7U && impl_->use_modern_rgbd) {
      msg += ". This cuVSLAM build may not support RGB-D mode.";
    }
    impl_->shutdown();
    if (error) {
      *error = msg;
    }
    return false;
  }

  impl_->width = width;
  impl_->height = height;
  impl_->initialized = true;
  return true;
#endif
}

bool LibCuVSLAMBackend::track(const FrameData& frame,
                              Pose& world_from_cam,
                              TrackingStats* stats,
                              double* track_time_ms,
                              std::string* error) {
  if (track_time_ms) {
    *track_time_ms = 0.0;
  }
  if (stats) {
    *stats = TrackingStats{};
  }

  if (!impl_->initialized || impl_->tracker == nullptr) {
    if (error) {
      *error = "libcuvslam backend is not initialized.";
    }
    return false;
  }

  if (frame.gray.empty() || frame.gray.type() != CV_8UC1) {
    if (error) {
      *error = "libcuvslam backend expects gray images (CV_8UC1).";
    }
    return false;
  }
  if (frame.gray.cols != impl_->width || frame.gray.rows != impl_->height) {
    if (error) {
      *error = "Frame size changed after libcuvslam initialization.";
    }
    return false;
  }

  if (impl_->use_modern_rgbd &&
      (frame.depth_u16.empty() || frame.depth_u16.type() != CV_16UC1 ||
       frame.depth_u16.cols != impl_->width || frame.depth_u16.rows != impl_->height)) {
    if (error) {
      *error = "libcuvslam RGB-D mode expects depth images in CV_16UC1 with matching frame size.";
    }
    return false;
  }

  const int64_t timestamp_ns = static_cast<int64_t>(std::llround(frame.meta.timestamp_s * 1e9));

  capi::CUVSLAM_PoseEstimate estimate = {};
  const auto t0 = std::chrono::steady_clock::now();
  capi::CUVSLAM_Status status = kCuvslamGenericError;
  if (impl_->use_modern_rgbd) {
    capi::CUVSLAM_Image image = {};
    image.pixels = frame.gray.ptr<uint8_t>();
    image.timestamp_ns = timestamp_ns;
    image.width = frame.gray.cols;
    image.height = frame.gray.rows;
    image.camera_index = 0;
    image.pitch = static_cast<int32_t>(frame.gray.step);
    image.image_encoding = capi::CUVSLAM_ImageEncoding::MONO8;
    image.input_mask = nullptr;
    image.mask_width = 0;
    image.mask_height = 0;
    image.mask_pitch = 0;

    capi::CUVSLAM_DepthImage depth = {};
    depth.pixels = frame.depth_u16.ptr<uint16_t>();
    depth.timestamp_ns = timestamp_ns;
    depth.width = frame.depth_u16.cols;
    depth.height = frame.depth_u16.rows;
    depth.camera_index = 0;
    depth.pitch = static_cast<int32_t>(frame.depth_u16.step);
    status = impl_->track_modern(impl_->tracker, &image, 1, &depth, nullptr, &estimate);
  } else {
    capi::CUVSLAM_ImageLegacy image = {};
    image.pixels = frame.gray.ptr<uint8_t>();
    image.timestamp_ns = timestamp_ns;
    image.width = frame.gray.cols;
    image.height = frame.gray.rows;
    image.camera_index = 0;
    image.pitch = static_cast<int32_t>(frame.gray.step);
    image.image_encoding = capi::CUVSLAM_ImageEncoding::MONO8;
    status = impl_->track_legacy(impl_->tracker, &image, 1, nullptr, &estimate);
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (track_time_ms) {
    *track_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  if (status == kCuvslamSuccess) {
    Pose pose = toPose(estimate.pose);
    if (!isFinitePose(pose)) {
      if (error) {
        *error = "libcuvslam returned non-finite pose.";
      }
      return false;
    }
    world_from_cam = pose;
    if (stats) {
      stats->pose_valid = true;
    }
    return true;
  }

  if (status == kCuvslamTrackingLost) {
    if (stats) {
      stats->pose_valid = false;
    }
    return true;
  }

  if (error) {
    *error = "CUVSLAM_Track failed with status " + statusToString(status);
  }
  return false;
}

bool LibCuVSLAMBackend::isInitialized() const {
  return impl_->initialized;
}

std::string LibCuVSLAMBackend::loadedLibraryPath() const {
  return impl_->loaded_library_path;
}

std::string LibCuVSLAMBackend::versionString() const {
  return impl_->version_string;
}

}  // namespace cuvslam
