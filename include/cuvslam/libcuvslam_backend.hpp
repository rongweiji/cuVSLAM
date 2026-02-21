#pragma once

#include "cuvslam/types.hpp"

#include <memory>
#include <string>

namespace cuvslam {

struct LibCuVSLAMOptions {
  std::string library_path;
  bool use_gpu = true;
  float depth_scale_m_per_unit = 0.001f;
  int verbosity = 0;
};

class LibCuVSLAMBackend {
 public:
  LibCuVSLAMBackend();
  ~LibCuVSLAMBackend();

  LibCuVSLAMBackend(const LibCuVSLAMBackend&) = delete;
  LibCuVSLAMBackend& operator=(const LibCuVSLAMBackend&) = delete;

  bool initialize(const CameraIntrinsics& intrinsics,
                  int width,
                  int height,
                  const LibCuVSLAMOptions& options,
                  std::string* error);

  bool track(const FrameData& frame,
             Pose& world_from_cam,
             TrackingStats* stats,
             double* track_time_ms,
             std::string* error);

  bool isInitialized() const;
  std::string loadedLibraryPath() const;
  std::string versionString() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cuvslam
