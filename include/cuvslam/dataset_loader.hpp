#pragma once

#include "cuvslam/types.hpp"

#include <string>
#include <vector>

namespace cuvslam {

class DatasetLoader {
 public:
  explicit DatasetLoader(std::string dataset_root, DatasetFormat dataset_format = DatasetFormat::kAuto);

  bool loadMetadata(std::string* error = nullptr);

  const CameraIntrinsics& intrinsics() const { return intrinsics_; }
  const std::vector<FramePaths>& frames() const { return frames_; }
  const std::string& datasetRoot() const { return dataset_root_; }
  DatasetFormat datasetFormat() const { return dataset_format_; }
  float recommendedDepthScaleMPerUnit() const { return recommended_depth_scale_m_per_unit_; }

  bool loadFrame(size_t index,
                 FrameData& frame,
                 bool load_rgb_color = true,
                 std::string* error = nullptr) const;

 private:
  bool detectDatasetFormat(std::string* error);
  bool parseCustomDataset(std::string* error);
  bool parseTumRgbdDataset(std::string* error);
  bool parseCalibration(const std::string& path, std::string* error);
  bool parseTimestamps(const std::string& path, std::string* error);
  bool parseTumIntrinsics(std::string* error);

  std::string dataset_root_;
  DatasetFormat dataset_format_ = DatasetFormat::kAuto;
  CameraIntrinsics intrinsics_;
  float recommended_depth_scale_m_per_unit_ = 0.001f;
  std::vector<FramePaths> frames_;
};

}  // namespace cuvslam
