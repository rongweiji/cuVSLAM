#include "cuvslam/dataset_loader.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

namespace cuvslam {
namespace fs = std::filesystem;

namespace {

struct TimedPath {
  double timestamp_s = 0.0;
  std::string relative_path;
};

std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

bool parseNumericList(const std::string& path, std::vector<double>& values, std::string* error) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (error) {
      *error = "Failed to open file: " + path;
    }
    return false;
  }

  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();

  const std::regex number_regex(R"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)");
  std::sregex_iterator it(text.begin(), text.end(), number_regex);
  std::sregex_iterator end;

  values.clear();
  for (; it != end; ++it) {
    values.push_back(std::stod(it->str()));
  }

  return true;
}

bool parseTumListFile(const std::string& path, std::vector<TimedPath>& entries, std::string* error) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (error) {
      *error = "Failed to open list file: " + path;
    }
    return false;
  }

  entries.clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream ss(line);
    TimedPath entry;
    if (!(ss >> entry.timestamp_s >> entry.relative_path)) {
      continue;
    }

    if (entry.relative_path.rfind("./", 0) == 0) {
      entry.relative_path = entry.relative_path.substr(2);
    }

    entries.push_back(std::move(entry));
  }

  std::sort(entries.begin(), entries.end(), [](const TimedPath& a, const TimedPath& b) {
    return a.timestamp_s < b.timestamp_s;
  });

  if (entries.empty()) {
    if (error) {
      *error = "No entries found in list file: " + path;
    }
    return false;
  }

  return true;
}

}  // namespace

DatasetLoader::DatasetLoader(std::string dataset_root, DatasetFormat dataset_format)
    : dataset_root_(std::move(dataset_root)), dataset_format_(dataset_format) {}

bool DatasetLoader::loadMetadata(std::string* error) {
  frames_.clear();

  if (!detectDatasetFormat(error)) {
    return false;
  }

  if (dataset_format_ == DatasetFormat::kCustomIphone) {
    recommended_depth_scale_m_per_unit_ = 0.001f;
    return parseCustomDataset(error);
  }

  if (dataset_format_ == DatasetFormat::kTumRgbd) {
    recommended_depth_scale_m_per_unit_ = 1.0f / 5000.0f;
    return parseTumRgbdDataset(error);
  }

  if (error) {
    *error = "Unsupported dataset format.";
  }
  return false;
}

bool DatasetLoader::detectDatasetFormat(std::string* error) {
  if (dataset_format_ != DatasetFormat::kAuto) {
    return true;
  }

  const fs::path root(dataset_root_);
  const bool has_custom = fs::exists(root / "iphone_calibration.yaml") && fs::exists(root / "timestamps.txt") &&
                          fs::exists(root / "iphone_mono") && fs::exists(root / "iphone_mono_depth");
  const bool has_tum = fs::exists(root / "rgb.txt") && fs::exists(root / "depth.txt");

  if (has_custom) {
    dataset_format_ = DatasetFormat::kCustomIphone;
    return true;
  }

  if (has_tum) {
    dataset_format_ = DatasetFormat::kTumRgbd;
    return true;
  }

  if (error) {
    *error = "Failed to auto-detect dataset format. Expected either custom iPhone dataset or TUM RGB-D files.";
  }
  return false;
}

bool DatasetLoader::parseCustomDataset(std::string* error) {
  const std::string calibration_path = (fs::path(dataset_root_) / "iphone_calibration.yaml").string();
  if (!parseCalibration(calibration_path, error)) {
    return false;
  }

  const std::string timestamps_path = (fs::path(dataset_root_) / "timestamps.txt").string();
  if (!parseTimestamps(timestamps_path, error)) {
    return false;
  }

  if (frames_.empty()) {
    if (error) {
      *error = "No frames discovered from timestamps file.";
    }
    return false;
  }

  return true;
}

bool DatasetLoader::parseTumRgbdDataset(std::string* error) {
  if (!parseTumIntrinsics(error)) {
    return false;
  }

  std::vector<TimedPath> rgb_entries;
  std::vector<TimedPath> depth_entries;

  if (!parseTumListFile((fs::path(dataset_root_) / "rgb.txt").string(), rgb_entries, error)) {
    return false;
  }
  if (!parseTumListFile((fs::path(dataset_root_) / "depth.txt").string(), depth_entries, error)) {
    return false;
  }

  constexpr double kAssociationToleranceS = 0.02;
  size_t depth_cursor = 0;
  int row_index = 0;

  for (const auto& rgb : rgb_entries) {
    while (depth_cursor + 1 < depth_entries.size() &&
           std::abs(depth_entries[depth_cursor + 1].timestamp_s - rgb.timestamp_s) <=
               std::abs(depth_entries[depth_cursor].timestamp_s - rgb.timestamp_s)) {
      ++depth_cursor;
    }

    if (depth_cursor >= depth_entries.size()) {
      break;
    }

    const auto& depth = depth_entries[depth_cursor];
    if (std::abs(depth.timestamp_s - rgb.timestamp_s) > kAssociationToleranceS) {
      continue;
    }

    FramePaths frame;
    frame.index = row_index;
    {
      std::ostringstream id;
      id << std::setw(7) << std::setfill('0') << row_index;
      frame.frame_id = id.str();
    }
    frame.timestamp_s = rgb.timestamp_s;

    const fs::path rgb_path = fs::path(dataset_root_) / rgb.relative_path;
    const fs::path depth_path = fs::path(dataset_root_) / depth.relative_path;

    if (!fs::exists(rgb_path) || !fs::exists(depth_path)) {
      continue;
    }

    frame.rgb_path = rgb_path.string();
    frame.depth_path = depth_path.string();
    frames_.push_back(std::move(frame));
    ++row_index;
    ++depth_cursor;
  }

  if (frames_.empty()) {
    if (error) {
      *error = "Failed to associate RGB and depth frames from TUM dataset.";
    }
    return false;
  }

  return true;
}

bool DatasetLoader::parseCalibration(const std::string& path, std::string* error) {
  std::vector<double> values;
  if (!parseNumericList(path, values, error)) {
    return false;
  }

  if (values.size() < 9) {
    if (error) {
      *error = "Calibration file does not contain a 3x3 matrix: " + path;
    }
    return false;
  }

  intrinsics_.fx = values[0];
  intrinsics_.cx = values[2];
  intrinsics_.fy = values[4];
  intrinsics_.cy = values[5];

  if (!intrinsics_.isValid()) {
    if (error) {
      *error = "Invalid intrinsics parsed from calibration file.";
    }
    return false;
  }

  return true;
}

bool DatasetLoader::parseTumIntrinsics(std::string* error) {
  const std::vector<std::string> candidates = {
      "camera.txt",
      "camera.yaml",
      "calibration.yaml",
      "intrinsics.txt",
  };

  for (const auto& name : candidates) {
    const fs::path path = fs::path(dataset_root_) / name;
    if (!fs::exists(path)) {
      continue;
    }

    std::vector<double> values;
    if (!parseNumericList(path.string(), values, error)) {
      return false;
    }

    if (values.size() >= 9 && std::abs(values[1]) < 1e-9 && std::abs(values[3]) < 1e-9) {
      intrinsics_.fx = values[0];
      intrinsics_.cx = values[2];
      intrinsics_.fy = values[4];
      intrinsics_.cy = values[5];
    } else if (values.size() >= 4) {
      intrinsics_.fx = values[0];
      intrinsics_.fy = values[1];
      intrinsics_.cx = values[2];
      intrinsics_.cy = values[3];
    }

    if (intrinsics_.isValid()) {
      return true;
    }
  }

  const std::string lower = toLower(dataset_root_);
  if (lower.find("freiburg1") != std::string::npos) {
    intrinsics_ = CameraIntrinsics{517.3, 516.5, 318.6, 255.3};
  } else if (lower.find("freiburg2") != std::string::npos) {
    intrinsics_ = CameraIntrinsics{520.9, 521.0, 325.1, 249.7};
  } else if (lower.find("freiburg3") != std::string::npos) {
    intrinsics_ = CameraIntrinsics{535.4, 539.2, 320.1, 247.6};
  } else {
    intrinsics_ = CameraIntrinsics{525.0, 525.0, 319.5, 239.5};
  }

  if (!intrinsics_.isValid()) {
    if (error) {
      *error = "Failed to determine camera intrinsics for TUM dataset.";
    }
    return false;
  }

  return true;
}

bool DatasetLoader::parseTimestamps(const std::string& path, std::string* error) {
  std::ifstream in(path);
  if (!in.is_open()) {
    if (error) {
      *error = "Failed to open timestamp file: " + path;
    }
    return false;
  }

  std::string line;
  if (!std::getline(in, line)) {
    if (error) {
      *error = "Timestamp file is empty: " + path;
    }
    return false;
  }

  int row_index = 0;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string frame_id;
    std::string timestamp_ns_str;

    if (!std::getline(ss, frame_id, ',') || !std::getline(ss, timestamp_ns_str)) {
      if (error) {
        *error = "Malformed timestamp row: " + line;
      }
      return false;
    }

    long long timestamp_ns = 0;
    try {
      timestamp_ns = std::stoll(timestamp_ns_str);
    } catch (const std::exception&) {
      if (error) {
        *error = "Invalid timestamp value: " + timestamp_ns_str;
      }
      return false;
    }

    FramePaths frame;
    frame.index = row_index;
    frame.frame_id = frame_id;
    frame.timestamp_s = static_cast<double>(timestamp_ns) * 1e-9;

    const fs::path rgb_path = fs::path(dataset_root_) / "iphone_mono" / (frame_id + ".png");
    const fs::path depth_path = fs::path(dataset_root_) / "iphone_mono_depth" / (frame_id + ".png");

    if (!fs::exists(rgb_path) || !fs::exists(depth_path)) {
      if (error) {
        *error = "Missing RGB or depth image for frame id " + frame_id;
      }
      return false;
    }

    frame.rgb_path = rgb_path.string();
    frame.depth_path = depth_path.string();
    frames_.push_back(std::move(frame));
    ++row_index;
  }

  return true;
}

bool DatasetLoader::loadFrame(size_t index, FrameData& frame, std::string* error) const {
  if (index >= frames_.size()) {
    if (error) {
      *error = "Frame index out of range.";
    }
    return false;
  }

  frame = {};
  frame.meta = frames_[index];

  frame.rgb = cv::imread(frame.meta.rgb_path, cv::IMREAD_COLOR);
  frame.depth_u16 = cv::imread(frame.meta.depth_path, cv::IMREAD_UNCHANGED);

  if (frame.rgb.empty()) {
    if (error) {
      *error = "Failed to load RGB image: " + frame.meta.rgb_path;
    }
    return false;
  }

  if (frame.depth_u16.empty()) {
    if (error) {
      *error = "Failed to load depth image: " + frame.meta.depth_path;
    }
    return false;
  }

  if (frame.depth_u16.type() != CV_16UC1) {
    if (error) {
      *error = "Depth image is not 16-bit single channel: " + frame.meta.depth_path;
    }
    return false;
  }

  cv::cvtColor(frame.rgb, frame.gray, cv::COLOR_BGR2GRAY);
  return true;
}

}  // namespace cuvslam

