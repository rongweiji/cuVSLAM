#include "test_harness.hpp"

#include "cuvslam/dataset_loader.hpp"
#include "cuvslam/evaluation.hpp"
#include "cuvslam/image_processing.hpp"
#include "cuvslam/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>

namespace {

bool testPoseComposeAndInverse() {
  cuvslam::Pose a;
  a.R = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX()).toRotationMatrix();
  a.t = Eigen::Vector3d(1.0, -2.0, 0.5);

  cuvslam::Pose b;
  b.R = Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  b.t = Eigen::Vector3d(-0.3, 0.2, 1.3);

  const cuvslam::Pose c = a * b;
  const cuvslam::Pose recovered_b = a.inverse() * c;

  TEST_EXPECT_NEAR((recovered_b.R - b.R).norm(), 0.0, 1e-9);
  TEST_EXPECT_NEAR((recovered_b.t - b.t).norm(), 0.0, 1e-9);
  return true;
}

bool testDatasetMetadataParsing() {
  cuvslam::DatasetLoader loader(CUVSLAM_TEST_DATASET_ROOT);
  std::string error;
  TEST_EXPECT_TRUE(loader.loadMetadata(&error));

  TEST_EXPECT_TRUE(loader.intrinsics().isValid());
  TEST_EXPECT_NEAR(loader.intrinsics().fx, 482.316, 1e-3);
  TEST_EXPECT_NEAR(loader.intrinsics().fy, 482.316, 1e-3);
  TEST_EXPECT_TRUE(loader.frames().size() == 1247U);

  cuvslam::FrameData frame;
  TEST_EXPECT_TRUE(loader.loadFrame(0, frame, &error));
  TEST_EXPECT_TRUE(!frame.rgb.empty());
  TEST_EXPECT_TRUE(!frame.depth_u16.empty());
  TEST_EXPECT_TRUE(frame.gray.empty());
  TEST_EXPECT_TRUE(frame.depth_u16.type() == CV_16UC1);

  return true;
}

bool testDatasetFrameGrayOnlyLoad() {
  cuvslam::DatasetLoader loader(CUVSLAM_TEST_DATASET_ROOT);
  std::string error;
  TEST_EXPECT_TRUE(loader.loadMetadata(&error));

  cuvslam::FrameData frame;
  TEST_EXPECT_TRUE(loader.loadFrame(0, frame, false, &error));
  TEST_EXPECT_TRUE(frame.rgb.empty());
  TEST_EXPECT_TRUE(frame.gray.type() == CV_8UC1);
  TEST_EXPECT_TRUE(frame.depth_u16.type() == CV_16UC1);
  return true;
}

bool testImageProcessorCpuPath() {
  cuvslam::ImageProcessor image_processor(false);

  cv::Mat bgr(1, 4, CV_8UC3);
  bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 255);
  bgr.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);
  bgr.at<cv::Vec3b>(0, 2) = cv::Vec3b(255, 0, 0);
  bgr.at<cv::Vec3b>(0, 3) = cv::Vec3b(255, 255, 255);

  cv::Mat gray;
  TEST_EXPECT_TRUE(image_processor.convertBgrToGray(bgr, gray));
  TEST_EXPECT_TRUE(gray.type() == CV_8UC1);
  TEST_EXPECT_TRUE(gray.rows == 1 && gray.cols == 4);
  TEST_EXPECT_TRUE(gray.at<uint8_t>(0, 0) == static_cast<uint8_t>(76));
  TEST_EXPECT_TRUE(gray.at<uint8_t>(0, 1) == static_cast<uint8_t>(150));
  TEST_EXPECT_TRUE(gray.at<uint8_t>(0, 2) == static_cast<uint8_t>(29));
  TEST_EXPECT_TRUE(gray.at<uint8_t>(0, 3) == static_cast<uint8_t>(255));

  return true;
}

bool testImageProcessorCudaParity() {
#ifdef CUVSLAM_WITH_CUDA
  cuvslam::ImageProcessor cpu_processor(false);
  cuvslam::ImageProcessor gpu_processor(true);

  cv::Mat bgr(64, 96, CV_8UC3);
  for (int r = 0; r < bgr.rows; ++r) {
    for (int c = 0; c < bgr.cols; ++c) {
      bgr.at<cv::Vec3b>(r, c) = cv::Vec3b(static_cast<uint8_t>((r + c) % 256),
                                          static_cast<uint8_t>((r * 3 + c * 5) % 256),
                                          static_cast<uint8_t>((r * 7 + c * 11) % 256));
    }
  }

  cv::Mat gray_cpu;
  cv::Mat gray_gpu;
  TEST_EXPECT_TRUE(cpu_processor.convertBgrToGray(bgr, gray_cpu));
  TEST_EXPECT_TRUE(gpu_processor.convertBgrToGray(bgr, gray_gpu));

  TEST_EXPECT_TRUE(gray_cpu.size() == gray_gpu.size());
  TEST_EXPECT_TRUE(gray_cpu.type() == gray_gpu.type());

  cv::Mat diff;
  cv::absdiff(gray_cpu, gray_gpu, diff);
  double max_diff = 0.0;
  cv::minMaxLoc(diff, nullptr, &max_diff);
  TEST_EXPECT_TRUE(max_diff <= 1.0);
#endif
  return true;
}

bool testTumReaderAndEvaluation() {
  std::string error;
  const std::filesystem::path dataset_root = std::filesystem::path(CUVSLAM_TEST_DATASET_ROOT);
  std::filesystem::path tum_path = dataset_root / "groundtruth.txt";
  if (!std::filesystem::exists(tum_path)) {
    for (const auto& entry : std::filesystem::directory_iterator(dataset_root)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      if (entry.path().extension() == ".tum") {
        tum_path = entry.path();
        break;
      }
    }
  }
  TEST_EXPECT_TRUE(std::filesystem::exists(tum_path));

  const auto reference = cuvslam::readTumTrajectory(tum_path.string(), &error);
  TEST_EXPECT_TRUE(error.empty());
  TEST_EXPECT_TRUE(!reference.empty());

  const auto metrics = cuvslam::evaluateTrajectory(reference, reference, 1e-6);
  TEST_EXPECT_TRUE(metrics.has_reference);
  TEST_EXPECT_TRUE(metrics.matched_samples == reference.size());
  TEST_EXPECT_NEAR(metrics.ate_rmse_m, 0.0, 1e-9);
  TEST_EXPECT_NEAR(metrics.rpe_rmse_m, 0.0, 1e-9);

  return true;
}

bool testTumDatasetParsingSynthetic() {
  const std::filesystem::path root = std::filesystem::path("/tmp") / "cuvslam_tum_unit";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "rgb");
  std::filesystem::create_directories(root / "depth");

  {
    std::ofstream camera(root / "camera.txt");
    camera << "525.0 525.0 319.5 239.5\n";
  }

  {
    std::ofstream rgb_list(root / "rgb.txt");
    rgb_list << "# timestamp filename\n";
    for (int i = 0; i < 3; ++i) {
      const double t = 1.0 + 0.033 * static_cast<double>(i);
      const char* file = (i == 0 ? "rgb/0000.png" : (i == 1 ? "rgb/0001.png" : "rgb/0002.png"));
      rgb_list << t << " " << file << "\n";
    }
  }

  {
    std::ofstream depth_list(root / "depth.txt");
    depth_list << "# timestamp filename\n";
    for (int i = 0; i < 3; ++i) {
      const double t = 1.0 + 0.033 * static_cast<double>(i) + 0.005;
      const char* file = (i == 0 ? "depth/0000.png" : (i == 1 ? "depth/0001.png" : "depth/0002.png"));
      depth_list << t << " " << file << "\n";
    }
  }

  for (int i = 0; i < 3; ++i) {
    cv::Mat rgb(480, 640, CV_8UC3, cv::Scalar(10 + i, 20 + i, 30 + i));
    cv::Mat depth(480, 640, CV_16UC1, cv::Scalar(1500 + i));

    const std::string idx = (i == 0 ? "0000" : (i == 1 ? "0001" : "0002"));
    cv::imwrite((root / "rgb" / (idx + ".png")).string(), rgb);
    cv::imwrite((root / "depth" / (idx + ".png")).string(), depth);
  }

  cuvslam::DatasetLoader loader(root.string(), cuvslam::DatasetFormat::kTumRgbd);
  std::string error;
  TEST_EXPECT_TRUE(loader.loadMetadata(&error));
  TEST_EXPECT_TRUE(loader.datasetFormat() == cuvslam::DatasetFormat::kTumRgbd);
  TEST_EXPECT_TRUE(loader.frames().size() == 3U);
  TEST_EXPECT_NEAR(loader.intrinsics().fx, 525.0, 1e-6);
  TEST_EXPECT_NEAR(loader.recommendedDepthScaleMPerUnit(), 1.0 / 5000.0, 1e-9);

  cuvslam::FrameData frame;
  TEST_EXPECT_TRUE(loader.loadFrame(0, frame, &error));
  TEST_EXPECT_TRUE(frame.rgb.type() == CV_8UC3);
  TEST_EXPECT_TRUE(frame.depth_u16.type() == CV_16UC1);
  return true;
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<bool()>>> tests = {
      {"Pose compose/inverse", testPoseComposeAndInverse},
      {"Dataset metadata parsing", testDatasetMetadataParsing},
      {"Dataset gray-only load", testDatasetFrameGrayOnlyLoad},
      {"Image processor CPU path", testImageProcessorCpuPath},
      {"Image processor CUDA parity", testImageProcessorCudaParity},
      {"TUM reader and evaluation", testTumReaderAndEvaluation},
      {"TUM dataset parsing (synthetic)", testTumDatasetParsingSynthetic},
  };

  return runTests(tests);
}
