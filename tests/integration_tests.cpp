#include "test_harness.hpp"

#include "cuvslam/pipeline.hpp"

#include <filesystem>

namespace {

bool testPipelineOnSampleDataset() {
  const std::filesystem::path output_dir = std::filesystem::path("/tmp") / "cuvslam_integration_output";
  std::filesystem::remove_all(output_dir);

  cuvslam::PipelineOptions options;
  options.dataset_root = CUVSLAM_TEST_DATASET_ROOT;
  options.output_dir = output_dir.string();
  options.max_frames = 300;
  options.use_cuda = false;
  options.depth_scale_m_per_unit = 0.001f;

  const cuvslam::PipelineResult result = cuvslam::runPipeline(options);
  if (!result.success) {
    const bool backend_unavailable =
        result.message.find("only supported on Linux") != std::string::npos ||
        result.message.find("Unable to load libcuvslam.so") != std::string::npos;
    if (backend_unavailable) {
      return true;
    }
  }

  TEST_EXPECT_TRUE(result.success);
  TEST_EXPECT_TRUE(result.summary.total_frames == 300U);
  TEST_EXPECT_TRUE(result.summary.total_time_ms > 0.0);
  TEST_EXPECT_TRUE(result.summary.average_fps > 1.0);

  const double tracked_ratio = result.summary.total_frames > 1
                                   ? static_cast<double>(result.summary.tracked_frames) /
                                         static_cast<double>(result.summary.total_frames - 1)
                                   : 0.0;
  TEST_EXPECT_TRUE(tracked_ratio > 0.85);

  TEST_EXPECT_TRUE(std::filesystem::exists(result.trajectory_path));
  TEST_EXPECT_TRUE(std::filesystem::exists(result.frame_metrics_path));
  TEST_EXPECT_TRUE(std::filesystem::exists(result.report_path));

  if (result.evaluation.has_reference) {
    TEST_EXPECT_TRUE(result.evaluation.matched_samples > 250U);
    TEST_EXPECT_TRUE(result.evaluation.ate_rmse_m < 1.5);
    TEST_EXPECT_TRUE(result.evaluation.rpe_rmse_m < 0.08);
  }

  return true;
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<bool()>>> tests = {
      {"Pipeline integration on data_sample", testPipelineOnSampleDataset},
  };

  return runTests(tests);
}
