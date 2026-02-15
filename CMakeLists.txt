cmake_minimum_required(VERSION 3.24)
project(cuvslam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CUVSLAM_BUILD_TESTS "Build tests" ON)
option(CUVSLAM_ENABLE_CUDA "Enable CUDA depth preprocessing" ON)
option(CUVSLAM_ENABLE_RERUN "Enable Rerun visualization support" OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d)
find_package(Eigen3 REQUIRED)

if(CUVSLAM_ENABLE_RERUN)
  include(FetchContent)
  FetchContent_Declare(
    rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
  )
  FetchContent_MakeAvailable(rerun_sdk)
endif()

include(CheckLanguage)
if(CUVSLAM_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUVSLAM_WITH_CUDA ON)
    find_package(CUDAToolkit REQUIRED)
  else()
    message(WARNING "CUDA compiler not found. Building CPU-only pipeline.")
    set(CUVSLAM_WITH_CUDA OFF)
  endif()
else()
  set(CUVSLAM_WITH_CUDA OFF)
endif()

add_library(cuvslam STATIC
  src/dataset_loader.cpp
  src/depth_processing.cpp
  src/pose_estimator.cpp
  src/evaluation.cpp
  src/pipeline.cpp
)

target_include_directories(cuvslam PUBLIC include)
target_link_libraries(cuvslam
  PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

target_compile_options(cuvslam PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

if(CUVSLAM_WITH_CUDA)
  target_sources(cuvslam PRIVATE src/cuda/depth_kernels.cu)
  target_compile_definitions(cuvslam PUBLIC CUVSLAM_WITH_CUDA=1)
  target_link_libraries(cuvslam PUBLIC CUDA::cudart)
  set_target_properties(cuvslam PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

if(CUVSLAM_ENABLE_RERUN)
  target_compile_definitions(cuvslam PUBLIC CUVSLAM_WITH_RERUN=1)
  target_link_libraries(cuvslam PUBLIC rerun_sdk)
endif()

add_executable(cuvslam_cli apps/cuvslam_cli.cpp)
target_link_libraries(cuvslam_cli PRIVATE cuvslam)

if(CUVSLAM_BUILD_TESTS)
  include(CTest)
  add_executable(cuvslam_unit_tests tests/unit_tests.cpp)
  target_link_libraries(cuvslam_unit_tests PRIVATE cuvslam)
  target_compile_definitions(cuvslam_unit_tests PRIVATE CUVSLAM_TEST_DATASET_ROOT="${CMAKE_SOURCE_DIR}/data_sample")

  add_executable(cuvslam_integration_tests tests/integration_tests.cpp)
  target_link_libraries(cuvslam_integration_tests PRIVATE cuvslam)
  target_compile_definitions(cuvslam_integration_tests PRIVATE CUVSLAM_TEST_DATASET_ROOT="${CMAKE_SOURCE_DIR}/data_sample")

  add_test(NAME cuvslam_unit_tests COMMAND cuvslam_unit_tests)
  add_test(NAME cuvslam_integration_tests COMMAND cuvslam_integration_tests)
endif()
