cmake_minimum_required(VERSION 3.24)
project(cuvslam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CUVSLAM_BUILD_TESTS "Build tests" ON)
option(CUVSLAM_ENABLE_CUDA "Enable CUDA grayscale preprocessing" ON)
option(CUVSLAM_ENABLE_RERUN "Enable Rerun visualization support" OFF)
option(CUVSLAM_REQUIRE_LIBRARY "Fail CMake configure when libcuvslam.so is not found" OFF)
set(CUVSLAM_SDK_ROOT "" CACHE PATH "Path to cuVSLAM SDK root containing lib/libcuvslam.so")
set(CUVSLAM_LIBRARY "" CACHE FILEPATH "Full path to libcuvslam.so")
set(CUVSLAM_AMENT_PREFIX_PATHS "" CACHE STRING
    "Semicolon-separated prefix paths to probe for ament_index resources (optional override)")
set(CUVSLAM_AMENT_RESOURCE_TYPE "isaac_ros_nitros" CACHE STRING
    "Preferred ament resource type used to discover cuVSLAM from ament_index")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 REQUIRED)

if(NOT CUVSLAM_LIBRARY AND CUVSLAM_SDK_ROOT)
  set(CUVSLAM_LIBRARY_CANDIDATE "${CUVSLAM_SDK_ROOT}/lib/libcuvslam.so")
  if(EXISTS "${CUVSLAM_LIBRARY_CANDIDATE}")
    set(CUVSLAM_LIBRARY "${CUVSLAM_LIBRARY_CANDIDATE}" CACHE FILEPATH "Full path to libcuvslam.so" FORCE)
  endif()
endif()

# Optional ROS-free emulation of Isaac ROS ament-index discovery:
# resolve
# share/ament_index/resource_index/<resource_type>/cuvslam
# and derive <prefix>/<relative>/lib/libcuvslam.so.
if(NOT CUVSLAM_LIBRARY)
  set(CUVSLAM_DISCOVERY_SOURCE "")
  set(CUVSLAM_AMENT_PREFIXES ${CUVSLAM_AMENT_PREFIX_PATHS})
  if(DEFINED ENV{AMENT_PREFIX_PATH})
    string(REPLACE ":" ";" CUVSLAM_ENV_AMENT_PREFIXES "$ENV{AMENT_PREFIX_PATH}")
    list(APPEND CUVSLAM_AMENT_PREFIXES ${CUVSLAM_ENV_AMENT_PREFIXES})
  endif()
  if(DEFINED ENV{AMENT_INDEX_PREFIX_PATH})
    string(REPLACE ":" ";" CUVSLAM_ENV_AMENT_INDEX_PREFIXES "$ENV{AMENT_INDEX_PREFIX_PATH}")
    list(APPEND CUVSLAM_AMENT_PREFIXES ${CUVSLAM_ENV_AMENT_INDEX_PREFIXES})
  endif()
  list(REMOVE_DUPLICATES CUVSLAM_AMENT_PREFIXES)

  foreach(AMENT_PREFIX IN LISTS CUVSLAM_AMENT_PREFIXES)
    if(NOT AMENT_PREFIX)
      continue()
    endif()

    set(CUVSLAM_PRIMARY_RESOURCE
        "${AMENT_PREFIX}/share/ament_index/resource_index/${CUVSLAM_AMENT_RESOURCE_TYPE}/cuvslam")
    set(CUVSLAM_RESOURCE_CANDIDATES "${CUVSLAM_PRIMARY_RESOURCE}")
    if(NOT EXISTS "${CUVSLAM_PRIMARY_RESOURCE}")
      file(GLOB CUVSLAM_ANY_RESOURCE
          "${AMENT_PREFIX}/share/ament_index/resource_index/*/cuvslam")
      list(APPEND CUVSLAM_RESOURCE_CANDIDATES ${CUVSLAM_ANY_RESOURCE})
    endif()

    foreach(CUVSLAM_RESOURCE_FILE IN LISTS CUVSLAM_RESOURCE_CANDIDATES)
      if(NOT EXISTS "${CUVSLAM_RESOURCE_FILE}")
        continue()
      endif()

      file(READ "${CUVSLAM_RESOURCE_FILE}" CUVSLAM_RELATIVE_PATH_RAW)
      string(STRIP "${CUVSLAM_RELATIVE_PATH_RAW}" CUVSLAM_RELATIVE_PATH)
      if(NOT CUVSLAM_RELATIVE_PATH)
        continue()
      endif()

      if(IS_ABSOLUTE "${CUVSLAM_RELATIVE_PATH}")
        set(CUVSLAM_AMENT_SDK_ROOT "${CUVSLAM_RELATIVE_PATH}")
      else()
        set(CUVSLAM_AMENT_SDK_ROOT "${AMENT_PREFIX}/${CUVSLAM_RELATIVE_PATH}")
      endif()

      set(CUVSLAM_AMENT_LIBRARY "${CUVSLAM_AMENT_SDK_ROOT}/lib/libcuvslam.so")
      if(EXISTS "${CUVSLAM_AMENT_LIBRARY}")
        set(CUVSLAM_LIBRARY "${CUVSLAM_AMENT_LIBRARY}" CACHE FILEPATH "Full path to libcuvslam.so" FORCE)
        if(NOT CUVSLAM_SDK_ROOT)
          set(CUVSLAM_SDK_ROOT "${CUVSLAM_AMENT_SDK_ROOT}" CACHE PATH
              "Path to cuVSLAM SDK root containing lib/libcuvslam.so" FORCE)
        endif()
        set(CUVSLAM_DISCOVERY_SOURCE "${CUVSLAM_RESOURCE_FILE}")
        break()
      endif()
    endforeach()

    if(CUVSLAM_LIBRARY)
      break()
    endif()
  endforeach()

  if(CUVSLAM_LIBRARY AND CUVSLAM_DISCOVERY_SOURCE)
    message(STATUS "libcuvslam discovered via ament index: ${CUVSLAM_LIBRARY}")
    message(STATUS "ament resource file: ${CUVSLAM_DISCOVERY_SOURCE}")
  endif()
endif()

if(NOT CUVSLAM_LIBRARY)
  set(CUVSLAM_HINT_DIRS "")
  if(CUVSLAM_SDK_ROOT)
    list(APPEND CUVSLAM_HINT_DIRS "${CUVSLAM_SDK_ROOT}/lib")
  endif()
  if(DEFINED ENV{CUVSLAM_LIB_PATH} AND NOT "$ENV{CUVSLAM_LIB_PATH}" STREQUAL "")
    if(IS_DIRECTORY "$ENV{CUVSLAM_LIB_PATH}")
      list(APPEND CUVSLAM_HINT_DIRS "$ENV{CUVSLAM_LIB_PATH}")
    elseif(EXISTS "$ENV{CUVSLAM_LIB_PATH}")
      set(CUVSLAM_LIBRARY "$ENV{CUVSLAM_LIB_PATH}" CACHE FILEPATH "Full path to libcuvslam.so" FORCE)
    endif()
  endif()
  if(DEFINED ENV{CUVSLAM_LIBRARY_PATH} AND NOT "$ENV{CUVSLAM_LIBRARY_PATH}" STREQUAL "")
    if(IS_DIRECTORY "$ENV{CUVSLAM_LIBRARY_PATH}")
      list(APPEND CUVSLAM_HINT_DIRS "$ENV{CUVSLAM_LIBRARY_PATH}")
    elseif(EXISTS "$ENV{CUVSLAM_LIBRARY_PATH}")
      set(CUVSLAM_LIBRARY "$ENV{CUVSLAM_LIBRARY_PATH}" CACHE FILEPATH "Full path to libcuvslam.so" FORCE)
    endif()
  endif()

  if(NOT CUVSLAM_LIBRARY)
    find_library(CUVSLAM_LIBRARY
      NAMES libcuvslam.so libcuvslam cuvslam
      HINTS ${CUVSLAM_HINT_DIRS}
    )
  endif()
endif()

set(CUVSLAM_MISSING_LIBRARY_HELP [[
libcuvslam.so not found at configure time.
Provide one of:
  -DCUVSLAM_LIBRARY=/absolute/path/to/libcuvslam.so
  -DCUVSLAM_SDK_ROOT=/absolute/path/to/cuvslam_sdk_root

For runtime-only loading, you can also use:
  --libcuvslam_path /absolute/path/to/libcuvslam.so
  CUVSLAM_LIB_PATH or CUVSLAM_LIBRARY_PATH environment variable

For Isaac-style discovery, ensure AMENT_PREFIX_PATH or AMENT_INDEX_PREFIX_PATH
contains a prefix with share/ament_index/resource_index/<type>/cuvslam.
]])

if(CUVSLAM_LIBRARY)
  message(STATUS "libcuvslam detected at: ${CUVSLAM_LIBRARY}")
elseif(CUVSLAM_REQUIRE_LIBRARY)
  message(FATAL_ERROR "${CUVSLAM_MISSING_LIBRARY_HELP}")
else()
  message(WARNING "${CUVSLAM_MISSING_LIBRARY_HELP}")
endif()

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
  src/image_processing.cpp
  src/libcuvslam_backend.cpp
  src/evaluation.cpp
  src/pipeline.cpp
)

target_include_directories(cuvslam PUBLIC include)
target_link_libraries(cuvslam
  PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

if(CUVSLAM_LIBRARY)
  target_compile_definitions(cuvslam PUBLIC CUVSLAM_DEFAULT_LIBRARY_PATH="${CUVSLAM_LIBRARY}")
endif()

if(CMAKE_DL_LIBS)
  target_link_libraries(cuvslam PUBLIC $<$<LINK_LANGUAGE:CXX>:${CMAKE_DL_LIBS}>)
endif()

target_compile_options(cuvslam PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)

if(CUVSLAM_WITH_CUDA)
  target_sources(cuvslam PRIVATE src/cuda/image_kernels.cu)
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
  if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
    set(CUVSLAM_TEST_ENV "LD_LIBRARY_PATH=/usr/lib/wsl/lib:$ENV{LD_LIBRARY_PATH}")
    set_tests_properties(cuvslam_unit_tests cuvslam_integration_tests PROPERTIES
      ENVIRONMENT "${CUVSLAM_TEST_ENV}"
    )
  endif()
endif()
