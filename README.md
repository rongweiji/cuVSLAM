# cuVSLAM (C++ / CUDA, Optional Rerun Visualization)

`cuVSLAM` is a Linux-first C++17 RGB-D visual SLAM/VO implementation that runs on WSL/Linux with optional CUDA acceleration and optional real-time Rerun visualization.

No Python runtime is required by the core SLAM pipeline.

## Implemented

- C++17 end-to-end RGB-D tracking pipeline:
  - dataset loading
  - grayscale preprocessing (CPU or CUDA)
  - native `libcuvslam` frame tracking
  - trajectory integration + export
  - accuracy evaluation against ground truth/reference
  - per-stage timing report
- Dataset support:
  - `custom_iphone` format used by `data_sample`
  - `tum_rgbd` format (`rgb.txt`, `depth.txt`, `groundtruth.txt`)
- Optional Rerun visualization integration:
  - live stream to viewer (spawn)
  - save `.rrd` for offline replay
  - camera pose + trajectory + RGB + depth logging
- Native NVIDIA cuVSLAM backend (`libcuvslam.so`) via runtime loading

## Project Layout

- `apps/cuvslam_cli.cpp`: CLI entry point
- `include/cuvslam/*.hpp`: public interfaces
- `src/*.cpp`: core implementation
- `src/cuda/image_kernels.cu`: CUDA grayscale conversion
- `tests/*.cpp`: unit + integration tests
- `scripts/check_cuvslam.sh`: validate `libcuvslam.so` + runtime dependencies
- `scripts/run_data_sample.sh`: build + run `data_sample` with strict dependency checks
- `scripts/run_data_sample_with_rerun.sh`: build with Rerun + run `data_sample` in live GUI mode
- `scripts/download_tum_rgbd.sh`: download TUM RGB-D sequences
- `scripts/install_rerun_cli.sh`: install local Rerun CLI binary
- `scripts/run_tum_eval_with_rerun.sh`: build with Rerun + run/evaluate + save `.rrd`

## Prepare cuVSLAM SDK (non-ROS)

This project needs NVIDIA `libcuvslam.so` on your machine. A practical local layout is:

```text
third_party/cuvslam_sdk/
├── include/cuvslam/...
└── lib/libcuvslam.so
```

`third_party/cuvslam_sdk/` is ignored by Git (`.gitignore`) so binaries do not enter your repository history.

Validate your library before build:

```bash
./scripts/check_cuvslam.sh third_party/cuvslam_sdk
```

## Build

Default build (without Rerun):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUVSLAM_REQUIRE_LIBRARY=ON \
  -DCUVSLAM_SDK_ROOT=$PWD/third_party/cuvslam_sdk
cmake --build build -j$(nproc)
```

If you prefer direct library path:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUVSLAM_REQUIRE_LIBRARY=ON \
  -DCUVSLAM_LIBRARY=/absolute/path/to/libcuvslam.so
```

Build with Rerun support:

```bash
cmake -S . -B build_rerun -DCMAKE_BUILD_TYPE=Release -DCUVSLAM_ENABLE_RERUN=ON \
  -DCUVSLAM_LIBRARY=/absolute/path/to/libcuvslam.so
cmake --build build_rerun -j$(nproc)
```

## Tests

```bash
cd build
ctest --output-on-failure
```

(Also passes in `build_rerun`.)

## CLI Usage

```bash
./build/cuvslam_cli [options]
```

Key options:

- `--dataset_root <path>`
- `--dataset_format auto|custom|tum`
- `--libcuvslam_path <path>`
- `--libcuvslam_verbosity <N>`
- `--reference_tum <path>`
- `--max_frames <N>`
- `--depth_scale <meters_per_unit>` (default auto by dataset)
- `--fx --fy --cx --cy` (override intrinsics)
- `--no_cuda`
- `--enable_rerun`
- `--rerun_spawn`
- `--rerun_save <file.rrd>`
- `--rerun_log_every_n <N>`
- `--realtime`
- `--realtime_speed <value>`

## `libcuvslam.so` Dependency

This project does not require ROS, but it does require NVIDIA `libcuvslam.so` to be available from one of:

- Isaac-style ament index discovery:
  - searches prefixes from `AMENT_PREFIX_PATH` / `AMENT_INDEX_PREFIX_PATH` (or `-DCUVSLAM_AMENT_PREFIX_PATHS=...`)
  - reads `share/ament_index/resource_index/<type>/cuvslam` (preferred type: `isaac_ros_nitros`, configurable via `-DCUVSLAM_AMENT_RESOURCE_TYPE=...`)
- CMake configure detection (`-DCUVSLAM_LIBRARY=/path/to/libcuvslam.so` or `-DCUVSLAM_SDK_ROOT=/path/to/sdk`)
- CLI override: `--libcuvslam_path /path/to/libcuvslam.so`
- Env var: `CUVSLAM_LIB_PATH` or `CUVSLAM_LIBRARY_PATH`
- System linker path (`libcuvslam.so`)

For WSL, if you hit CUDA driver version mismatch errors, prefer NVIDIA's WSL driver loader path:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
```

(`scripts/run_data_sample.sh` already applies this automatically when that path exists.)

## Dataset Download (TUM RGB-D)

```bash
./scripts/download_tum_rgbd.sh rgbd_dataset_freiburg1_xyz third_party_datasets/tum_rgbd
```

Dataset path after download:

- `third_party_datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz`

## Run Examples

One-command local smoke run (`data_sample`):

```bash
./scripts/run_data_sample.sh outputs/data_sample third_party/cuvslam_sdk
```

Real-time sample GUI run with Rerun C++ SDK (spawns viewer + saves `.rrd`):

```bash
./scripts/run_data_sample_with_rerun.sh outputs/data_sample_rerun third_party/cuvslam_sdk 1.0
```

Run on provided `data_sample`:

```bash
./build/cuvslam_cli \
  --dataset_root data_sample \
  --dataset_format custom \
  --output_dir outputs/full_run_optimized \
  --no_cuda
```

Run with explicit `libcuvslam` library path:

```bash
./build/cuvslam_cli \
  --dataset_root data_sample \
  --dataset_format custom \
  --libcuvslam_path /path/to/libcuvslam.so \
  --output_dir outputs/full_run_libcuvslam
```

Run on TUM RGB-D with ground truth evaluation:

```bash
./build/cuvslam_cli \
  --dataset_root third_party_datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz \
  --dataset_format tum \
  --reference_tum third_party_datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  --output_dir outputs/tum_freiburg1_xyz \
  --no_cuda
```

One-shot Rerun run (build + run + save `.rrd`):

```bash
./scripts/run_tum_eval_with_rerun.sh \
  third_party_datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz \
  outputs/tum_freiburg1_xyz_rerun \
  300
```

Open recorded `.rrd`:

```bash
./.tools/rerun/rerun outputs/tum_freiburg1_xyz_rerun/cuvslam.rrd
```

## Output Artifacts

Each run produces:

- `estimated_trajectory.tum`
- `frame_metrics.csv`
- `performance_report.md`

With Rerun enabled and `--rerun_save`:

- `cuvslam.rrd`

## Current Measured Results

### `data_sample` (custom_iphone, 1247 frames)

From `outputs/full_run_optimized/performance_report.md`:

- Tracked frames: `1236 / 1247`
- FPS: `19.681`
- ATE RMSE: `0.7820 m`
- RPE RMSE: `0.0123 m`

### TUM RGB-D `rgbd_dataset_freiburg1_xyz` (794 frames)

From `outputs/tum_freiburg1_xyz/performance_report.md`:

- Tracked frames: `793 / 794`
- FPS: `17.590`
- ATE RMSE: `0.0533 m`
- RPE RMSE: `0.0063 m`

### TUM + Rerun recording run (300 frames)

From `outputs/tum_freiburg1_xyz_rerun/performance_report.md`:

- Tracked frames: `299 / 300`
- ATE RMSE: `0.0273 m`
- RPE RMSE: `0.0062 m`
- Rerun file: `outputs/tum_freiburg1_xyz_rerun/cuvslam.rrd`
