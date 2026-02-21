# Performance Baseline

Environment:

- Date: 2026-02-15
- OS: Ubuntu on WSL2
- CPU threads: 16
- GPU: NVIDIA RTX 3060
- Build type: `Release`

## A. Custom Dataset (`data_sample`)

Dataset format: `custom_iphone`

Reference: pass with `--reference_tum <path>` when available

Run command:

```bash
./build/cuvslam_cli --dataset_root data_sample --dataset_format custom --output_dir outputs/full_run_optimized --no_cuda
```

Results (`outputs/full_run_optimized/performance_report.md`):

- Frames: `1247`
- Tracked frames: `1236`
- ATE RMSE: `0.781992 m`
- RPE RMSE: `0.0123045 m`
- FPS: `19.6806`

## B. TUM RGB-D (`rgbd_dataset_freiburg1_xyz`)

Dataset format: `tum_rgbd`

Ground truth: `groundtruth.txt`

Run command:

```bash
./build/cuvslam_cli \
  --dataset_root third_party_datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz \
  --dataset_format tum \
  --reference_tum third_party_datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  --output_dir outputs/tum_freiburg1_xyz \
  --no_cuda
```

Results (`outputs/tum_freiburg1_xyz/performance_report.md`):

- Frames: `794`
- Tracked frames: `793`
- ATE RMSE: `0.0532896 m`
- RPE RMSE: `0.00625302 m`
- FPS: `17.5903`

## C. TUM + Rerun Recording Run (300 frames)

Run command:

```bash
./scripts/run_tum_eval_with_rerun.sh \
  third_party_datasets/tum_rgbd/rgbd_dataset_freiburg1_xyz \
  outputs/tum_freiburg1_xyz_rerun \
  300
```

Results (`outputs/tum_freiburg1_xyz_rerun/performance_report.md`):

- Frames: `300`
- Tracked frames: `299`
- ATE RMSE: `0.0273335 m`
- RPE RMSE: `0.00616864 m`
- FPS: `16.7471`
- Rerun output: `outputs/tum_freiburg1_xyz_rerun/cuvslam.rrd`

## Notes

- The new estimator significantly reduced drift on `data_sample` compared to the previous baseline.
- TUM results are substantially better because sequence calibration and depth scale are well-defined.
- Enabling Rerun introduces additional logging overhead; this is expected.
