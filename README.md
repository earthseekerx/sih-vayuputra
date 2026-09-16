# VAYUPUTRA — SkyRecon Prototype v2

SIH 2026 • Problem Statement 26158 • Single-Pass Drone Video to Accurate 3D Model Generation System

## What this prototype does

1. Upload a single-pass UAV video.
2. Inspect video metadata (duration, FPS, resolution, frame count).
3. Scan frames for sharpness, exposure and texture quality.
4. Select temporally separated, high-quality keyframes.
5. Run **Depth Anything V2 Small** locally when enabled, or use a lightweight OpenCV fallback.
6. Optionally mask dynamic objects with Ultralytics YOLO when the optional package/model is available.
7. Reproject depth into a colored 3D point cloud.
8. Fuse all selected frames and calculate confidence statistics.
9. Visualize the point cloud interactively.
10. Export PLY + JSON reconstruction report.
11. Optionally export an OBJ mesh when Open3D is installed.

## Run

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## AI depth model

The app uses the Hugging Face checkpoint:

`depth-anything/Depth-Anything-V2-Small-hf`

The first time AI depth is enabled, Transformers will download the model into the local Hugging Face cache. Internet is therefore needed for the first model download unless you have already cached the model.

## Optional upgrades

### Open3D
```bash
pip install open3d
```
Enables point-cloud filtering and Poisson mesh export.

### Ultralytics
```bash
pip install ultralytics
```
Enables the optional YOLO dynamic-object masking stage. The model weights may also be downloaded on first use.

## Important engineering limitation

The demo currently produces a **relative** scene. Metric/georeferenced output is not claimed from video alone. For the production/SIH version, add:

- calibrated camera intrinsics/distortion
- COLMAP / VIO / SLAM camera poses
- GPS + IMU + RTK/PPK fusion
- multi-frame depth consistency and uncertainty propagation
- MVS/TSDF surface reconstruction
- geospatial coordinate conversion and GeoTIFF / 3D Tiles / GLB export

This is intentional: the UI exposes the complete architecture while keeping the MVP runnable on a normal development laptop.
