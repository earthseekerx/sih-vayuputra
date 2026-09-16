import io
import json
import math
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import open3d as o3d
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

APP_VERSION = "2.0.0"
PROJECT = "VAYUPUTRA / SkyRecon"
PS_ID = "26158"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

st.set_page_config(page_title="VAYUPUTRA | SkyRecon AI Reconstruction", page_icon="✈️", layout="wide")

# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; max-width: 1500px;}
    .hero {padding: 20px 24px; border: 1px solid rgba(128,128,128,.22); border-radius: 18px; background: linear-gradient(135deg, rgba(70,90,130,.16), rgba(40,45,65,.08));}
    .hero h1 {margin: 0 0 6px 0; font-size: 2.25rem;}
    .hero p {margin: 0; color: #8d96a5;}
    .pill {display:inline-block; padding:4px 10px; border-radius:99px; border:1px solid rgba(128,128,128,.25); margin-right:6px; font-size:.82rem;}
    .stage {font-size:.9rem; color:#9aa3b2; text-transform:uppercase; letter-spacing:.08em;}
    .small {font-size:.84rem; color:#8d96a5;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class='hero'>
      <div class='stage'>SIH 2026 • PS {PS_ID} • Robotics & Drones</div>
      <h1>VAYUPUTRA — SkyRecon AI Geospatial Reconstruction Engine</h1>
      <p>Single-pass UAV video → quality-aware keyframes → AI depth → pose estimation → confidence-aware 3D point cloud → digital-twin export</p>
      <div style='margin-top:12px'>
        <span class='pill'>v{APP_VERSION}</span><span class='pill'>Depth Anything V2</span><span class='pill'>OpenCV</span><span class='pill'>Open3D-ready</span><span class='pill'>YOLO-ready</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Utilities -----------------------------

def get_video_info(path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"fps": 30.0, "frames": 0, "duration": 0.0, "width": 0, "height": 0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "frames": frames, "duration": frames / fps if fps else 0, "width": width, "height": height}


@st.cache_data(show_spinner=False)
def save_upload(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(path).write_bytes(data)
    return path


def sharpness_score(frame: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap, float(min(1.0, lap / 250.0))


def exposure_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = float(gray.mean())
    clipped = float(((gray < 8) | (gray > 247)).mean())
    centered = max(0.0, 1.0 - abs(mean - 128.0) / 128.0)
    return float(max(0.0, centered * (1.0 - min(0.7, clipped * 2.0))))


def score_frame(frame: np.ndarray) -> Dict[str, float]:
    lap, sh = sharpness_score(frame)
    ex = exposure_score(frame)
    # Edge density helps reject flat frames with little geometry.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(np.mean(edges > 0))
    texture = float(min(1.0, edge_density / 0.12))
    quality = 0.5 * sh + 0.3 * ex + 0.2 * texture
    return {"sharpness": lap, "sharpness_score": sh, "exposure_score": ex, "texture_score": texture, "quality": quality}


@st.cache_data(show_spinner=False)
def select_keyframes(path: str, sample_every: int, max_keyframes: int, min_gap: int) -> Tuple[List[dict], List[dict]]:
    cap = cv2.VideoCapture(path)
    candidates = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            s = score_frame(frame)
            # Store compressed preview to keep Streamlit cache small.
            thumb = cv2.resize(frame, (min(640, frame.shape[1]), int(frame.shape[0] * min(640, frame.shape[1]) / frame.shape[1])))
            ok_jpg, enc = cv2.imencode('.jpg', thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok_jpg:
                s.update({"frame": idx, "preview": enc.tobytes()})
                candidates.append(s)
        idx += 1
    cap.release()
    if not candidates:
        return [], []

    # Greedy quality + temporal spacing selection.
    ranked = sorted(candidates, key=lambda x: x["quality"], reverse=True)
    selected = []
    for c in ranked:
        if all(abs(c["frame"] - x["frame"]) >= min_gap for x in selected):
            selected.append(c)
        if len(selected) >= max_keyframes:
            break
    selected.sort(key=lambda x: x["frame"])
    return selected, sorted(candidates, key=lambda x: x["frame"])


def read_frame(path: str, frame_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


@st.cache_resource(show_spinner=False)
def load_depth_model():
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
        model = model.to(device)
        model.eval()
        return {"ok": True, "torch": torch, "processor": processor, "model": model, "device": device}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def depth_anything(frame_bgr: np.ndarray, model_bundle: dict) -> np.ndarray:
    torch = model_bundle["torch"]
    processor = model_bundle["processor"]
    model = model_bundle["model"]
    device = model_bundle["device"]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    post = processor.post_process_depth_estimation(outputs, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])])
    depth = post[0]["predicted_depth"]
    d = depth.detach().float().cpu().numpy()
    d = (d - np.percentile(d, 2)) / (np.percentile(d, 98) - np.percentile(d, 2) + 1e-6)
    return np.clip(d, 0, 1).astype(np.float32)


def depth_fallback(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (0, 0), 3)
    local = np.abs(gray - smooth)
    d = 0.7 * (1.0 - gray) + 0.3 * (1.0 - np.clip(local * 4, 0, 1))
    d = cv2.GaussianBlur(d, (0, 0), 2)
    return cv2.normalize(d, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)


def try_yolo_mask(frame: np.ndarray) -> Tuple[np.ndarray, str]:
    """Optional dynamic-object mask. Returns boolean static mask + status."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        result = model.predict(frame, verbose=False, conf=0.35, imgsz=640)[0]
        # COCO-ish dynamic classes: person, bicycle, car, motorcycle, bus, truck, boat, animal.
        dynamic = {0, 1, 2, 3, 5, 7, 8, 14, 15, 16, 17, 18, 19, 20}
        mask = np.ones(frame.shape[:2], dtype=bool)
        if result.boxes is not None:
            for cls, box in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()):
                if int(cls) in dynamic:
                    x1, y1, x2, y2 = map(int, box)
                    mask[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)] = False
        return mask, "YOLO dynamic-object mask"
    except Exception as e:
        # Classical motion/foreground fallback for a single frame is intentionally conservative.
        return np.ones(frame.shape[:2], dtype=bool), "Dynamic masking unavailable (optional Ultralytics dependency/model not loaded)"


def reprojection_points(frame: np.ndarray, depth: np.ndarray, confidence: float, pixel_stride: int, depth_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    ys = np.arange(0, h, pixel_stride)
    xs = np.arange(0, w, pixel_stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dd = depth[::pixel_stride, ::pixel_stride]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[::pixel_stride, ::pixel_stride]

    # Normalized pinhole camera model. Camera calibration/RTK should replace this for metric output.
    fx = fy = 0.9 * w
    cx, cy = w / 2.0, h / 2.0
    z = (0.35 + dd * 1.65) * depth_scale
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    mask = np.isfinite(z) & (z > 0.05)
    p = np.column_stack((x[mask], y[mask], z[mask])).astype(np.float32)
    c = rgb[mask].astype(np.uint8)
    conf = np.full((len(p),), confidence, dtype=np.float32)
    return p, c, conf


def fuse_cloud(clouds: List[np.ndarray], colors: List[np.ndarray], confs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P = np.vstack(clouds)
    C = np.vstack(colors)
    F = np.concatenate(confs)
    # Robust normalization for a stable demo view.
    P = P - np.median(P, axis=0, keepdims=True)
    radius = np.percentile(np.linalg.norm(P, axis=1), 98) + 1e-6
    P = P / radius * 10.0
    return P, C, F


def make_ply(points: np.ndarray, colors: np.ndarray) -> bytes:
    out = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue", "end_header"
    ]
    out.extend(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}" for p, c in zip(points, colors))
    return "\n".join(out).encode("utf-8")


def try_open3d_mesh(points: np.ndarray, colors: np.ndarray) -> Tuple[Optional[bytes], str]:
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        pc = pc.voxel_down_sample(voxel_size=max(0.05, np.ptp(points[:, 0]) / 180.0))
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=18, std_ratio=2.0)
        if len(pc.points) < 50:
            return None, "Not enough points for meshing"
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pc.orient_normals_consistent_tangent_plane(k=15)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=7)
        mesh.compute_vertex_normals()
        # Write temporary OBJ for browser/download.
        fd, path = tempfile.mkstemp(suffix=".obj")
        os.close(fd)
        o3d.io.write_triangle_mesh(path, mesh, write_vertex_normals=True)
        data = Path(path).read_bytes()
        try:
            os.remove(path)
        except OSError:
            pass
        return data, "Open3D Poisson mesh"
    except Exception as e:
        return None, f"Mesh unavailable: {e}"

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("Reconstruction controls")
    quality_tab = st.slider("Sample every N frames", 1, 60, 10)
    max_keyframes = st.slider("Maximum keyframes", 3, 24, 10)
    min_gap = st.slider("Minimum keyframe gap (frames)", 1, 300, 30)
    sharpness_gate = st.slider("Quality gate", 0.0, 1.0, 0.35, 0.05)
    pixel_stride = st.slider("Point-cloud pixel stride", 2, 16, 7)
    depth_scale = st.slider("Relative depth scale", 0.2, 8.0, 1.0)
    use_ai = st.toggle("Use Depth Anything V2", value=False)
    use_dynamic = st.toggle("Dynamic-object masking", value=False)
    st.divider()
    st.caption("For SIH demo: start with a 20–60 second 1080p UAV clip. Real metric/georeferenced output requires calibrated intrinsics plus GPS/IMU/RTK or a reliable pose estimate.")

# ----------------------------- Input -----------------------------
video = st.file_uploader("Upload single-pass UAV video", type=["mp4", "mov", "avi", "mkv"])
telemetry = st.file_uploader("Optional flight telemetry CSV (frame,time,lat,lon,alt)", type=["csv"])

if video is None:
    st.info("Upload a UAV video to start the reconstruction pipeline.")
    st.markdown("#### Pipeline")
    st.code("CAPTURE → QUALITY GATE → KEYFRAMES → AI DEPTH → DYNAMIC MASK → POSE → CONFIDENCE FUSION → 3D DIGITAL TWIN")
    st.stop()

path = save_upload(video.getvalue(), Path(video.name).suffix or ".mp4")
info = get_video_info(path)

# ----------------------------- Overview metrics -----------------------------
cols = st.columns(5)
cols[0].metric("Duration", f"{info['duration']:.1f} s")
cols[1].metric("Frames", f"{info['frames']:,}")
cols[2].metric("Resolution", f"{info['width']}×{info['height']}")
cols[3].metric("FPS", f"{info['fps']:.1f}")
cols[4].metric("Telemetry", "Attached" if telemetry else "Not attached")

# ----------------------------- Keyframes -----------------------------
st.markdown("### 01 · Quality-aware keyframe selection")
with st.spinner("Scanning video and ranking frames..."):
    selected, candidates = select_keyframes(path, quality_tab, max_keyframes, min_gap)

if not selected:
    st.error("No frames could be selected from the video.")
    st.stop()

usable = [x for x in selected if x["quality"] >= sharpness_gate]
if len(usable) < 3:
    st.warning(f"Only {len(usable)} selected frames clear the current quality gate. Consider lowering the gate or using a smoother flight video.")
    usable = selected

# Candidate quality chart
fig_q = go.Figure()
fig_q.add_trace(go.Scatter(x=[c["frame"] for c in candidates], y=[c["quality"] for c in candidates], mode="lines+markers", name="quality"))
fig_q.add_trace(go.Scatter(x=[c["frame"] for c in usable], y=[c["quality"] for c in usable], mode="markers", name="selected", marker=dict(size=9)))
fig_q.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Frame", yaxis_title="Quality")
st.plotly_chart(fig_q, use_container_width=True)

preview_cols = st.columns(min(5, len(usable)))
for i, s in enumerate(usable):
    with preview_cols[i % len(preview_cols)]:
        st.image(s["preview"], caption=f"F{s['frame']} • q={s['quality']:.2f}", use_container_width=True)

# ----------------------------- Depth model -----------------------------
model_bundle = None
if use_ai:
    with st.spinner("Loading Depth Anything V2 Small..."):
        model_bundle = load_depth_model()
    if not model_bundle["ok"]:
        st.error("Depth Anything V2 could not be loaded in this environment. Falling back to the lightweight depth proxy for this run.")
        st.caption(model_bundle.get("error", "Unknown error"))
        use_ai = False

# ----------------------------- Reconstruction -----------------------------
st.markdown("### 02 · AI depth + confidence-aware 3D reconstruction")
progress = st.progress(0)
status = st.empty()
clouds, colors, confs = [], [], []
depth_previews = []
mask_status = "Disabled"

for k, item in enumerate(usable):
    frame_idx = int(item["frame"])
    status.write(f"Processing keyframe {k+1}/{len(usable)} • frame {frame_idx}")
    frame = read_frame(path, frame_idx)
    if frame is None:
        continue
    depth = depth_anything(frame, model_bundle) if use_ai and model_bundle else depth_fallback(frame)

    if use_dynamic:
        static_mask, mask_status = try_yolo_mask(frame)
        depth = np.where(static_mask, depth, np.nan)
    else:
        static_mask = np.ones(depth.shape, dtype=bool)

    # Confidence blends keyframe quality + local depth stability + optional masking.
    grad = np.abs(cv2.Sobel(np.nan_to_num(depth), cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(np.nan_to_num(depth), cv2.CV_32F, 0, 1))
    stability = float(max(0.0, 1.0 - min(1.0, np.nanmean(grad) * 4.0))) if np.isfinite(depth).any() else 0.0
    confidence = float(np.clip(0.75 * item["quality"] + 0.25 * stability, 0, 1))

    # Save depth preview for UI.
    dshow = np.nan_to_num(depth, nan=0.0)
    dshow = (np.clip(dshow, 0, 1) * 255).astype(np.uint8)
    depth_previews.append((frame_idx, dshow, confidence))

    p, c, f = reprojection_points(frame, np.nan_to_num(depth, nan=0.0), confidence, pixel_stride, depth_scale)
    valid = np.isfinite(p).all(axis=1)
    if valid.any():
        clouds.append(p[valid]); colors.append(c[valid]); confs.append(f[valid])
    progress.progress((k+1) / max(1, len(usable)))

status.empty()
progress.empty()

if not clouds:
    st.error("Reconstruction produced no valid points.")
    st.stop()

P, C, CF = fuse_cloud(clouds, colors, confs)

# ----------------------------- Results -----------------------------
res = st.columns(4)
res[0].metric("3D points", f"{len(P):,}")
res[1].metric("High confidence", f"{np.mean(CF >= 0.75) * 100:.1f}%")
res[2].metric("Medium confidence", f"{np.mean((CF >= 0.45) & (CF < 0.75)) * 100:.1f}%")
res[3].metric("Low confidence", f"{np.mean(CF < 0.45) * 100:.1f}%")

left, right = st.columns([1.8, 1])
with left:
    fig = go.Figure(data=[go.Scatter3d(
        x=P[:,0], y=P[:,1], z=P[:,2], mode="markers",
        marker=dict(size=2, color=np.mean(C, axis=1), colorscale="Turbo", opacity=0.78, colorbar=dict(title="RGB intensity")),
        hoverinfo="skip"
    )])
    fig.update_layout(height=680, margin=dict(l=0,r=0,t=0,b=0), scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Depth previews")
    for idx, dshow, conf in depth_previews[:6]:
        st.image(dshow, caption=f"Frame {idx} • confidence {conf:.2f}", use_container_width=True)

st.markdown("### 03 · Confidence map")
hist, bins = np.histogram(CF, bins=12, range=(0,1))
fig_c = go.Figure(go.Bar(x=((bins[:-1] + bins[1:]) / 2), y=hist))
fig_c.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Confidence", yaxis_title="Points")
st.plotly_chart(fig_c, use_container_width=True)

# ----------------------------- Mesh / export -----------------------------
st.markdown("### 04 · Digital twin export")
mesh_data, mesh_status = try_open3d_mesh(P, C)
st.caption(mesh_status)

ply_bytes = make_ply(P, C)
report = {
    "project": PROJECT,
    "problem_statement_id": PS_ID,
    "prototype_version": APP_VERSION,
    "input_video": video.name,
    "video": info,
    "selected_keyframes": [
        {k: (int(v) if k == "frame" else float(v)) for k, v in s.items() if k not in {"preview"}}
        for s in usable
    ],
    "depth_engine": DEPTH_MODEL_ID if use_ai else "OpenCV fallback proxy",
    "dynamic_masking": mask_status,
    "points_generated": int(len(P)),
    "confidence": {
        "mean": float(np.mean(CF)),
        "high_pct": float(np.mean(CF >= 0.75) * 100),
        "medium_pct": float(np.mean((CF >= 0.45) & (CF < 0.75)) * 100),
        "low_pct": float(np.mean(CF < 0.45) * 100),
    },
    "coordinate_system": "relative prototype coordinates",
    "metric_georeferencing": "not applied in this demo; requires calibrated camera + pose/GPS/IMU/RTK fusion",
    "telemetry_file": telemetry.name if telemetry else None,
    "recommended_next_stage": [
        "COLMAP/VIO or camera-IMU pose estimation",
        "GPS/RTK/PPK georeferencing and metric scale",
        "multi-frame confidence-aware depth fusion",
        "dynamic-object masking with YOLO segmentation",
        "Open3D TSDF/MVS surface fusion",
        "GeoTIFF/3D Tiles/OBJ/GLB export",
    ],
}

b1, b2, b3 = st.columns(3)
with b1:
    st.download_button("⬇ Export point cloud (.PLY)", ply_bytes, file_name="vayuputra_reconstruction.ply", mime="application/octet-stream", use_container_width=True)
with b2:
    st.download_button("⬇ Export report (.JSON)", json.dumps(report, indent=2).encode(), file_name="vayuputra_report.json", mime="application/json", use_container_width=True)
with b3:
    if mesh_data:
        st.download_button("⬇ Export mesh (.OBJ)", mesh_data, file_name="vayuputra_digital_twin.obj", mime="text/plain", use_container_width=True)
    else:
        st.button("Mesh export unavailable", disabled=True, use_container_width=True)

# ----------------------------- Architecture -----------------------------
with st.expander("System architecture / SIH explanation", expanded=False):
    st.markdown(
        """
        **Capture** → UAV video + optional GPS/IMU/RTK telemetry  
        **Pre-processing** → blur/exposure/texture quality gate + temporal keyframe selection  
        **AI reconstruction** → monocular depth + optional dynamic-object masking + confidence scoring  
        **Geometry** → normalized pinhole reprojection → multi-frame fusion  
        **Output** → interactive point cloud + optional Open3D mesh + machine-readable reconstruction report  

        The production version should add calibrated intrinsics, COLMAP/VIO camera poses, sensor-fused metric scale, explicit uncertainty propagation and geospatial export. The current demo deliberately exposes where those production-grade modules plug in.
        """)

st.success("SkyRecon prototype run complete — single-pass video transformed into a confidence-aware 3D scene representation.")
