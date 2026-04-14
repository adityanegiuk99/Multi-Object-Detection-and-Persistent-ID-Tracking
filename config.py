"""
Configuration Module — Multi-Object Detection & Persistent ID Tracking Pipeline
================================================================================

Centralizes ALL tunable parameters for the pipeline. Override any parameter
via command-line arguments (see --help).

Hardware Target: Acer Aspire 7 — i5-13420H, RTX 3050 6GB, 24GB RAM
"""

import argparse
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"
INPUT_DIR = PROJECT_ROOT / "input"

# Auto-create directories
for _dir in (OUTPUT_DIR, SAMPLES_DIR, INPUT_DIR):
    _dir.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Video Source Configuration
# ──────────────────────────────────────────────────────────────
VIDEO_URL = "https://www.youtube.com/watch?v=OFbyNU6UQQs"
VIDEO_FILENAME = "portugal_vs_spain_2018.mp4"
INPUT_VIDEO = INPUT_DIR / VIDEO_FILENAME

CLIP_FILENAME = "clip_90s.mp4"
CLIP_VIDEO = INPUT_DIR / CLIP_FILENAME

# Trimming parameters — selects a segment with dense player interaction,
# camera panning, and partial occlusion from the Portugal vs Spain match.
CLIP_START = 170         # Start time (seconds into the source video)
CLIP_DURATION = 90       # Duration (seconds)

# ──────────────────────────────────────────────────────────────
# Detection Configuration
# ──────────────────────────────────────────────────────────────
# YOLO11m chosen for optimal accuracy/speed trade-off on RTX 3050.
# - Higher mAP than YOLOv8m with fewer parameters (20.1M vs 25.9M)
# - C2PSA attention module improves detection in crowded scenes
# - m-variant balances speed (~35 FPS on 3050) and accuracy
YOLO_MODEL = "yolo11m.pt"
DETECTION_CONF = 0.3          # Min confidence threshold (low for rescue matching)
DETECTION_IOU_NMS = 0.5       # NMS IoU threshold
PERSON_CLASS_ID = 0            # COCO class ID for 'person'
INFERENCE_SIZE = 640           # YOLO input resolution (multiple of 32)

# Detection post-filtering
MIN_BBOX_AREA = 400            # Min bounding box area (px²) to filter noise
MAX_BBOX_ASPECT_RATIO = 6.0    # Max H/W ratio — removes thin vertical artifacts

# ──────────────────────────────────────────────────────────────
# Tracker Configuration
# ──────────────────────────────────────────────────────────────
TRACKER_TYPE = "bytetrack"     # Options: "bytetrack", "botsort", "deepsort"

# ByteTrack / BoT-SORT — Ultralytics built-in tracker parameters.
# ByteTrack excels via two-stage association: high-confidence detections
# matched first, then low-confidence detections "rescue" lost tracks.
TRACK_HIGH_THRESH = 0.5       # 1st-stage association confidence
TRACK_LOW_THRESH = 0.1        # 2nd-stage rescue confidence
TRACK_BUFFER = 30             # Frames to buffer lost tracks before deletion

# DeepSORT — Appearance-based re-identification tracker.
# Uses deep appearance embeddings for association but adds ~5-10ms overhead.
DEEPSORT_MAX_AGE = 30          # Max frames without match before track deletion
DEEPSORT_N_INIT = 3            # Min consecutive hits to confirm a new track
DEEPSORT_MAX_IOU_DISTANCE = 0.7
DEEPSORT_MAX_COSINE_DISTANCE = 0.3
DEEPSORT_NN_BUDGET = 100       # Gallery size for appearance features

# ──────────────────────────────────────────────────────────────
# Scene Cut / Camera Motion Detection
# ──────────────────────────────────────────────────────────────
# Used to reset tracker state on hard cuts (replays, camera switches)
# to prevent cross-scene ID pollution.
SCENE_CUT_THRESHOLD = 0.35    # Histogram correlation below this = scene cut
SCENE_CUT_RESET_TRACKER = True # Reset all tracks on scene cut

# ──────────────────────────────────────────────────────────────
# Visualization Settings
# ──────────────────────────────────────────────────────────────
TRAIL_LENGTH = 30              # Trajectory trail points per track
BBOX_THICKNESS = 2             # Bounding box line width
FONT_SCALE = 0.6               # Label font size
FONT_THICKNESS = 2             # Label font weight
SHOW_CONFIDENCE = True         # Display confidence with ID label
SHOW_TRAJECTORY = True         # Draw trajectory trails
SHOW_STATS_PANEL = True        # Overlay statistics panel

# ──────────────────────────────────────────────────────────────
# Heatmap Configuration
# ──────────────────────────────────────────────────────────────
HEATMAP_RESOLUTION = (108, 192)  # Grid resolution (H, W) — 16:9 aspect
HEATMAP_DECAY = 0.995            # Temporal decay factor (0.99 = slow fade)
HEATMAP_ALPHA = 0.4              # Overlay blend transparency
HEATMAP_COLORMAP = 11            # OpenCV colormap ID (11 = COLORMAP_JET)

# ──────────────────────────────────────────────────────────────
# Speed Estimation (Approximate)
# ──────────────────────────────────────────────────────────────
# Rough pixel-to-meter calibration based on standard football field.
# Used for approximate speed display — not scientifically precise.
FIELD_LENGTH_METERS = 105.0
ASSUMED_FIELD_WIDTH_PIXELS = 1280
PIXELS_PER_METER = ASSUMED_FIELD_WIDTH_PIXELS / FIELD_LENGTH_METERS
SPEED_SMOOTHING_WINDOW = 5      # Frames to average speed over

# ──────────────────────────────────────────────────────────────
# Performance Optimization
# ──────────────────────────────────────────────────────────────
FRAME_SKIP = 0                   # 0 = every frame; N = skip N frames between
INPUT_RESIZE = None              # (width, height) or None for original resolution
DEVICE = "auto"                  # "auto" (prefers cuda), "cuda", "cpu"
HALF_PRECISION = True            # FP16 inference on GPU (2x throughput on 3050)
BATCH_SIZE = 1                   # Single-frame batch (streaming mode)

# ──────────────────────────────────────────────────────────────
# Output Configuration
# ──────────────────────────────────────────────────────────────
OUTPUT_VIDEO_CODEC = "mp4v"
OUTPUT_FPS = None                # None = match input FPS
SAVE_STATISTICS = True
STATISTICS_FILE = OUTPUT_DIR / "tracking_statistics.json"
HEATMAP_OUTPUT = OUTPUT_DIR / "movement_heatmap.png"

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = OUTPUT_DIR / "pipeline.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"


def get_device_string(preference: str = "auto") -> str:
    """Resolve device preference to a concrete device string.

    Args:
        preference: One of 'auto', 'cuda', 'cpu'.

    Returns:
        Device string suitable for Ultralytics/PyTorch.
    """
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            print(f"  ✓ GPU detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
            return "cuda"
        else:
            print("  ⚠ No CUDA GPU found — falling back to CPU")
            return "cpu"
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Install CUDA toolkit.")
        return "cuda"
    return "cpu"


def parse_args():
    """Parse command-line arguments with comprehensive options.

    Returns:
        argparse.Namespace with all configuration overrides.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection & Persistent ID Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Examples:
  python main.py                                # Default (ByteTrack + YOLO11m)
  python main.py --tracker deepsort             # Compare with DeepSORT
  python main.py --input myvideo.mp4            # Custom input video
  python main.py --skip-download                # Skip video download
  python main.py --frame-skip 2 --device cuda   # Optimize for speed
  python main.py --compare                      # Run ByteTrack vs DeepSORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )

    # ── Input / Output ──
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--input", type=str, default=None,
                          help="Path to input video (overrides auto-download)")
    io_group.add_argument("--output", type=str, default=None,
                          help="Path for output video file")

    # ── Model ──
    model_group = parser.add_argument_group("Detection Model")
    model_group.add_argument("--model", type=str, default=YOLO_MODEL,
                             help=f"YOLO model variant (default: {YOLO_MODEL})")
    model_group.add_argument("--conf", type=float, default=DETECTION_CONF,
                             help=f"Detection confidence threshold (default: {DETECTION_CONF})")
    model_group.add_argument("--imgsz", type=int, default=INFERENCE_SIZE,
                             help=f"Inference image size (default: {INFERENCE_SIZE})")

    # ── Tracker ──
    tracker_group = parser.add_argument_group("Tracking Algorithm")
    tracker_group.add_argument("--tracker", type=str, default=TRACKER_TYPE,
                               choices=["bytetrack", "botsort", "deepsort"],
                               help=f"Tracker type (default: {TRACKER_TYPE})")
    tracker_group.add_argument("--compare", action="store_true",
                               help="Run comparison: ByteTrack vs DeepSORT")

    # ── Device / Performance ──
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--device", type=str, default=DEVICE,
                            choices=["auto", "cuda", "cpu"],
                            help=f"Compute device (default: {DEVICE})")
    perf_group.add_argument("--frame-skip", type=int, default=FRAME_SKIP,
                            help=f"Skip N frames between processing (default: {FRAME_SKIP})")
    perf_group.add_argument("--half", action="store_true", default=HALF_PRECISION,
                            help="Use FP16 inference (default: enabled for GPU)")
    perf_group.add_argument("--no-half", action="store_true",
                            help="Disable FP16 inference")

    # ── Video Download ──
    dl_group = parser.add_argument_group("Video Download")
    dl_group.add_argument("--skip-download", action="store_true",
                          help="Skip yt-dlp video download step")
    dl_group.add_argument("--clip-start", type=int, default=CLIP_START,
                          help=f"Clip start time in seconds (default: {CLIP_START})")
    dl_group.add_argument("--clip-duration", type=int, default=CLIP_DURATION,
                          help=f"Clip duration in seconds (default: {CLIP_DURATION})")

    # ── Display ──
    disp_group = parser.add_argument_group("Display")
    disp_group.add_argument("--no-display", action="store_true",
                            help="Suppress live preview window")
    disp_group.add_argument("--no-trajectory", action="store_true",
                            help="Disable trajectory trails")
    disp_group.add_argument("--no-heatmap", action="store_true",
                            help="Disable heatmap generation")

    args = parser.parse_args()

    # Post-processing
    if args.no_half:
        args.half = False

    return args
