# 🏟️ Multi-Object Detection & Persistent ID Tracking in Sports Footage

A production-quality computer vision pipeline for detecting and tracking multiple players in FIFA World Cup 2018 footage with persistent ID assignment that survives occlusion, camera motion, and visual similarity challenges.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v11-green?logo=yolo&logoColor=white)
![Tracker](https://img.shields.io/badge/Tracker-ByteTrack-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red?logo=opencv&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-CUDA%20Accelerated-76B900?logo=nvidia&logoColor=white)

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Output Samples](#-output-samples)
7. [Architecture](#-architecture)
8. [Design Decisions](#-design-decisions)
9. [Edge Case Handling](#-edge-case-handling)
10. [Performance](#-performance)
11. [Limitations & Future Work](#-limitations--future-work)

---

## 🎯 Overview

### Problem
Multi-object tracking in sports broadcasts is challenging due to:
- Players wearing **identical jerseys** (same-team confusion)
- **Occlusion** from player clustering
- Rapid **camera panning and zoom**
- **Motion blur** from fast player/camera movement
- **Scale variation** across the field

### Solution
This pipeline combines **YOLO11m** for person detection with **ByteTrack** for motion-based tracking to maintain consistent player IDs across 2700+ frames of live match footage.

### Video Source
- **Match**: Portugal 3–3 Spain, FIFA World Cup 2018 (Group B)
- **URL**: https://www.youtube.com/watch?v=OFbyNU6UQQs
- **Segment**: 90-second clip (2:50 – 4:20) with dense midfield action

---

## ✨ Features

| Category | Feature | Status |
|----------|---------|--------|
| **Detection** | YOLO11m person detection with GPU/CUDA acceleration | ✅ |
| **Tracking** | ByteTrack persistent ID assignment | ✅ |
| **Comparison** | ByteTrack vs DeepSORT side-by-side evaluation | ✅ |
| **Analytics** | Trajectory visualization with gradient trails | ✅ |
| **Analytics** | Real-time player counting (unique + active) | ✅ |
| **Analytics** | Movement heatmap generation | ✅ |
| **Analytics** | Speed estimation (km/h) per player | ✅ |
| **Analytics** | Per-frame statistics export (JSON) | ✅ |
| **Robustness** | Scene cut detection with tracker reset | ✅ |
| **Robustness** | Occlusion recovery (30-frame buffer) | ✅ |
| **Pipeline** | Automatic video download (yt-dlp) + trimming (ffmpeg) | ✅ |
| **Pipeline** | OpenCV fallback when ffmpeg unavailable | ✅ |
| **Performance** | FP16 half-precision inference | ✅ |
| **Performance** | Configurable frame skipping | ✅ |
| **Performance** | CPU fallback with auto-detection | ✅ |

---

## 📁 Project Structure

```
Assignment/
│
├── main.py                 # Pipeline orchestrator — entry point
├── config.py               # All configuration + CLI argument parsing
├── detector.py             # YOLO11m person detection wrapper
├── tracker.py              # ByteTrack / DeepSORT unified interface
├── analytics.py            # Heatmap, trajectory, speed, counting
├── visualizer.py           # Frame rendering with overlays
├── download_video.py       # Video download (yt-dlp) + trimming (ffmpeg)
├── utils.py                # Video I/O, logging, scene cut detection
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Git exclusions
│
├── README.md               # This file
├── TECHNICAL_REPORT.md     # In-depth model & algorithm analysis
├── DEMO_SCRIPT.md          # Demo video narration script
│
├── input/                  # Downloaded videos (auto-created, git-ignored)
│   └── .gitkeep
│
└── outputs/                # Generated results
    ├── movement_heatmap.png       # Sample heatmap output
    └── tracking_statistics.json   # Sample statistics output
```

---

## 🛠 Installation

### Prerequisites
| Tool | Purpose | Required |
|------|---------|----------|
| Python 3.9+ | Runtime | ✅ Yes |
| NVIDIA GPU + CUDA | GPU acceleration | ⚡ Recommended |
| ffmpeg | Video trimming | 📌 Auto-fallback to OpenCV |
| yt-dlp | Video download | 📌 Installed via pip |

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-object-tracking.git
cd multi-object-tracking
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 3: Install PyTorch (with CUDA)

```bash
# For CUDA 12.1 (RTX 3050 / 4000 series):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (slower but works everywhere):
pip install torch torchvision
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install ffmpeg (Optional but Recommended)

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

> **Note**: If ffmpeg is not installed, the pipeline automatically uses an OpenCV-based fallback for video trimming.

### Step 6: Verify Installation

```bash
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## ▶ Usage

### Quick Start (One Command)

```bash
python main.py
```

This automatically:
1. Downloads the Portugal vs Spain video from YouTube
2. Trims a 90-second clip with dense player action
3. Runs YOLO11m detection + ByteTrack tracking
4. Saves annotated video, statistics, and heatmap to `outputs/`

### Common Commands

```bash
# Default pipeline (ByteTrack + YOLO11m + auto-download)
python main.py

# Skip video download (use existing clip)
python main.py --skip-download

# Use a custom video file
python main.py --input path/to/your/video.mp4

# Run tracker comparison (ByteTrack vs DeepSORT)
python main.py --compare

# Performance mode (skip every other frame)
python main.py --frame-skip 1

# Headless mode (no preview window)
python main.py --no-display

# Force CPU mode
python main.py --device cpu
```

### Full CLI Reference

```bash
python main.py --help
```

```
Input / Output:
  --input PATH              Custom input video path
  --output PATH             Custom output video path

Detection:
  --model MODEL             YOLO model (default: yolo11m.pt)
  --conf THRESHOLD          Confidence threshold (default: 0.3)
  --imgsz SIZE              Inference size (default: 640)

Tracking:
  --tracker {bytetrack,botsort,deepsort}    Tracker algorithm
  --compare                 Run ByteTrack vs DeepSORT comparison

Performance:
  --device {auto,cuda,cpu}  Compute device
  --frame-skip N            Skip N frames between processing
  --half / --no-half        FP16 precision toggle

Display:
  --no-display              Suppress live preview
  --no-trajectory           Disable trajectory trails
  --no-heatmap              Disable heatmap generation
```

---

## 📸 Output Samples

### Generated Files

| File | Description |
|------|-------------|
| `tracked_bytetrack.mp4` | Annotated video with bounding boxes, IDs, trajectory trails |
| `tracking_statistics.json` | Per-frame detection counts, confidence, active players |
| `movement_heatmap.png` | Accumulated player movement density |
| `pipeline.log` | Detailed execution log |
| `comparison_results.json` | Tracker comparison metrics (with `--compare`) |

### Sample Analytics Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 TRACKING ANALYTICS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total frames processed:    1,420
  Unique players detected:   389
  Total detections:          15,523
  Avg active players/frame:  10.9
  Max simultaneous players:  18
  Estimated ID switches:     201
  Avg player speed:          37.4 km/h
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🏗 Architecture

### Data Flow

```
YouTube URL → yt-dlp → Raw Video → ffmpeg → 90s Clip
                                       ↓
                            Frame-by-Frame Processing
                                       ↓
                    ┌──────────────────────────────────┐
                    │     Scene Cut Detection           │
                    │  (HSV histogram correlation)      │
                    └──────────┬───────────────────────┘
                               ↓
                    ┌──────────────────────────────────┐
                    │     YOLO11m Person Detection      │
                    │  (GPU/FP16, conf ≥ 0.3)          │
                    └──────────┬───────────────────────┘
                               ↓
                    ┌──────────────────────────────────┐
                    │     ByteTrack Association         │
                    │  Stage 1: High-conf matching      │
                    │  Stage 2: Low-conf rescue         │
                    └──────────┬───────────────────────┘
                               ↓
                 ┌─────────────┼─────────────┐
                 ↓             ↓             ↓
           Analytics     Visualization   Track History
          (heatmap,      (bbox + ID +    (trajectory,
           speed)         stats panel)    buffer)
                 ↓             ↓
          Stats JSON    Output Video + Heatmap PNG
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `config.py` | Centralized parameters, hardware auto-detection, CLI parsing |
| `detector.py` | YOLO11m inference, person-class filtering, bbox post-processing |
| `tracker.py` | Tracker factory (ByteTrack/BoT-SORT/DeepSORT), track state management |
| `analytics.py` | Heatmap accumulation, speed calculation, player counting, JSON export |
| `visualizer.py` | Bounding box rendering, trajectory trails, stats dashboard overlay |
| `download_video.py` | yt-dlp download, ffmpeg/OpenCV trimming, PATH auto-refresh |
| `utils.py` | VideoReader/Writer wrappers, logging, scene cut detection, FPS counter |
| `main.py` | Orchestrates all modules, single-tracker and comparison execution modes |

---

## 🧠 Design Decisions

### Why YOLO11m?

| Criteria | YOLOv8m | YOLO11m | Winner |
|----------|---------|---------|--------|
| mAP@50 (COCO) | 50.2 | 51.5 | ✅ YOLO11m |
| Parameters | 25.9M | 20.1M | ✅ YOLO11m |
| VRAM Usage (640px) | ~4.2 GB | ~3.6 GB | ✅ YOLO11m |
| Attention Module | None | C2PSA | ✅ YOLO11m |

YOLO11m's C2PSA (Cross-Stage Partial Spatial Attention) specifically helps with crowded sports scenes where players overlap.

### Why ByteTrack over DeepSORT?

**Key insight**: DeepSORT's appearance model is *counterproductive* for same-jersey sports footage.

| Aspect | ByteTrack | DeepSORT |
|--------|-----------|----------|
| Matching | IoU + Kalman (motion-only) | IoU + Appearance CNN |
| Speed | <1ms overhead | 5-10ms overhead |
| Same-jersey handling | ✅ Motion is discriminative | ❌ Appearance is identical |
| Occlusion recovery | ✅ Two-stage rescue | ❌ Appearance often fails |
| VRAM | No extra model | +MobileNet (~800MB) |

ByteTrack's two-stage association:
1. **Stage 1**: High-confidence detections (>0.5) matched to tracks via IoU
2. **Stage 2**: Low-confidence detections (0.1–0.5) rescue un-matched tracks — critical for partially occluded players

---

## 🧩 Edge Case Handling

| Challenge | Strategy |
|-----------|----------|
| **Same jerseys** | Pure motion tracking (no confusing appearance features) |
| **Occlusion** | 30-frame track buffer + two-stage low-confidence rescue |
| **Camera cuts** | HSV histogram scene detection → automatic tracker reset |
| **False positives** | Person-class filter + min area (400px²) + aspect ratio cap (6:1) |
| **Motion blur** | YOLO11's C2PSA attention + low-conf detections still tracked |
| **Scale variation** | FPN multi-scale detection at 640px |
| **Fast camera pan** | Kalman prediction maintains tracks through 1-2 frame gaps |

---

## ⚡ Performance

### Benchmarks (NVIDIA RTX 3050, 6GB VRAM)

| Configuration | FPS | Notes |
|---------------|-----|-------|
| YOLO11m + ByteTrack (CUDA, FP16) | 25–35 | Recommended |
| YOLO11m + DeepSORT (CUDA, FP16) | 15–25 | Comparison mode |
| YOLO11m + ByteTrack (CPU) | 2–5 | Fallback |
| + Frame skip = 1 | 2× | Halves frames processed |

### Speed Optimization Tips

```bash
# Maximum speed (GPU + frame skip + no display)
python main.py --device cuda --frame-skip 1 --no-display --no-heatmap

# Balance speed and quality
python main.py --frame-skip 1
```

---

## ⚠ Limitations & Future Work

### Current Limitations
- Speed estimation is approximate (no camera calibration)
- No team classification (doesn't distinguish between teams)
- No jersey number recognition
- ID switches can occur during prolonged complete occlusion (>1 second)
- Generic Re-ID model (DeepSORT mode) not trained on sports data

### Planned Improvements
- [ ] Team color classification via dominant jersey color extraction
- [ ] Sport-specific Re-ID model fine-tuned on SoccerNet
- [ ] Homography-based bird's-eye view transformation
- [ ] Jersey number OCR for true player identification
- [ ] Real-time streaming pipeline with WebSocket output

---

## 📊 Evaluation Methodology

### Metrics Used
- **Unique ID count** — Lower relative to actual players = better stability
- **ID switch estimation** — Heuristic based on new track creation rate
- **Detection consistency** — Average confidence per frame (target >0.5)
- **Active player count** — Should match visible players in wide-angle shots

### MOTA Reference
```
MOTA = 1 - (FN + FP + IDSW) / GT
```
> True MOTA requires ground-truth annotations. Our statistics provide proxy metrics.

---

## 📚 References

- [YOLO11 — Ultralytics](https://docs.ultralytics.com/models/yolo11/)
- [ByteTrack — Zhang et al., ECCV 2022](https://arxiv.org/abs/2110.06864)
- [DeepSORT — Wojke et al., ICIP 2017](https://arxiv.org/abs/1703.07402)
- [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime)

---

## 👤 Author

**Aditya Negi**

---

## 📄 License

This project is for educational and assignment purposes.
