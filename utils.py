"""
Utility Module — Video I/O, Logging, Scene Detection, Helper Functions
========================================================================

Provides foundational utilities used across all pipeline components:
- Video reading/writing with OpenCV
- Structured logging setup
- Scene cut detection for tracker reset
- Color generation for persistent ID visualization
- FPS measurement
"""

import colorsys
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config


# ──────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────

def setup_logging(level: str = config.LOG_LEVEL,
                  log_file: Optional[Path] = config.LOG_FILE) -> logging.Logger:
    """Configure structured logging to console and file.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. None for console-only.

    Returns:
        Root logger instance.
    """
    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(config.LOG_FORMAT)

    # Console handler — with color support
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper()))
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ──────────────────────────────────────────────────────────────
# Video Reader
# ──────────────────────────────────────────────────────────────

class VideoReader:
    """OpenCV-based video reader with metadata extraction and frame iteration.

    Usage:
        reader = VideoReader("input.mp4")
        for frame_idx, frame in reader:
            # process frame
        reader.release()
    """

    def __init__(self, path: str, resize: Optional[Tuple[int, int]] = None):
        """
        Args:
            path: Path to video file.
            resize: Optional (width, height) to resize frames.

        Raises:
            FileNotFoundError: If video file does not exist.
            RuntimeError: If video cannot be opened.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")

        self.resize = resize
        self.frame_idx = 0

        # Extract metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Effective dimensions after resize
        if self.resize:
            self.eff_width, self.eff_height = self.resize
        else:
            self.eff_width, self.eff_height = self.width, self.height

    def __iter__(self):
        self.frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration

        if self.resize:
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)

        idx = self.frame_idx
        self.frame_idx += 1
        return idx, frame

    def __len__(self):
        return self.total_frames

    def get_info(self) -> dict:
        """Return video metadata dictionary."""
        return {
            "path": str(self.path),
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "effective_resolution": f"{self.eff_width}x{self.eff_height}",
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration, 2),
        }

    def release(self):
        """Release the video capture resource."""
        if self.cap:
            self.cap.release()

    def __del__(self):
        self.release()


# ──────────────────────────────────────────────────────────────
# Video Writer
# ──────────────────────────────────────────────────────────────

class VideoWriter:
    """OpenCV-based video writer with H.264/MP4V encoding.

    Usage:
        writer = VideoWriter("output.mp4", fps=30.0, frame_size=(1280, 720))
        writer.write(frame)
        writer.release()
    """

    def __init__(self, path: str, fps: float, frame_size: Tuple[int, int],
                 codec: str = config.OUTPUT_VIDEO_CODEC):
        """
        Args:
            path: Output video file path.
            fps: Output frames per second.
            frame_size: (width, height) of output frames.
            codec: FourCC codec code (default: mp4v).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, frame_size)

        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {self.path}")

        self.frames_written = 0
        self.logger = logging.getLogger("pipeline.writer")

    def write(self, frame: np.ndarray):
        """Write a single frame to the output video."""
        self.writer.write(frame)
        self.frames_written += 1

    def release(self):
        """Release the writer and finalize the video file."""
        if self.writer:
            self.writer.release()
            self.logger.info(f"Output video saved: {self.path} ({self.frames_written} frames)")

    def __del__(self):
        self.release()


# ──────────────────────────────────────────────────────────────
# Scene Cut Detection
# ──────────────────────────────────────────────────────────────

class SceneCutDetector:
    """Detects hard scene cuts (camera switches, replay transitions) using
    histogram correlation between consecutive frames.

    When a scene cut is detected, the tracker should reset its state to
    prevent cross-scene ID contamination.

    Design Decision:
        We use HSV histogram comparison rather than pixel-level difference
        because it's invariant to minor exposure changes / auto-white-balance
        but sensitive to major scene composition changes.
    """

    def __init__(self, threshold: float = config.SCENE_CUT_THRESHOLD):
        """
        Args:
            threshold: Correlation threshold; values below this indicate a cut.
                       Lower = more sensitive. Default 0.35 catches hard cuts.
        """
        self.threshold = threshold
        self.prev_hist = None
        self.logger = logging.getLogger("pipeline.scene_cut")

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute normalized HSV histogram for a frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def check(self, frame: np.ndarray) -> bool:
        """Check if the current frame represents a scene cut.

        Args:
            frame: Current BGR frame.

        Returns:
            True if a scene cut was detected.
        """
        hist = self._compute_histogram(frame)

        if self.prev_hist is None:
            self.prev_hist = hist
            return False

        # Compare via correlation — 1.0 = identical, 0.0 = no match
        correlation = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
        self.prev_hist = hist

        is_cut = correlation < self.threshold
        if is_cut:
            self.logger.warning(f"Scene cut detected (correlation={correlation:.3f})")
        return is_cut

    def reset(self):
        """Reset stored histogram (e.g., after seeking in video)."""
        self.prev_hist = None


# ──────────────────────────────────────────────────────────────
# Color Generation for Track IDs
# ──────────────────────────────────────────────────────────────

def id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a unique, visually distinct BGR color for a given track ID.

    Uses the golden ratio to distribute hues evenly across the color wheel,
    ensuring that sequentially assigned IDs get maximally different colors.

    Args:
        track_id: Integer track identifier.

    Returns:
        BGR color tuple (0-255 per channel).
    """
    golden_ratio_conjugate = 0.618033988749895
    hue = (track_id * golden_ratio_conjugate) % 1.0
    # High saturation + value for vibrant, visible colors
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV


def get_contrasting_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return black or white text color based on background luminance.

    Args:
        bg_color: BGR background color.

    Returns:
        (0,0,0) for light backgrounds, (255,255,255) for dark backgrounds.
    """
    b, g, r = bg_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


# ──────────────────────────────────────────────────────────────
# FPS Counter
# ──────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling FPS calculator using exponential moving average.

    Provides smooth FPS readings suitable for display overlays.
    """

    def __init__(self, smoothing: float = 0.9):
        """
        Args:
            smoothing: EMA factor (higher = smoother, slower response).
        """
        self.smoothing = smoothing
        self._fps = 0.0
        self._last_time = None

    def tick(self) -> float:
        """Record a frame timestamp and return the current FPS estimate."""
        now = time.perf_counter()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self._fps = self.smoothing * self._fps + (1 - self.smoothing) * instant_fps
        self._last_time = now
        return self._fps

    @property
    def fps(self) -> float:
        """Current smoothed FPS value."""
        return self._fps


# ──────────────────────────────────────────────────────────────
# Misc Helpers
# ──────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Format seconds as MM:SS string."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def clamp_bbox(bbox: Tuple[int, int, int, int],
               frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Clamp bounding box coordinates to frame boundaries.

    Args:
        bbox: (x1, y1, x2, y2) coordinates.
        frame_w: Frame width.
        frame_h: Frame height.

    Returns:
        Clamped (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w - 1))
    y2 = max(0, min(y2, frame_h - 1))
    return (x1, y1, x2, y2)


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box1, box2: (x1, y1, x2, y2) format.

    Returns:
        IoU score in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0
