"""
Tracker Module — Multi-Object Tracking Abstraction Layer
============================================================

Provides a unified interface for multiple tracking algorithms:
1. ByteTrack  — via Ultralytics built-in (primary recommendation)
2. BoT-SORT   — via Ultralytics built-in (alternative)
3. DeepSORT   — via deep-sort-realtime library (comparison baseline)

Each tracker returns a standardized TrackResult format, enabling
seamless switching via config/CLI for comparison experiments.

Architecture Decision — Why Abstraction Layer:
    The assignment requires comparing ByteTrack vs DeepSORT. Rather than
    duplicating pipeline code, we define a common interface (BaseTracker)
    and swap implementations via factory pattern. This also allows adding
    new trackers (OC-SORT, StrongSORT) with zero pipeline changes.

Edge Case Handling:
    - Scene cut detection triggers tracker reset to prevent cross-scene ID
    - Lost track buffer enables re-identification after short occlusion
    - Minimum-area filtering removes spurious micro-detections
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from detector import PersonDetector, DetectionResult


@dataclass
class TrackResult:
    """Standardized tracking output — used by all tracker implementations.

    Attributes:
        track_id: Persistent integer ID assigned by the tracker.
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        confidence: Detection confidence score [0, 1].
        center: (cx, cy) bounding box center point.
        is_confirmed: Whether the track is confirmed (vs tentative).
    """
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[int, int] = field(init=False)
    is_confirmed: bool = True

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


# ──────────────────────────────────────────────────────────────
# Track History Manager
# ──────────────────────────────────────────────────────────────

class TrackHistoryManager:
    """Maintains per-ID trajectory history for analytics and visualization.

    Stores the last N positions for each track ID, enabling:
    - Trajectory trail rendering
    - Speed estimation
    - Movement heatmap accumulation

    Design Note:
        This is separate from the tracker itself because tracker
        implementations have different internal state representations.
        Centralizing history here avoids duplication.
    """

    def __init__(self, max_history: int = config.TRAIL_LENGTH):
        self.max_history = max_history
        self.histories: Dict[int, list] = defaultdict(list)
        self.all_ids: set = set()
        self.frame_active_ids: set = set()

    def update(self, tracks: List[TrackResult], frame_idx: int):
        """Record positions from current frame's tracks.

        Args:
            tracks: List of TrackResult from current frame.
            frame_idx: Current frame index.
        """
        self.frame_active_ids = set()
        for track in tracks:
            tid = track.track_id
            self.histories[tid].append({
                "center": track.center,
                "bbox": track.bbox,
                "frame": frame_idx,
                "confidence": track.confidence,
            })
            # Trim to max history length
            if len(self.histories[tid]) > self.max_history * 3:
                self.histories[tid] = self.histories[tid][-self.max_history * 3:]
            self.all_ids.add(tid)
            self.frame_active_ids.add(tid)

    def get_trail(self, track_id: int, length: Optional[int] = None) -> List[Tuple[int, int]]:
        """Get recent position trail for a track ID.

        Args:
            track_id: Target track ID.
            length: Trail length (None = default from config).

        Returns:
            List of (cx, cy) positions, most recent last.
        """
        n = length or self.max_history
        history = self.histories.get(track_id, [])
        return [h["center"] for h in history[-n:]]

    def get_speed_pixels_per_frame(self, track_id: int,
                                    window: int = config.SPEED_SMOOTHING_WINDOW) -> float:
        """Estimate instantaneous speed in pixels/frame.

        Uses average displacement over the last `window` frames
        to smooth out noise from detection jitter.

        Args:
            track_id: Target track ID.
            window: Number of frames to average over.

        Returns:
            Speed in pixels per frame (0.0 if insufficient data).
        """
        history = self.histories.get(track_id, [])
        if len(history) < 2:
            return 0.0

        recent = history[-window:]
        if len(recent) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(1, len(recent)):
            cx1, cy1 = recent[i - 1]["center"]
            cx2, cy2 = recent[i]["center"]
            total_dist += np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

        return total_dist / (len(recent) - 1)

    def get_unique_count(self) -> int:
        """Total unique IDs seen across all frames."""
        return len(self.all_ids)

    def get_active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self.frame_active_ids)

    def reset(self):
        """Clear all history (e.g., on scene cut)."""
        self.histories.clear()
        self.frame_active_ids.clear()
        # Note: all_ids is NOT cleared — cumulative count persists


# ──────────────────────────────────────────────────────────────
# Base Tracker Interface
# ──────────────────────────────────────────────────────────────

class BaseTracker(ABC):
    """Abstract base class for all tracker implementations."""

    def __init__(self, detector: PersonDetector):
        self.detector = detector
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")
        self.frame_count = 0

    @abstractmethod
    def update(self, frame: np.ndarray) -> List[TrackResult]:
        """Process a frame and return tracked objects.

        Args:
            frame: BGR image.

        Returns:
            List of TrackResult with persistent IDs.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset tracker state (called on scene cuts)."""
        ...

    def get_info(self) -> dict:
        """Return tracker metadata for reporting."""
        return {"type": self.__class__.__name__}


# ──────────────────────────────────────────────────────────────
# Ultralytics Tracker (ByteTrack / BoT-SORT)
# ──────────────────────────────────────────────────────────────

class UltralyticsTracker(BaseTracker):
    """ByteTrack / BoT-SORT via Ultralytics' native tracking API.

    ByteTrack Key Strengths:
        - Two-stage association: high-conf detections matched first (IoU),
          then low-conf detections "rescue" lost tracks. This dramatically
          reduces ID switches during occlusion.
        - No appearance model overhead — pure motion-based, extremely fast.
        - <1ms tracking overhead per frame on RTX 3050.

    BoT-SORT Enhancement:
        - Adds camera-motion compensation (GMC) via sparse optical flow.
        - Better suited for heavy camera panning (common in sports broadcasts).
        - Slight speed penalty (~2ms overhead) but improved accuracy.

    Usage:
        tracker = UltralyticsTracker(detector, tracker_type="bytetrack")
        tracks = tracker.update(frame)
    """

    def __init__(self, detector: PersonDetector,
                 tracker_type: str = "bytetrack"):
        """
        Args:
            detector: PersonDetector instance (shared model).
            tracker_type: 'bytetrack' or 'botsort'.
        """
        super().__init__(detector)
        self.tracker_type = tracker_type
        self.logger.info(f"Initialized {tracker_type.upper()} tracker (Ultralytics)")

    def update(self, frame: np.ndarray) -> List[TrackResult]:
        """Run detection + tracking in a single Ultralytics call.

        The model.track() method internally:
        1. Runs YOLO detection
        2. Passes detections to the built-in ByteTrack/BoT-SORT tracker
        3. Returns results with .boxes.id containing track IDs

        Args:
            frame: BGR image.

        Returns:
            List of TrackResult with persistent IDs.
        """
        result = self.detector.track(frame, self.tracker_type)
        self.frame_count += 1
        return self._parse_tracks(result)

    def _parse_tracks(self, result) -> List[TrackResult]:
        """Convert Ultralytics tracking result to standardized TrackResult list.

        Args:
            result: Ultralytics Results object with tracking IDs.

        Returns:
            Filtered list of TrackResult objects.
        """
        tracks = []

        if result.boxes is None or result.boxes.id is None:
            return tracks

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        confs = result.boxes.conf.cpu().numpy()

        for bbox, tid, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1
            h = y2 - y1

            # Post-filter: minimum area
            if w * h < config.MIN_BBOX_AREA:
                continue

            # Post-filter: aspect ratio
            if h / max(w, 1) > config.MAX_BBOX_ASPECT_RATIO:
                continue

            tracks.append(TrackResult(
                track_id=int(tid),
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
            ))

        return tracks

    def reset(self):
        """Reset Ultralytics tracker state on scene cut."""
        self.detector.reset()
        self.logger.info(f"{self.tracker_type.upper()} tracker state reset")

    def get_info(self) -> dict:
        return {
            "type": f"Ultralytics {self.tracker_type.upper()}",
            "tracker_config": f"{self.tracker_type}.yaml",
            "tracking_overhead_ms": "<1" if self.tracker_type == "bytetrack" else "~2",
        }


# ──────────────────────────────────────────────────────────────
# DeepSORT Tracker
# ──────────────────────────────────────────────────────────────

class DeepSORTTracker(BaseTracker):
    """DeepSORT tracker using the deep-sort-realtime library.

    DeepSORT Key Characteristics:
        - Uses Kalman filter for motion prediction (same as SORT/ByteTrack)
        - ADDS a deep appearance embedding network (Re-ID CNN)
        - Association uses both motion (Mahalanobis distance) and
          appearance (cosine distance of embeddings)

    Trade-offs vs ByteTrack:
        + Theoretically better re-identification after long occlusion
        + Appearance model helps distinguish different-looking individuals
        - 5-10ms extra overhead per frame for CNN feature extraction
        - Generic Re-ID model FAILS on same-jersey players (sports!)
        - Requires more GPU memory for the embedding network

    In Practice for Sports:
        - The appearance model is trained on pedestrian datasets, not sports.
        - All players on the same team look nearly identical (same jersey).
        - The Re-ID features add noise rather than signal for same-team players.
        - ByteTrack's motion-only approach is more robust here.

    Usage:
        tracker = DeepSORTTracker(detector)
        tracks = tracker.update(frame)
    """

    def __init__(self, detector: PersonDetector):
        super().__init__(detector)

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            self.logger.error(
                "deep-sort-realtime not installed. "
                "Install via: pip install deep-sort-realtime"
            )
            raise

        self.deepsort = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE,
            max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
            nn_budget=config.DEEPSORT_NN_BUDGET,
            embedder="mobilenet",        # Lightweight Re-ID backbone
            embedder_gpu=(detector.device == "cuda"),
        )

        self.logger.info("Initialized DeepSORT tracker (deep-sort-realtime)")

    def update(self, frame: np.ndarray) -> List[TrackResult]:
        """Run detection, then pass detections to DeepSORT for tracking.

        Flow:
        1. YOLO detection → list of (bbox, confidence)
        2. DeepSORT feature extraction → appearance embeddings
        3. DeepSORT association → Kalman prediction + Hungarian matching

        Args:
            frame: BGR image.

        Returns:
            List of TrackResult with persistent IDs.
        """
        # Step 1: Detection
        detections = self.detector.detect(frame)
        self.frame_count += 1

        if not detections:
            # Must still update tracker with empty detections
            # so that internal Kalman filters advance their state
            self.deepsort.update_tracks([], frame=frame)
            return []

        # Step 2: Format detections for DeepSORT
        # Expected format: list of ([x1, y1, w, h], confidence, class_name)
        bbs = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            bbs.append(([x1, y1, w, h], det.confidence, "person"))

        # Step 3: Update tracker
        raw_tracks = self.deepsort.update_tracks(bbs, frame=frame)

        # Step 4: Parse confirmed tracks
        tracks = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, ltrb)

            # Confidence from last detection (if available)
            det_conf = track.get_det_conf()
            conf = float(det_conf) if det_conf is not None else 0.5

            tracks.append(TrackResult(
                track_id=int(track.track_id),
                bbox=(x1, y1, x2, y2),
                confidence=conf,
            ))

        return tracks

    def reset(self):
        """Reset DeepSORT tracker state."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.deepsort = DeepSort(
                max_age=config.DEEPSORT_MAX_AGE,
                n_init=config.DEEPSORT_N_INIT,
                max_iou_distance=config.DEEPSORT_MAX_IOU_DISTANCE,
                max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
                nn_budget=config.DEEPSORT_NN_BUDGET,
                embedder="mobilenet",
                embedder_gpu=(self.detector.device == "cuda"),
            )
            self.logger.info("DeepSORT tracker state reset")
        except Exception as e:
            self.logger.error(f"Failed to reset DeepSORT: {e}")

    def get_info(self) -> dict:
        return {
            "type": "DeepSORT (deep-sort-realtime)",
            "max_age": config.DEEPSORT_MAX_AGE,
            "n_init": config.DEEPSORT_N_INIT,
            "embedder": "mobilenet",
            "tracking_overhead_ms": "5-10",
        }


# ──────────────────────────────────────────────────────────────
# Factory Function
# ──────────────────────────────────────────────────────────────

def create_tracker(tracker_type: str, detector: PersonDetector) -> BaseTracker:
    """Factory function to instantiate the appropriate tracker.

    Args:
        tracker_type: One of 'bytetrack', 'botsort', 'deepsort'.
        detector: Initialized PersonDetector instance.

    Returns:
        BaseTracker implementation.

    Raises:
        ValueError: If tracker_type is not recognized.
    """
    tracker_map = {
        "bytetrack": lambda: UltralyticsTracker(detector, "bytetrack"),
        "botsort": lambda: UltralyticsTracker(detector, "botsort"),
        "deepsort": lambda: DeepSORTTracker(detector),
    }

    if tracker_type not in tracker_map:
        raise ValueError(
            f"Unknown tracker: '{tracker_type}'. "
            f"Available: {list(tracker_map.keys())}"
        )

    return tracker_map[tracker_type]()
