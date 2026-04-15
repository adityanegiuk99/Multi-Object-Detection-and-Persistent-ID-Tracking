"""
Detection Module — YOLO11 Person Detection Wrapper
======================================================

Wraps the Ultralytics YOLO model for person-only detection with:
- Automatic GPU/CPU device selection
- FP16 (half-precision) acceleration on CUDA
- Post-filtering: minimum area, aspect ratio, class filtering
- Standardized output format via DetectionResult dataclass

Design Decision — Why YOLO11m:
    - 20.1M params vs YOLOv8m's 25.9M → lower VRAM usage on RTX 3050 (6GB)
    - C2PSA attention module improves crowded-scene detection (key for sports)
    - mAP@50 of 51.5 vs 50.2 on COCO — measurably better
    - Native Ultralytics integration for seamless tracking
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

import config


@dataclass
class DetectionResult:
    """Standardized detection output format.

    Attributes:
        bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
        confidence: Detection confidence score [0, 1].
        class_id: COCO class ID (0 = person).
        center: (cx, cy) center point of bounding box.
        area: Bounding box area in pixels².
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    center: Tuple[int, int] = field(init=False)
    area: int = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.area = (x2 - x1) * (y2 - y1)


class PersonDetector:
    """YOLO11-based person detector optimized for sports footage.

    Provides two modes of operation:
    1. detect() — Detection only (used with DeepSORT)
    2. track() — Detection + tracking via Ultralytics built-in tracker
                  (used with ByteTrack / BoT-SORT)

    Usage:
        detector = PersonDetector("yolo11m.pt", device="cuda")

        # Detection only
        detections = detector.detect(frame)

        # Detection + tracking (ByteTrack)
        result = detector.track(frame, tracker_type="bytetrack")
    """

    def __init__(self,
                 model_name: str = config.YOLO_MODEL,
                 device: str = "auto",
                 conf_threshold: float = config.DETECTION_CONF,
                 iou_threshold: float = config.DETECTION_IOU_NMS,
                 img_size: int = config.INFERENCE_SIZE,
                 half: bool = config.HALF_PRECISION):
        """Initialize the person detector.

        Args:
            model_name: YOLO model file name (auto-downloads if not found).
            device: Compute device ('auto', 'cuda', or 'cpu').
            conf_threshold: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            img_size: Input image size for inference (pixels).
            half: Enable FP16 inference (GPU only).
        """
        self.logger = logging.getLogger("pipeline.detector")
        self.device = config.get_device_string(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.half = half and (self.device == "cuda")

        # Load model
        self.logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)

        # Warm up model on the target device
        self.logger.info(f"Device: {self.device.upper()} | FP16: {self.half}")
        self._warmup()

        self.logger.info("✓ Detector initialized successfully")

    def _warmup(self):
        """Run a dummy inference to warm up the model & CUDA kernels."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy,
                device=self.device,
                half=self.half,
                verbose=False,
                conf=self.conf_threshold,
            )
            self.logger.debug("Model warm-up complete")
        except Exception as e:
            self.logger.warning(f"Warm-up failed (non-critical): {e}")

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run person detection on a single frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of DetectionResult for detected persons,
            filtered by confidence, area, and aspect ratio.
        """
        results = self.model.predict(
            frame,
            classes=[config.PERSON_CLASS_ID],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        return self._parse_detections(results[0])

    def track(self, frame: np.ndarray,
              tracker_type: str = "bytetrack") -> 'ultralytics.engine.results.Results':
        """Run detection + tracking using Ultralytics built-in tracker.

        This method is used for ByteTrack and BoT-SORT, which are natively
        integrated into the Ultralytics pipeline.

        Args:
            frame: BGR image (numpy array).
            tracker_type: Tracker config name ('bytetrack' or 'botsort').

        Returns:
            Raw Ultralytics Results object with tracking IDs attached.
        """
        results = self.model.track(
            frame,
            classes=[config.PERSON_CLASS_ID],
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            half=self.half,
            persist=True,                             # Maintain state across frames
            tracker=f"{tracker_type}.yaml",           # Built-in tracker config
            verbose=False,
        )
        return results[0]

    def _parse_detections(self, result) -> List[DetectionResult]:
        """Convert Ultralytics result to standardized DetectionResult list.

        Applies post-filtering:
        - Minimum bounding box area (removes tiny noise detections)
        - Aspect ratio check (removes thin vertical artifacts)

        Args:
            result: Single Ultralytics Results object.

        Returns:
            Filtered list of DetectionResult objects.
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)
        confs = result.boxes.conf.cpu().numpy()        # (N,)
        classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for bbox, conf, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # Filter: minimum area
            if area < config.MIN_BBOX_AREA:
                continue

            # Filter: aspect ratio (height/width)
            aspect_ratio = h / max(w, 1)
            if aspect_ratio > config.MAX_BBOX_ASPECT_RATIO:
                continue

            detections.append(DetectionResult(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                class_id=int(cls_id),
            ))

        return detections

    def get_model_info(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "model": str(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else config.YOLO_MODEL,
            "device": self.device,
            "half_precision": self.half,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "inference_size": self.img_size,
        }

    def reset(self):
        """Reset the internal tracker state (used on scene cuts).

        This is critical for ByteTrack/BoT-SORT when a scene cut is detected,
        ensuring no track IDs bleed across unrelated scenes.
        """
        try:
            self.model.predictor.trackers[0].reset()
            self.logger.info("Tracker state reset (scene cut)")
        except (AttributeError, IndexError):
            # Tracker may not be initialized yet
            self.logger.debug("Tracker reset skipped (not yet initialized)")
