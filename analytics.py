"""
Analytics Module — Advanced Features Engine
================================================

Implements the mandatory advanced features:
1. Trajectory Visualization — path lines per tracked ID
2. Player Counting — unique + active ID tracking
3. Movement Heatmap — spatial density accumulation
4. Speed Estimation — approximate velocity in m/s
5. Frame-wise Statistics — per-frame metrics collection

Also provides:
- ID stability metrics (proxy for MOTA)
- JSON export of complete tracking statistics
- Final summary statistics for the technical report
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from tracker import TrackResult


class AnalyticsEngine:
    """Central analytics engine that consumes tracking results per-frame
    and produces advanced analytical outputs.

    Usage:
        engine = AnalyticsEngine(frame_shape=(720, 1280), fps=30.0)

        # Per frame:
        engine.update(tracks, frame_idx)

        # After processing:
        engine.save_heatmap("heatmap.png")
        engine.save_statistics("stats.json")
        summary = engine.get_summary()
    """

    def __init__(self, frame_shape: Tuple[int, int],
                 fps: float = 30.0):
        """
        Args:
            frame_shape: (height, width) of the video frames.
            fps: Video frame rate (used for speed estimation).
        """
        self.logger = logging.getLogger("pipeline.analytics")
        self.frame_h, self.frame_w = frame_shape
        self.fps = fps

        # ── Trajectory storage ──
        # Dict of track_id -> list of (cx, cy, frame_idx)
        self.trajectories: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

        # ── Player counting ──
        self.unique_ids: set = set()
        self.active_ids_per_frame: List[int] = []
        self.total_detections: int = 0

        # ── Heatmap ──
        self.heatmap_h, self.heatmap_w = config.HEATMAP_RESOLUTION
        self.heatmap = np.zeros((self.heatmap_h, self.heatmap_w), dtype=np.float64)
        self.heatmap_decay = config.HEATMAP_DECAY

        # ── Speed estimation ──
        # Stores last N positions per ID for velocity computation
        self._speed_buffer: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
        self._speed_cache: Dict[int, float] = {}  # Cached speed per ID

        # ── Frame-wise statistics ──
        self.frame_stats: List[dict] = []

        # ── ID stability tracking ──
        self._prev_frame_ids: set = set()
        self.id_switches: int = 0  # Approximate count based on ID set changes
        self.total_frames_processed: int = 0

        # ── Scene cuts ──
        self.scene_cuts: List[int] = []

    def update(self, tracks: List[TrackResult], frame_idx: int,
               is_scene_cut: bool = False):
        """Process tracking results for a single frame.

        This is the core update loop, called once per processed frame.
        Updates all analytics subsystems simultaneously.

        Args:
            tracks: List of TrackResult from tracker.
            frame_idx: Current frame index (0-based).
            is_scene_cut: Whether a scene cut was detected this frame.
        """
        self.total_frames_processed += 1
        current_ids = set()
        confidences = []

        for track in tracks:
            tid = track.track_id
            cx, cy = track.center
            current_ids.add(tid)
            self.unique_ids.add(tid)
            confidences.append(track.confidence)

            # ── Update trajectory ──
            self.trajectories[tid].append((cx, cy, frame_idx))

            # ── Update heatmap ──
            # Map frame coordinates to heatmap grid coordinates
            hx = int(cx / self.frame_w * self.heatmap_w)
            hy = int(cy / self.frame_h * self.heatmap_h)
            hx = max(0, min(hx, self.heatmap_w - 1))
            hy = max(0, min(hy, self.heatmap_h - 1))
            self.heatmap[hy, hx] += 1.0

            # ── Update speed buffer ──
            self._speed_buffer[tid].append((cx, cy, frame_idx))
            if len(self._speed_buffer[tid]) > config.SPEED_SMOOTHING_WINDOW * 2:
                self._speed_buffer[tid] = self._speed_buffer[tid][-config.SPEED_SMOOTHING_WINDOW * 2:]

            # ── Compute speed ──
            self._speed_cache[tid] = self._compute_speed(tid)

        # ── Apply heatmap temporal decay ──
        self.heatmap *= self.heatmap_decay

        # ── Track active player counts ──
        self.active_ids_per_frame.append(len(current_ids))
        self.total_detections += len(tracks)

        # ── ID stability: detect switches ──
        # A rough heuristic: if IDs appear/disappear without scene cut,
        # some may be switches rather than genuine entries/exits
        if not is_scene_cut and self._prev_frame_ids:
            lost = self._prev_frame_ids - current_ids
            gained = current_ids - self._prev_frame_ids
            # Heuristic: if same number lost and gained in one frame, likely switches
            probable_switches = min(len(lost), len(gained))
            if probable_switches > 0 and len(lost) <= 3:
                self.id_switches += probable_switches
        self._prev_frame_ids = current_ids

        # ── Scene cut tracking ──
        if is_scene_cut:
            self.scene_cuts.append(frame_idx)

        # ── Record frame statistics ──
        self.frame_stats.append({
            "frame": frame_idx,
            "num_detections": len(tracks),
            "active_track_ids": sorted(current_ids),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "is_scene_cut": is_scene_cut,
        })

    def _compute_speed(self, track_id: int) -> float:
        """Compute approximate speed for a track in m/s.

        Uses pixel displacement over the smoothing window, then converts
        to approximate real-world speed using the field calibration estimate.

        Note: This is an approximation. True speed requires camera
        calibration and homography transformation to field coordinates.

        Args:
            track_id: Target track ID.

        Returns:
            Estimated speed in m/s (0.0 if insufficient data).
        """
        buffer = self._speed_buffer.get(track_id, [])
        window = config.SPEED_SMOOTHING_WINDOW

        if len(buffer) < 2:
            return 0.0

        recent = buffer[-window:]
        if len(recent) < 2:
            return 0.0

        # Total pixel displacement
        total_pixels = 0.0
        total_frames = 0
        for i in range(1, len(recent)):
            x1, y1, _ = recent[i - 1]
            x2, y2, _ = recent[i]
            total_pixels += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_frames += 1

        if total_frames == 0:
            return 0.0

        # Pixels per frame → pixels per second → meters per second
        pixels_per_frame = total_pixels / total_frames
        pixels_per_second = pixels_per_frame * self.fps
        meters_per_second = pixels_per_second / config.PIXELS_PER_METER

        # Clamp to realistic human running speed (0 - 12 m/s ≈ 43 km/h)
        return min(meters_per_second, 12.0)

    def get_speed(self, track_id: int) -> float:
        """Get cached speed for a track ID in m/s."""
        return self._speed_cache.get(track_id, 0.0)

    def get_speed_kmh(self, track_id: int) -> float:
        """Get cached speed for a track ID in km/h."""
        return self.get_speed(track_id) * 3.6

    def get_trajectory(self, track_id: int,
                       length: Optional[int] = None) -> List[Tuple[int, int]]:
        """Get trajectory points for a specific track.

        Args:
            track_id: Target track ID.
            length: Max points to return (None = all).

        Returns:
            List of (cx, cy) positions, chronologically ordered.
        """
        points = self.trajectories.get(track_id, [])
        if length:
            points = points[-length:]
        return [(p[0], p[1]) for p in points]

    def render_heatmap(self, frame: np.ndarray,
                       alpha: float = config.HEATMAP_ALPHA) -> np.ndarray:
        """Render the accumulated heatmap as a semi-transparent overlay.

        Args:
            frame: BGR frame to overlay on.
            alpha: Blend transparency (0 = invisible, 1 = opaque).

        Returns:
            Frame with heatmap overlay.
        """
        if self.heatmap.max() == 0:
            return frame

        # Normalize heatmap to [0, 255]
        heatmap_norm = self.heatmap.copy()
        heatmap_norm = (heatmap_norm / heatmap_norm.max() * 255).astype(np.uint8)

        # Apply Gaussian blur for smoother visualization
        heatmap_norm = cv2.GaussianBlur(heatmap_norm, (15, 15), 0)

        # Resize to frame dimensions
        heatmap_resized = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_CUBIC)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, config.HEATMAP_COLORMAP)

        # Blend with frame
        # Only overlay where heatmap has values (mask low areas)
        mask = heatmap_resized > 10  # Threshold to avoid noise
        output = frame.copy()
        output[mask] = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)[mask]

        return output

    def save_heatmap(self, output_path: Optional[str] = None):
        """Save the accumulated heatmap as a standalone image.

        Args:
            output_path: Path to save heatmap image (default from config).
        """
        path = Path(output_path or config.HEATMAP_OUTPUT)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.heatmap.max() == 0:
            self.logger.warning("Heatmap is empty — nothing to save")
            return

        # High-res heatmap rendering
        heatmap_norm = self.heatmap.copy()
        heatmap_norm = (heatmap_norm / heatmap_norm.max() * 255).astype(np.uint8)
        heatmap_norm = cv2.GaussianBlur(heatmap_norm, (15, 15), 0)
        heatmap_large = cv2.resize(heatmap_norm, (self.frame_w, self.frame_h),
                                    interpolation=cv2.INTER_CUBIC)
        heatmap_colored = cv2.applyColorMap(heatmap_large, config.HEATMAP_COLORMAP)

        cv2.imwrite(str(path), heatmap_colored)
        self.logger.info(f"Movement heatmap saved: {path}")

    def save_statistics(self, output_path: Optional[str] = None):
        """Save complete tracking statistics to JSON.

        Includes:
        - Per-frame statistics (detection counts, active IDs, confidence)
        - Summary metrics (unique players, avg detections, ID switches)
        - Trajectory data (reduced to key points for efficiency)

        Args:
            output_path: Path to save JSON file (default from config).
        """
        path = Path(output_path or config.STATISTICS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_summary()

        output = {
            "summary": summary,
            "frame_statistics": self.frame_stats,
            "scene_cuts": self.scene_cuts,
        }

        with open(str(path), 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)

        self.logger.info(f"Tracking statistics saved: {path}")

    def get_summary(self) -> dict:
        """Compute and return a summary of all analytics.

        Returns:
            Dictionary with key metrics for reporting.
        """
        avg_active = (np.mean(self.active_ids_per_frame)
                      if self.active_ids_per_frame else 0)
        max_active = max(self.active_ids_per_frame) if self.active_ids_per_frame else 0

        # Average speed across all tracked players
        speeds = [s for s in self._speed_cache.values() if s > 0.1]
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0

        return {
            "total_frames_processed": self.total_frames_processed,
            "unique_players_detected": len(self.unique_ids),
            "total_detections": self.total_detections,
            "avg_active_players_per_frame": round(float(avg_active), 1),
            "max_simultaneous_players": max_active,
            "estimated_id_switches": self.id_switches,
            "scene_cuts_detected": len(self.scene_cuts),
            "avg_player_speed_ms": round(float(avg_speed), 2),
            "max_player_speed_ms": round(float(max_speed), 2),
            "avg_player_speed_kmh": round(float(avg_speed * 3.6), 1),
            "max_player_speed_kmh": round(float(max_speed * 3.6), 1),
        }

    def print_summary(self):
        """Print a formatted summary to console and log."""
        summary = self.get_summary()
        lines = [
            "",
            "━" * 60,
            "  📊 TRACKING ANALYTICS SUMMARY",
            "━" * 60,
            f"  Total frames processed:    {summary['total_frames_processed']}",
            f"  Unique players detected:   {summary['unique_players_detected']}",
            f"  Total detections:          {summary['total_detections']}",
            f"  Avg active players/frame:  {summary['avg_active_players_per_frame']}",
            f"  Max simultaneous players:  {summary['max_simultaneous_players']}",
            f"  Estimated ID switches:     {summary['estimated_id_switches']}",
            f"  Scene cuts detected:       {summary['scene_cuts_detected']}",
            f"  Avg player speed:          {summary['avg_player_speed_kmh']} km/h",
            f"  Max player speed:          {summary['max_player_speed_kmh']} km/h",
            "━" * 60,
        ]

        for line in lines:
            print(line)
            self.logger.info(line.strip())
