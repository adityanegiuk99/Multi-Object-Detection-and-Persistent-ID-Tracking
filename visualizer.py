"""
Visualizer Module — Frame Annotation & Rendering Engine
==========================================================

Handles all visual output rendering:
- Colored bounding boxes with unique colors per persistent ID
- ID labels with confidence scores
- Trajectory trail lines
- Statistics overlay panel
- Heatmap overlay (via AnalyticsEngine)
- Speed annotation per player

Design Decisions:
    - Colors use golden-ratio hue spacing for maximum visual distinctness
    - Labels use contrasting text for readability on any background
    - Stats panel is semi-transparent to avoid obstructing video content
    - Trail length is configurable for performance/visual balance
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from analytics import AnalyticsEngine
from tracker import TrackResult
from utils import id_to_color, get_contrasting_text_color


class Visualizer:
    """Rendering engine for annotated output video frames.

    Composes multiple visual layers on each frame:
    1. Bounding boxes + ID labels (per-track coloring)
    2. Trajectory trails (polylines showing recent movement)
    3. Stats panel (player count, FPS, frame info)
    4. Optional heatmap overlay

    Usage:
        viz = Visualizer(frame_shape=(720, 1280))
        annotated = viz.render(frame, tracks, analytics, fps=30.0)
    """

    def __init__(self, frame_shape: Tuple[int, int]):
        """
        Args:
            frame_shape: (height, width) of the video frames.
        """
        self.logger = logging.getLogger("pipeline.visualizer")
        self.frame_h, self.frame_w = frame_shape

    def render(self, frame: np.ndarray,
               tracks: List[TrackResult],
               analytics: AnalyticsEngine,
               fps: float = 0.0,
               frame_idx: int = 0,
               total_frames: int = 0,
               show_heatmap: bool = False) -> np.ndarray:
        """Render all visual annotations on a single frame.

        Args:
            frame: Input BGR frame (will be copied, not modified).
            tracks: Current frame's tracking results.
            analytics: AnalyticsEngine instance for trajectory/heatmap data.
            fps: Current processing FPS for display.
            frame_idx: Current frame number.
            total_frames: Total frames in video (for progress).
            show_heatmap: Whether to overlay the movement heatmap.

        Returns:
            Annotated BGR frame.
        """
        output = frame.copy()

        # Layer 1: Heatmap overlay (bottom layer)
        if show_heatmap:
            output = analytics.render_heatmap(output)

        # Layer 2: Trajectory trails
        if config.SHOW_TRAJECTORY:
            output = self._draw_trajectories(output, tracks, analytics)

        # Layer 3: Bounding boxes + labels
        output = self._draw_detections(output, tracks, analytics)

        # Layer 4: Statistics panel
        if config.SHOW_STATS_PANEL:
            output = self._draw_stats_panel(
                output, tracks, analytics, fps, frame_idx, total_frames
            )

        return output

    def _draw_detections(self, frame: np.ndarray,
                         tracks: List[TrackResult],
                         analytics: AnalyticsEngine) -> np.ndarray:
        """Draw bounding boxes and ID labels for each tracked player.

        Each player gets a unique color based on their persistent track ID.
        Labels show: ID number, confidence score, and optional speed.

        Args:
            frame: BGR frame to annotate.
            tracks: Current frame's tracking results.
            analytics: AnalyticsEngine for speed data.

        Returns:
            Annotated frame.
        """
        for track in tracks:
            color = id_to_color(track.track_id)
            x1, y1, x2, y2 = track.bbox

            # ── Bounding Box ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

            # ── Corner accents (visual polish) ──
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 3:
                t = config.BBOX_THICKNESS + 1
                # Top-left
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
                # Top-right
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
                # Bottom-left
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
                # Bottom-right
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

            # ── Label ──
            label_parts = [f"ID:{track.track_id}"]
            if config.SHOW_CONFIDENCE:
                label_parts.append(f"{track.confidence:.2f}")

            # Add speed if available
            speed = analytics.get_speed_kmh(track.track_id)
            if speed > 0.5:
                label_parts.append(f"{speed:.0f}km/h")

            label = " | ".join(label_parts)

            # Label background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
            )
            label_y = max(y1 - 8, text_h + 4)
            label_x = x1

            # Draw label background
            cv2.rectangle(
                frame,
                (label_x, label_y - text_h - 6),
                (label_x + text_w + 8, label_y + 2),
                color, -1  # Filled
            )

            # Draw label text
            text_color = get_contrasting_text_color(color)
            cv2.putText(
                frame, label,
                (label_x + 4, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, text_color, config.FONT_THICKNESS,
                cv2.LINE_AA
            )

            # ── Center dot ──
            cx, cy = track.center
            cv2.circle(frame, (cx, cy), 3, color, -1)

        return frame

    def _draw_trajectories(self, frame: np.ndarray,
                           tracks: List[TrackResult],
                           analytics: AnalyticsEngine) -> np.ndarray:
        """Draw trajectory trail lines for active tracks.

        Uses anti-aliased polylines with gradual alpha fade for
        a polished, professional appearance.

        Args:
            frame: BGR frame to annotate.
            tracks: Current frame's tracking results.
            analytics: AnalyticsEngine for trajectory data.

        Returns:
            Annotated frame.
        """
        for track in tracks:
            trail = analytics.get_trajectory(track.track_id, config.TRAIL_LENGTH)
            if len(trail) < 2:
                continue

            color = id_to_color(track.track_id)

            # Draw trail as connected line segments with varying thickness
            points = np.array(trail, dtype=np.int32)

            # Gradient effect: older segments are thinner and more transparent
            num_segments = len(points) - 1
            for i in range(num_segments):
                # Progress: 0.0 (oldest) → 1.0 (newest)
                progress = i / max(num_segments - 1, 1)
                thickness = max(1, int(1 + progress * 3))

                # Fade color towards dark for older positions
                fade = 0.3 + 0.7 * progress
                faded_color = tuple(int(c * fade) for c in color)

                cv2.line(
                    frame,
                    tuple(points[i]),
                    tuple(points[i + 1]),
                    faded_color,
                    thickness,
                    cv2.LINE_AA
                )

        return frame

    def _draw_stats_panel(self, frame: np.ndarray,
                          tracks: List[TrackResult],
                          analytics: AnalyticsEngine,
                          fps: float,
                          frame_idx: int,
                          total_frames: int) -> np.ndarray:
        """Draw a semi-transparent statistics overlay panel.

        Displays:
        - Current active player count
        - Total unique players detected
        - Processing FPS
        - Frame progress
        - Estimated ID switches

        Args:
            frame: BGR frame to annotate.
            tracks: Current frame's tracking results.
            analytics: AnalyticsEngine for statistics.
            fps: Current processing FPS.
            frame_idx: Current frame number.
            total_frames: Total video frames.

        Returns:
            Annotated frame.
        """
        # Panel dimensions and position (top-left)
        panel_w = 320
        panel_h = 180
        margin = 10
        x1, y1 = margin, margin
        x2, y2 = x1 + panel_w, y1 + panel_h

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Panel border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # Header
        cv2.putText(frame, "TRACKING DASHBOARD",
                    (x1 + 10, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

        # Separator line
        cv2.line(frame, (x1 + 10, y1 + 35), (x2 - 10, y1 + 35), (80, 80, 80), 1)

        # Stats lines
        stats_lines = [
            (f"Active Players:  {len(tracks)}", (100, 255, 100)),
            (f"Unique IDs:      {analytics.get_summary()['unique_players_detected']}",
             (100, 200, 255)),
            (f"ID Switches:     ~{analytics.id_switches}", (100, 180, 255)),
            (f"FPS:             {fps:.1f}", (255, 255, 100)),
            (f"Frame:           {frame_idx + 1}/{total_frames}", (200, 200, 200)),
        ]

        y_offset = y1 + 55
        for text, color in stats_lines:
            cv2.putText(frame, text,
                        (x1 + 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            y_offset += 25

        # Progress bar
        if total_frames > 0:
            bar_x1 = x1 + 15
            bar_x2 = x2 - 15
            bar_y = y2 - 15
            bar_h = 6
            progress = frame_idx / max(total_frames - 1, 1)

            # Background
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h),
                          (60, 60, 60), -1)
            # Fill
            fill_x = int(bar_x1 + (bar_x2 - bar_x1) * progress)
            cv2.rectangle(frame, (bar_x1, bar_y), (fill_x, bar_y + bar_h),
                          (0, 200, 255), -1)

        return frame

    def create_comparison_frame(self,
                                frame_bt: np.ndarray,
                                frame_ds: np.ndarray,
                                label_bt: str = "ByteTrack",
                                label_ds: str = "DeepSORT") -> np.ndarray:
        """Create a side-by-side comparison frame for ByteTrack vs DeepSORT.

        Used in --compare mode to generate a split-screen output video.

        Args:
            frame_bt: ByteTrack-annotated frame.
            frame_ds: DeepSORT-annotated frame.
            label_bt: Label for left panel.
            label_ds: Label for right panel.

        Returns:
            Combined side-by-side frame.
        """
        h = max(frame_bt.shape[0], frame_ds.shape[0])
        w1, w2 = frame_bt.shape[1], frame_ds.shape[1]

        # Resize to same height if different
        if frame_bt.shape[0] != h:
            scale = h / frame_bt.shape[0]
            frame_bt = cv2.resize(frame_bt, (int(w1 * scale), h))
        if frame_ds.shape[0] != h:
            scale = h / frame_ds.shape[0]
            frame_ds = cv2.resize(frame_ds, (int(w2 * scale), h))

        # Concatenate side by side
        combined = np.hstack([frame_bt, frame_ds])

        # Add divider line
        div_x = frame_bt.shape[1]
        cv2.line(combined, (div_x, 0), (div_x, h), (255, 255, 255), 2)

        # Add labels
        for label, x_pos in [(label_bt, 10), (label_ds, div_x + 10)]:
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(combined, (x_pos, 5), (x_pos + tw + 10, th + 15),
                          (0, 0, 0), -1)
            cv2.putText(combined, label, (x_pos + 5, th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        return combined
