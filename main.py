"""
Main Pipeline — Multi-Object Detection & Persistent ID Tracking
===================================================================

Entry point for the complete computer vision pipeline that:
1. Downloads and prepares the source video (Portugal vs Spain, WC 2018)
2. Runs YOLO11 person detection with GPU acceleration
3. Applies multi-object tracking (ByteTrack / BoT-SORT / DeepSORT)
4. Manages persistent IDs with occlusion recovery
5. Computes analytics (trajectory, heatmap, speed, counting)
6. Renders annotated output video with professional overlays
7. Exports statistics and evaluation metrics

Supports three modes of operation:
- Single tracker:   python main.py --tracker bytetrack
- Comparison mode:  python main.py --compare
- Custom input:     python main.py --input myvideo.mp4

Hardware Target: RTX 3050 (6GB VRAM) + i5-13420H + 24GB RAM
Expected Performance: ~25-35 FPS with YOLO11m @ 640px on CUDA

Author: Aditya Negi
Assignment: Multi-Object Detection and Persistent ID Tracking
Video Source: https://www.youtube.com/watch?v=OFbyNU6UQQs
"""

import json
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np

# Local modules
import config
from utils import (
    setup_logging, VideoReader, VideoWriter,
    SceneCutDetector, FPSCounter, format_time,
)
from detector import PersonDetector
from tracker import create_tracker, TrackHistoryManager
from analytics import AnalyticsEngine
from visualizer import Visualizer
from download_video import download_and_prepare


def run_single_tracker(args) -> dict:
    """Execute the pipeline with a single tracker configuration.

    This is the primary execution path. Processes the video frame-by-frame:
    Detection → Tracking → Analytics → Visualization → Output

    Args:
        args: Parsed CLI arguments (from config.parse_args()).

    Returns:
        Summary dictionary with pipeline results and metrics.
    """
    logger = logging.getLogger("pipeline.main")

    # ── Resolve input video path ──
    if args.input:
        input_path = args.input
    else:
        input_path = str(config.CLIP_VIDEO)

    # ── Resolve output video path ──
    if args.output:
        output_path = args.output
    else:
        tracker_name = args.tracker
        output_path = str(config.OUTPUT_DIR / f"tracked_{tracker_name}.mp4")

    # ══════════════════════════════════════════════════════════
    # Phase 1: Initialize Components
    # ══════════════════════════════════════════════════════════

    print("\n" + "═" * 60)
    print("  🔧 INITIALIZING PIPELINE")
    print("═" * 60)
    print(f"  Tracker:  {args.tracker.upper()}")
    print(f"  Model:    {args.model}")
    print(f"  Device:   {args.device}")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")

    # Video reader
    print("\n  Loading video...")
    reader = VideoReader(input_path, resize=config.INPUT_RESIZE)
    video_info = reader.get_info()
    logger.info(f"Video loaded: {json.dumps(video_info, indent=2)}")

    for k, v in video_info.items():
        print(f"    {k}: {v}")

    # Detector
    print("\n  Initializing detector...")
    detector = PersonDetector(
        model_name=args.model,
        device=args.device,
        conf_threshold=args.conf,
        img_size=args.imgsz,
        half=args.half,
    )

    # Tracker
    print(f"  Initializing {args.tracker.upper()} tracker...")
    tracker = create_tracker(args.tracker, detector)

    # Analytics engine
    analytics = AnalyticsEngine(
        frame_shape=(reader.eff_height, reader.eff_width),
        fps=reader.fps,
    )

    # Visualizer
    visualizer = Visualizer(
        frame_shape=(reader.eff_height, reader.eff_width),
    )

    # Scene cut detector
    scene_cut_detector = SceneCutDetector(threshold=config.SCENE_CUT_THRESHOLD)

    # FPS counter
    fps_counter = FPSCounter(smoothing=0.92)

    # Video writer
    output_fps = config.OUTPUT_FPS or reader.fps
    writer = VideoWriter(
        output_path,
        fps=output_fps,
        frame_size=(reader.eff_width, reader.eff_height),
    )

    # Track history (for trajectories)
    track_history = TrackHistoryManager()

    print("\n  ✓ All components initialized")
    print("═" * 60)

    # ══════════════════════════════════════════════════════════
    # Phase 2: Frame-by-Frame Processing
    # ══════════════════════════════════════════════════════════

    print("\n  🚀 PROCESSING VIDEO...")
    print(f"  Total frames: {reader.total_frames}")
    print(f"  Frame skip: {args.frame_skip}")
    effective_frames = reader.total_frames // (args.frame_skip + 1)
    print(f"  Effective frames to process: ~{effective_frames}")
    print()

    start_time = time.perf_counter()
    processed_count = 0
    skipped_count = 0

    try:
        for frame_idx, frame in reader:
            # ── Frame skipping ──
            if args.frame_skip > 0 and frame_idx % (args.frame_skip + 1) != 0:
                skipped_count += 1
                continue

            # ── Scene cut detection ──
            is_scene_cut = scene_cut_detector.check(frame)
            if is_scene_cut and config.SCENE_CUT_RESET_TRACKER:
                tracker.reset()
                track_history.reset()
                logger.warning(f"Frame {frame_idx}: Scene cut → tracker reset")

            # ── Tracking (detection + association) ──
            try:
                tracks = tracker.update(frame)
            except Exception as e:
                logger.error(f"Tracking error at frame {frame_idx}: {e}")
                tracks = []

            # ── Update analytics ──
            analytics.update(tracks, frame_idx, is_scene_cut)
            track_history.update(tracks, frame_idx)

            # ── FPS measurement ──
            current_fps = fps_counter.tick()

            # ── Visualization ──
            annotated_frame = visualizer.render(
                frame=frame,
                tracks=tracks,
                analytics=analytics,
                fps=current_fps,
                frame_idx=frame_idx,
                total_frames=reader.total_frames,
                show_heatmap=(not args.no_heatmap if hasattr(args, 'no_heatmap') else False),
            )

            # ── Write output frame ──
            writer.write(annotated_frame)
            processed_count += 1

            # ── Live preview (optional) ──
            if not args.no_display:
                # Resize for display if frame is too large
                display_frame = annotated_frame
                if display_frame.shape[1] > 1280:
                    scale = 1280 / display_frame.shape[1]
                    display_frame = cv2.resize(
                        display_frame,
                        (1280, int(display_frame.shape[0] * scale)),
                    )
                cv2.imshow("Multi-Object Tracking", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User pressed 'q' — stopping early")
                    print("\n  ⚠ Stopped by user (q pressed)")
                    break
                elif key == ord('h'):
                    # Toggle heatmap on 'h' key
                    if hasattr(args, 'no_heatmap'):
                        args.no_heatmap = not args.no_heatmap

            # ── Progress display ──
            if processed_count % 30 == 0 or frame_idx == reader.total_frames - 1:
                progress = (frame_idx + 1) / reader.total_frames * 100
                elapsed = time.perf_counter() - start_time
                eta = (elapsed / max(processed_count, 1)) * (effective_frames - processed_count)

                sys.stdout.write(
                    f"\r  [{progress:5.1f}%] Frame {frame_idx + 1}/{reader.total_frames} "
                    f"| Players: {len(tracks):2d} "
                    f"| FPS: {current_fps:5.1f} "
                    f"| Elapsed: {format_time(elapsed)} "
                    f"| ETA: {format_time(eta)}"
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user (Ctrl+C)")
        logger.info("Pipeline interrupted by user")

    finally:
        # ── Cleanup ──
        total_time = time.perf_counter() - start_time
        reader.release()
        writer.release()
        cv2.destroyAllWindows()

    # ══════════════════════════════════════════════════════════
    # Phase 3: Post-Processing & Reporting
    # ══════════════════════════════════════════════════════════

    print(f"\n\n  ⏱ Total processing time: {format_time(total_time)} "
          f"({processed_count / max(total_time, 0.01):.1f} avg FPS)")

    # Save analytics outputs
    if config.SAVE_STATISTICS:
        analytics.save_statistics()

    if not (hasattr(args, 'no_heatmap') and args.no_heatmap):
        analytics.save_heatmap()

    # Print summary
    analytics.print_summary()
    summary = analytics.get_summary()

    # Add pipeline metadata to summary
    summary["pipeline"] = {
        "tracker": args.tracker,
        "model": args.model,
        "device": detector.device,
        "half_precision": detector.half,
        "total_time_sec": round(total_time, 2),
        "avg_fps": round(processed_count / max(total_time, 0.01), 1),
        "frames_processed": processed_count,
        "frames_skipped": skipped_count,
        "output_video": output_path,
    }

    print(f"\n  📹 Output video: {output_path}")
    print(f"  📊 Statistics:   {config.STATISTICS_FILE}")
    print(f"  🔥 Heatmap:      {config.HEATMAP_OUTPUT}")

    return summary


def run_comparison(args) -> dict:
    """Execute ByteTrack vs DeepSORT comparison.

    Runs both trackers on the same video and generates:
    - Separate output videos for each tracker
    - Side-by-side comparison summary
    - Performance metrics for both

    Args:
        args: Parsed CLI arguments.

    Returns:
        Combined comparison results.
    """
    logger = logging.getLogger("pipeline.compare")

    print("\n" + "█" * 60)
    print("  🔁 TRACKER COMPARISON MODE")
    print("  ByteTrack vs DeepSORT")
    print("█" * 60)

    results = {}

    # ── Run ByteTrack ──
    print("\n" + "─" * 60)
    print("  [1/2] Running ByteTrack...")
    print("─" * 60)
    args.tracker = "bytetrack"
    args.output = str(config.OUTPUT_DIR / "tracked_bytetrack.mp4")
    results["bytetrack"] = run_single_tracker(args)

    # ── Run DeepSORT ──
    print("\n" + "─" * 60)
    print("  [2/2] Running DeepSORT...")
    print("─" * 60)
    args.tracker = "deepsort"
    args.output = str(config.OUTPUT_DIR / "tracked_deepsort.mp4")
    results["deepsort"] = run_single_tracker(args)

    # ══════════════════════════════════════════════════════════
    # Comparison Summary
    # ══════════════════════════════════════════════════════════

    print("\n\n" + "█" * 60)
    print("  📊 COMPARISON RESULTS")
    print("█" * 60)

    bt = results["bytetrack"]
    ds = results["deepsort"]

    comparison_table = f"""
    ┌─────────────────────────┬──────────────┬──────────────┐
    │ Metric                  │  ByteTrack   │   DeepSORT   │
    ├─────────────────────────┼──────────────┼──────────────┤
    │ Avg FPS                 │ {bt['pipeline']['avg_fps']:>10.1f}  │ {ds['pipeline']['avg_fps']:>10.1f}  │
    │ Unique Players          │ {bt['unique_players_detected']:>10}  │ {ds['unique_players_detected']:>10}  │
    │ Avg Active/Frame        │ {bt['avg_active_players_per_frame']:>10.1f}  │ {ds['avg_active_players_per_frame']:>10.1f}  │
    │ Est. ID Switches        │ {bt['estimated_id_switches']:>10}  │ {ds['estimated_id_switches']:>10}  │
    │ Scene Cuts              │ {bt['scene_cuts_detected']:>10}  │ {ds['scene_cuts_detected']:>10}  │
    │ Avg Speed (km/h)        │ {bt['avg_player_speed_kmh']:>10.1f}  │ {ds['avg_player_speed_kmh']:>10.1f}  │
    │ Processing Time (s)     │ {bt['pipeline']['total_time_sec']:>10.1f}  │ {ds['pipeline']['total_time_sec']:>10.1f}  │
    └─────────────────────────┴──────────────┴──────────────┘
    """
    print(comparison_table)

    # Save comparison results
    comparison_path = config.OUTPUT_DIR / "comparison_results.json"
    with open(str(comparison_path), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Comparison saved: {comparison_path}")

    # ── Conclusion ──
    bt_fps = bt['pipeline']['avg_fps']
    ds_fps = ds['pipeline']['avg_fps']
    bt_switches = bt['estimated_id_switches']
    ds_switches = ds['estimated_id_switches']

    print("\n  🏆 RECOMMENDATION:")
    if bt_fps > ds_fps * 1.2:
        print(f"  ByteTrack is {bt_fps/max(ds_fps,0.1):.1f}x faster than DeepSORT")
    if bt_switches <= ds_switches:
        print("  ByteTrack has equal or fewer ID switches")
        print("  → ByteTrack is recommended for this sports footage scenario")
    else:
        print("  DeepSORT shows fewer ID switches (better Re-ID)")
        print("  → Consider DeepSORT if ID consistency is paramount")

    print("█" * 60 + "\n")

    return results


def main():
    """Main entry point for the pipeline.

    Orchestrates the complete flow:
    1. Parse arguments
    2. Setup logging
    3. Download/prepare video
    4. Run tracking pipeline (single or comparison)
    5. Report results
    """
    # ── Parse CLI arguments ──
    args = config.parse_args()

    # ── Setup logging ──
    logger = setup_logging(level=config.LOG_LEVEL)

    # ── Banner ──
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║      Multi-Object Detection & Persistent ID Tracking    ║
    ║      ─────────────────────────────────────────────────   ║
    ║      YOLO11 + ByteTrack/DeepSORT Pipeline               ║
    ║      FIFA World Cup 2018: Portugal vs Spain             ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    logger.info("Pipeline started")
    logger.info(f"Arguments: {vars(args)}")

    # ── Phase 1: Video Preparation ──
    try:
        if args.input:
            # User provided custom input — skip download
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"  ❌ Input file not found: {args.input}")
                sys.exit(1)
            print(f"  Using custom input: {args.input}")
        else:
            # Auto-download and trim
            clip_path = download_and_prepare(
                url=config.VIDEO_URL,
                start_sec=args.clip_start,
                duration_sec=args.clip_duration,
                skip_download=args.skip_download,
            )
            # Verify clip exists
            if not clip_path.exists():
                print(f"  ❌ Clip file not found after preparation: {clip_path}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Video preparation failed: {e}")
        print(f"\n  ❌ Video preparation failed: {e}")
        print("  Tip: Use --input <path> to provide a local video file")
        print("  Tip: Ensure yt-dlp and ffmpeg are installed")
        sys.exit(1)

    # ── Phase 2: Run Pipeline ──
    try:
        if args.compare:
            results = run_comparison(args)
        else:
            results = run_single_tracker(args)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n  ❌ Pipeline error: {e}")
        sys.exit(1)

    # ── Done ──
    print("\n  ✅ Pipeline completed successfully!")
    print(f"  📂 Outputs: {config.OUTPUT_DIR}")
    logger.info("Pipeline completed successfully")

    return results


if __name__ == "__main__":
    main()
