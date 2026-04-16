"""
Video Download Utility — YouTube Video Acquisition
======================================================

Downloads and trims the specified FIFA World Cup 2018 video
(Portugal vs Spain) using yt-dlp and ffmpeg.

Video Source:
    Title:  Portugal v Spain | 2018 FIFA World Cup
    URL:    https://www.youtube.com/watch?v=OFbyNU6UQQs
    Match:  Group B — Portugal 3-3 Spain (Ronaldo hat-trick)

The script:
1. Downloads the video at ≤720p using yt-dlp (saves bandwidth & VRAM)
2. Trims to a configurable segment using ffmpeg (default: 60s-150s)
3. Selects a segment with dense player interaction and camera motion

Requirements:
    - yt-dlp installed (pip install yt-dlp)
    - ffmpeg installed and in PATH
"""

import logging
import subprocess
import sys
from pathlib import Path

import config

logger = logging.getLogger("pipeline.downloader")


def _refresh_path_windows():
    """Refresh PATH from the Windows registry so newly installed tools are found
    without requiring a shell restart."""
    import os
    import platform
    if platform.system() == "Windows":
        try:
            machine = os.environ.get("Path", "")
            # Read fresh values from registry
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                machine = winreg.QueryValueEx(key, "Path")[0]
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                user = winreg.QueryValueEx(key, "Path")[0]
            os.environ["PATH"] = machine + ";" + user
        except Exception:
            pass  # Non-critical — fall through to existing PATH


def check_dependency(command: str) -> bool:
    """Check if a command-line tool is available in PATH.

    On Windows, also refreshes PATH from registry to detect
    tools installed in the current session (e.g., via winget).

    Args:
        command: Tool name (e.g., 'ffmpeg', 'yt-dlp').

    Returns:
        True if the tool is available.
    """
    # First attempt with current PATH
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Refresh PATH (Windows: picks up winget/choco installs without shell restart)
    _refresh_path_windows()

    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_video(url: str = config.VIDEO_URL,
                   output_path: Path = config.INPUT_VIDEO,
                   max_height: int = 720) -> Path:
    """Download video from YouTube using yt-dlp.

    Downloads at ≤720p to balance quality with processing efficiency.
    720p is sufficient for person detection and provides good FPS
    on RTX 3050 (6GB VRAM).

    Args:
        url: YouTube video URL.
        output_path: Path to save downloaded video.
        max_height: Maximum video height in pixels.

    Returns:
        Path to downloaded video file.

    Raises:
        RuntimeError: If download fails.
    """
    if output_path.exists():
        logger.info(f"Video already downloaded: {output_path}")
        return output_path

    if not check_dependency("yt-dlp"):
        raise RuntimeError(
            "yt-dlp not found. Install via: pip install yt-dlp"
        )

    logger.info(f"Downloading video from: {url}")
    logger.info(f"Max resolution: {max_height}p")
    logger.info(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-playlist",
        "--retries", "3",
        "--no-warnings",
        url,
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )

        if result.returncode != 0:
            logger.error(f"yt-dlp stderr: {result.stderr}")
            raise RuntimeError(f"yt-dlp failed with code {result.returncode}")

        if not output_path.exists():
            # yt-dlp sometimes adds extension — check for variations
            possible = list(output_path.parent.glob(f"{output_path.stem}*"))
            if possible:
                actual = possible[0]
                actual.rename(output_path)
                logger.info(f"Renamed {actual.name} → {output_path.name}")
            else:
                raise RuntimeError("Download completed but file not found")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Download complete: {output_path.name} ({file_size_mb:.1f} MB)")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Download timed out (>10 minutes)")


def _trim_with_opencv(input_path: Path, output_path: Path,
                      start_sec: int, duration_sec: int) -> Path:
    """Fallback: trim video using OpenCV when ffmpeg is not available.

    Slower than ffmpeg stream-copy but requires no external tools.

    Args:
        input_path: Source video path.
        output_path: Output clip path.
        start_sec: Start time in seconds.
        duration_sec: Duration in seconds.

    Returns:
        Path to trimmed clip.
    """
    import cv2

    logger.info("Using OpenCV fallback for video trimming (ffmpeg not found)")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)
    end_frame = min(end_frame, total_frames)

    logger.info(f"  FPS: {fps}, Resolution: {width}x{height}")
    logger.info(f"  Frames: {start_frame} → {end_frame} ({end_frame - start_frame} frames)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0
    target_frames = end_frame - start_frame

    while frames_written < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frames_written += 1

        if frames_written % 300 == 0:
            logger.info(f"  Trimming progress: {frames_written}/{target_frames} frames")

    cap.release()
    writer.release()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Clip created (OpenCV): {output_path.name} "
                f"({file_size_mb:.1f} MB, {frames_written} frames)")
    return output_path


def trim_video(input_path: Path = config.INPUT_VIDEO,
               output_path: Path = config.CLIP_VIDEO,
               start_sec: int = config.CLIP_START,
               duration_sec: int = config.CLIP_DURATION) -> Path:
    """Trim video to a specific time segment.

    Strategy (in order of preference):
    1. ffmpeg stream copy (instant, lossless)
    2. ffmpeg re-encode (slower but reliable)
    3. OpenCV fallback (no external tools needed)

    Args:
        input_path: Path to source video.
        output_path: Path for trimmed clip.
        start_sec: Start time in seconds.
        duration_sec: Duration in seconds.

    Returns:
        Path to trimmed video clip.
    """
    if output_path.exists():
        logger.info(f"Clip already exists: {output_path}")
        return output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Source video not found: {input_path}")

    # ── Try ffmpeg first ──
    if check_dependency("ffmpeg"):
        return _trim_with_ffmpeg(input_path, output_path, start_sec, duration_sec)

    # ── Fallback to OpenCV ──
    logger.warning("ffmpeg not found — using OpenCV fallback for trimming")
    print("  ⚠ ffmpeg not found — using OpenCV-based trimming (slower)")
    return _trim_with_opencv(input_path, output_path, start_sec, duration_sec)


def _trim_with_ffmpeg(input_path: Path, output_path: Path,
                      start_sec: int, duration_sec: int) -> Path:
    """Trim video using ffmpeg (preferred method).

    Args:
        input_path: Source video path.
        output_path: Output clip path.
        start_sec: Start time in seconds.
        duration_sec: Duration in seconds.

    Returns:
        Path to trimmed clip.
    """
    # Format time as HH:MM:SS
    start_str = f"{start_sec // 3600:02d}:{(start_sec % 3600) // 60:02d}:{start_sec % 60:02d}"

    logger.info(f"Trimming video (ffmpeg): {input_path.name}")
    logger.info(f"  Start: {start_str} | Duration: {duration_sec}s")
    logger.info(f"  Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Method 1: Stream copy (fast, lossless)
    cmd = [
        "ffmpeg",
        "-y",                           # Overwrite output
        "-ss", start_str,               # Seek to start (before -i for fast seek)
        "-i", str(input_path),          # Input file
        "-t", str(duration_sec),        # Duration
        "-c", "copy",                   # Stream copy (no re-encode)
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.warning("Stream copy failed, trying re-encode...")
            # Method 2: Re-encode (slower but more reliable)
            cmd_reencode = [
                "ffmpeg",
                "-y",
                "-ss", start_str,
                "-i", str(input_path),
                "-t", str(duration_sec),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                str(output_path),
            ]
            result = subprocess.run(
                cmd_reencode,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"ffmpeg stderr: {result.stderr}")
                raise RuntimeError(f"ffmpeg failed with code {result.returncode}")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Clip created: {output_path.name} ({file_size_mb:.1f} MB)")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Video trimming timed out")


def download_and_prepare(url: str = config.VIDEO_URL,
                          start_sec: int = config.CLIP_START,
                          duration_sec: int = config.CLIP_DURATION,
                          skip_download: bool = False) -> Path:
    """Complete video preparation pipeline: download + trim.

    Args:
        url: YouTube video URL.
        start_sec: Clip start time in seconds.
        duration_sec: Clip duration in seconds.
        skip_download: If True, skip downloading (use existing file).

    Returns:
        Path to the ready-to-process video clip.
    """
    print("\n" + "=" * 60)
    print("  🎥 VIDEO PREPARATION")
    print("=" * 60)

    # Step 1: Download
    if not skip_download:
        print("\n  [1/2] Downloading video from YouTube...")
        download_video(url)
    else:
        if not config.INPUT_VIDEO.exists():
            logger.warning("--skip-download specified but source video not found")
            print("  ⚠ Source video not found — attempting download anyway...")
            download_video(url)
        else:
            print("  [1/2] Skipping download (--skip-download)")

    # Step 2: Trim
    print("\n  [2/2] Trimming to clip...")
    clip_path = trim_video(
        start_sec=start_sec,
        duration_sec=duration_sec,
    )

    print(f"\n  ✓ Video ready: {clip_path}")
    print("=" * 60 + "\n")

    return clip_path


# ──────────────────────────────────────────────────────────────
# Standalone execution
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Run video download + trim as a standalone utility."""
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    import argparse
    parser = argparse.ArgumentParser(description="Download and trim video clip")
    parser.add_argument("--url", default=config.VIDEO_URL, help="YouTube URL")
    parser.add_argument("--start", type=int, default=config.CLIP_START, help="Start time (sec)")
    parser.add_argument("--duration", type=int, default=config.CLIP_DURATION, help="Duration (sec)")
    args = parser.parse_args()

    try:
        clip = download_and_prepare(
            url=args.url,
            start_sec=args.start,
            duration_sec=args.duration,
        )
        print(f"\n  ✅ Ready to process: {clip}")
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        sys.exit(1)
