# 🎬 Loom Video Script — Multi-Object Detection & Persistent ID Tracking

**Total Duration**: 4–5 minutes
**Format**: Loom screen recording with face cam
**Tip**: Keep Loom face cam in bottom-left corner, share full screen

---

## 🟢 INTRO — Who Am I & What Did I Build [0:00 – 0:25]

**[Screen: GitHub repo page, README visible]**

> "Hey! I'm Aditya Negi, and in this video I'll walk you through my Multi-Object Detection and Persistent ID Tracking pipeline."
>
> "This is a computer vision system that detects and tracks football players in live FIFA World Cup footage — assigning each player a unique, persistent ID that survives occlusion, camera cuts, and fast motion."
>
> "Let me show you what it does and how it works."

---

## 🟢 THE PROBLEM [0:25 – 0:55]

**[Screen: Open the raw clip in a video player — `input/clip_90s.mp4`]**
*(If clip isn't available, show the YouTube video briefly)*

> "Here's the raw footage — Portugal versus Spain from the 2018 World Cup."
>
> "Now, tracking players sounds simple, but look at the challenges:"
>
> **[Point/circle with mouse as you mention each]**
>
> "First — **occlusion**. Players constantly overlap and block each other."
>
> "Second — **same jerseys**. All teammates look identical to a computer."
>
> "Third — the **camera never stops moving** — it pans, zooms, and cuts to replays."
>
> "And fourth — **scale variation** — some players are 200 pixels tall, others are barely 30."
>
> "Traditional appearance-based trackers completely fail here. My solution handles all of these."

---

## 🟢 THE SOLUTION — Architecture [0:55 – 1:40]

**[Screen: Scroll to the Architecture section in README.md]**

> "Here's the pipeline architecture. It's modular — six components, each with a single responsibility."
>
> **[Point to each module as you mention it]**
>
> "Step one — **Video Acquisition**. The pipeline automatically downloads the match from YouTube using yt-dlp, then trims a 90-second clip using ffmpeg. If ffmpeg isn't installed, it falls back to OpenCV — so it works everywhere."
>
> "Step two — **Detection**. I'm using **YOLO11m** — not YOLOv8 — YOLO **11**. It has 23% fewer parameters, lower VRAM usage, and a new attention module called C2PSA that specifically helps in crowded scenes."
>
> "Step three — **Tracking**. This is where it gets interesting."

---

## 🟢 KEY DESIGN DECISION — Why ByteTrack [1:40 – 2:20]

**[Screen: Scroll to Design Decisions table in README]**

> "I chose **ByteTrack** over DeepSORT, and here's the counterintuitive reason."
>
> "DeepSORT uses a neural network to extract appearance features — what a person *looks like* — to match them across frames. That's great for pedestrians who all wear different clothes."
>
> "But in football? All teammates wear the **exact same jersey**. The appearance model can't tell them apart — it actually adds *noise* to the matching and causes *more* ID switches."
>
> "ByteTrack uses **pure motion** — Kalman filter prediction plus IoU matching. No appearance, no confusion."
>
> "But the real killer feature is **two-stage association**:"
>
> "Stage one matches high-confidence detections. Stage two takes the *unmatched* tracks and tries to rescue them using low-confidence detections — the partially occluded players that other trackers would just lose."
>
> "This is what makes it robust under real-world occlusion."

---

## 🟢 LIVE DEMO — Running the Pipeline [2:20 – 3:20]

**[Screen: Switch to terminal in VS Code]**

> "Let me run it live."

**[Type and run]:**
```
python main.py --skip-download --no-display
```

> "I'm using `--skip-download` since I already have the video, and `--no-display` for cleaner recording."
>
> **[Wait for initialization to show]**
>
> "You can see it initializes YOLO11m, sets up ByteTrack, and starts processing frames."
>
> "Look at the live stats — it's detecting 10 to 16 players per frame, running at about 3 to 5 FPS on CPU. On a CUDA GPU like the RTX 3050, this would be 25 to 35 FPS — near real-time."

**[Let it run for ~15 seconds showing progress, then switch to output]**

> "Let me show you the output."

**[Screen: Open `outputs/tracked_bytetrack.mp4` in a video player]**

> "Each player has a **unique colored bounding box** with a persistent ID number."
>
> **[Point to specific things as the video plays]**
>
> "Watch this player move — the ID stays consistent. See the **trajectory trails** following each player's movement path."
>
> "And in the top corner, there's a live dashboard showing active players, unique IDs, and FPS."
>
> **[If a scene cut is visible]**
> "Notice when the camera cuts to a replay — the system detects the scene change using histogram analysis and resets all tracks. This prevents IDs from bleeding across unrelated scenes."

---

## 🟢 ADVANCED ANALYTICS [3:20 – 3:50]

**[Screen: Show `outputs/movement_heatmap.png`]**

> "Beyond tracking, the system computes **five advanced analytics**:"
>
> "One — **trajectory visualization** with gradient-fade trails."
>
> "Two — **player counting** — both unique IDs and active count per frame."
>
> "Three — this **movement heatmap** showing where players spend the most time. You can clearly see the midfield concentration."
>
> "Four — **speed estimation** in kilometers per hour for each player."

**[Screen: Open `outputs/tracking_statistics.json` briefly]**

> "And five — **frame-wise statistics** exported as JSON for downstream analysis."

---

## 🟢 RESULTS & NUMBERS [3:50 – 4:15]

**[Screen: Show the analytics summary from terminal or README]**

> "Here are the final numbers from our test run:"
>
> "**1,420 frames** processed from a 90-second clip"
>
> "**10.9 average players** detected per frame, with a max of **18 simultaneous**"
>
> "And all of this with a clean, modular codebase — eight Python files, full CLI with `--help`, and comprehensive documentation."

**[Screen: Show the GitHub repo — scroll through files briefly]**

> "The entire pipeline runs with a single command — `python main.py`. It downloads the video, downloads the model, processes everything, and generates the output. Zero manual steps."

---

## 🟢 WHAT MAKES THIS STAND OUT [4:15 – 4:40]

**[Screen: GitHub README — scroll to Edge Case Handling table]**

> "Three things make this production-quality:"
>
> "**One** — it's not just detect-and-track. It handles edge cases: scene cuts reset the tracker, false positives are filtered by area and aspect ratio, and occlusion is handled by ByteTrack's two-stage rescue."
>
> "**Two** — it's hardware-aware. It auto-detects CUDA, enables FP16 for double throughput, and gracefully falls back to CPU."
>
> "**Three** — it includes a full comparison mode. Run `python main.py --compare` and it benchmarks ByteTrack against DeepSORT on the same clip, with a side-by-side results table."

---

## 🟢 CLOSING [4:40 – 5:00]

**[Screen: GitHub repo main page]**

> "So to summarize — YOLO11m for detection, ByteTrack for tracking, with scene cut handling, five analytics features, and a one-command pipeline that works out of the box."
>
> "The code, documentation, and technical report are all in the GitHub repo linked below."
>
> "Thanks for watching!"

---

# 📝 Pre-Recording Checklist

Before hitting record on Loom:

- [ ] **Terminal ready**: `cd` to the project directory
- [ ] **Video player ready**: Have the output video file accessible (run the pipeline once before recording if needed)
- [ ] **GitHub repo open**: Have the repo page loaded in a browser tab
- [ ] **Clean desktop**: Close unnecessary windows
- [ ] **Dark theme**: Use VS Code dark theme + dark terminal for professional look
- [ ] **Font size**: Increase terminal and editor font to 16-18px so Loom captures it clearly
- [ ] **Tab order**: Arrange tabs in presentation order: GitHub → Terminal → Video Player → Heatmap

# 🎯 Key Points to Emphasize

1. **"YOLO11, not YOLOv8"** — Shows you're using the latest, not just the popular one
2. **"ByteTrack over DeepSORT because of same jerseys"** — Shows deep understanding, not just plugging libraries
3. **"Two-stage association for occlusion rescue"** — The technical differentiator
4. **"One command to run everything"** — Production quality, not a notebook hack
5. **"Scene cut detection resets tracker"** — Shows you thought about real-world edge cases

# ⏱ Timing Summary

| Section | Duration | Cumulative |
|---------|----------|------------|
| Intro | 25s | 0:25 |
| Problem | 30s | 0:55 |
| Architecture | 45s | 1:40 |
| ByteTrack Decision | 40s | 2:20 |
| Live Demo | 60s | 3:20 |
| Analytics | 30s | 3:50 |
| Results | 25s | 4:15 |
| Standout Points | 25s | 4:40 |
| Closing | 20s | 5:00 |
