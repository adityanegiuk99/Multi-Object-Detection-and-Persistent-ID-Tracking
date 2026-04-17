# Demo Video Script — Multi-Object Detection & Persistent ID Tracking

**Duration**: 3–5 minutes
**Format**: Screen recording with narration
**Target Audience**: Technical evaluators (beginner-friendly but technically strong)

---

## SLIDE 1: Title & Introduction [0:00 – 0:30]

**[Show: Project title slide / terminal with project banner]**

> "Hi, I'm Aditya Negi. Today I'll walk you through my Multi-Object Detection and Persistent ID Tracking pipeline — a production-quality computer vision system that detects and tracks football players in real FIFA World Cup footage."
>
> "The system uses YOLO11 for detection and ByteTrack for persistent ID assignment, and handles real-world challenges like occlusion, camera motion, and players wearing identical jerseys."

---

## SLIDE 2: Problem Statement & Challenges [0:30 – 1:00]

**[Show: Raw video clip — highlight problem areas with annotations]**

> "Let me show you the actual video we're working with — Portugal vs Spain from the 2018 World Cup."
>
> "Why is tracking players hard? Look at these challenges:"
> - **[Point to cluster]** "Multiple players overlapping — causing occlusion"
> - **[Point to camera pan]** "Camera is constantly panning — objects move in frame even when standing still"
> - **[Point to same-jersey players]** "These players wear identical jerseys — traditional appearance models can't tell them apart"
> - **[Point to distant players]** "Scale variation — some players are 200 pixels tall, others are just 30"
>
> "My pipeline handles all of these. Let me show you how."

---

## SLIDE 3: Architecture Overview [1:00 – 1:45]

**[Show: Architecture diagram from README]**

> "Here's the pipeline architecture. It's a modular system with six main components:"
>
> 1. "**Video acquisition**: yt-dlp downloads the video, ffmpeg trims a 90-second clip"
> 2. "**Detection**: YOLO11m detects all persons with GPU-accelerated inference"
> 3. "**Tracking**: ByteTrack associates detections across frames with persistent IDs"
> 4. "**Analytics**: Calculates trajectories, heatmaps, speed, and player counts"
> 5. "**Visualization**: Renders annotated frames with colored bounding boxes and trails"
> 6. "**Output**: Saves annotated video, statistics JSON, and heatmap image"
>
> "Everything is configurable through a central config file and CLI arguments."

---

## SLIDE 4: Why YOLO11 + ByteTrack? [1:45 – 2:30]

**[Show: Comparison table from README]**

> "A critical design decision: why YOLO11m over YOLOv8m?"
>
> "YOLO11m has 20 million parameters versus YOLOv8m's 26 million — that's 23% fewer parameters. On our RTX 3050 with just 6 gigs of VRAM, that matters. It also has a C2PSA attention module that specifically helps with crowded scenes."
>
> "For tracking, I chose ByteTrack over DeepSORT, and here's the counterintuitive reason:"
>
> "DeepSORT uses an appearance model — a CNN that extracts visual features to recognize people. Sounds great, right? But in football, all teammates wear the SAME jersey. The appearance model can't distinguish them. It actually adds noise to the matching process."
>
> "ByteTrack uses pure motion — Kalman filter prediction plus IoU matching. It's simpler, faster, and more robust for this exact scenario."
>
> "ByteTrack also has a killer feature: two-stage association. High-confidence detections get matched first, then low-confidence detections rescue tracks that would otherwise be lost during occlusion."

---

## SLIDE 5: Live Demo — Running the Pipeline [2:30 – 3:30]

**[Show: Terminal — run the pipeline]**

> "Let me run this live. I'll execute: `python main.py`"
>
> **[Show pipeline running]**
>
> "You can see it's initializing YOLO11m on CUDA, loading the video, and now processing frames."
>
> "Notice the FPS counter — we're getting about 28 frames per second on the RTX 3050. That's near real-time."
>
> **[Switch to output video playing]**
>
> "Here's the output. Each player has a unique colored bounding box with a persistent ID. Watch as this player" **[point]** "moves behind another — the ID stays the same."
>
> "The trajectory trails show recent movement paths. The dashboard in the corner shows active players, unique IDs, and FPS."
>
> **[Show a scene cut moment]**
>
> "Notice when the camera cuts to a replay — the system detects the scene change and resets all tracks. This prevents IDs from one scene bleeding into another."

---

## SLIDE 6: Advanced Features [3:30 – 4:15]

**[Show: Output files and heatmap]**

> "Beyond basic tracking, the system computes five advanced analytics:"
>
> 1. **[Show trajectory trails]** "Trajectory visualization — gradient-faded path lines for each player"
> 2. **[Show player count]** "Player counting — unique IDs across the entire clip, plus per-frame active count"
> 3. **[Show heatmap image]** "Movement heatmap — you can see the high-activity areas of the pitch"
> 4. **[Show speed label]** "Speed estimation — approximate player velocity in km/h"
> 5. **[Show JSON file]** "Frame-wise statistics — exported as JSON for downstream analysis"
>
> "The heatmap is particularly interesting — you can clearly see the midfield concentration and attacking movements."

---

## SLIDE 7: Edge Cases & Comparison [4:15 – 4:45]

**[Show: Comparison mode results — if run with --compare]**

> "I also built a comparison mode: `python main.py --compare`"
>
> "This runs both ByteTrack and DeepSORT on the same clip. The results confirm our hypothesis — ByteTrack is about 40% faster and shows fewer ID switches in this sports scenario."
>
> "Key edge cases handled:"
> - "Scene cuts detected via histogram analysis — tracker resets automatically"
> - "Small/false detections filtered by minimum area and aspect ratio"
> - "Occlusion handled by keeping lost tracks for 30 frames"

---

## SLIDE 8: Conclusion & Future Work [4:45 – 5:00]

**[Show: Summary slide]**

> "To summarize: YOLO11m + ByteTrack achieves robust, real-time player tracking at 25-35 FPS on consumer hardware. The modular architecture makes it easy to swap models or trackers."
>
> "Future improvements would include sport-specific Re-ID models, homography-based speed estimation, and team classification using jersey colors."
>
> "Thank you for watching. The full source code, documentation, and technical report are included in the submission."

---

## Recording Tips

1. **Screen setup**: Use a resolution of 1920×1080, dark terminal theme
2. **Recording tool**: OBS Studio (free) or PowerPoint screen recording
3. **Audio**: Use a quiet room, speak clearly and at moderate pace
4. **Demo sections**: Pre-run the pipeline once to download models and video, so the live demo is smooth
5. **Time management**: Practice once to hit the 3-5 minute target
