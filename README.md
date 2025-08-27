# Driver Activity Detection (Face + Drowsiness cues)

This project detects a driver's face and classifies activity state (Active vs Inactive/Drowsy) using MediaPipe Face Mesh and OpenCV. It computes eye aspect ratio (EAR), mouth opening (MAR proxy), and applies temporal smoothing.

## Features
- Real-time face detection with MediaPipe Face Mesh
- EAR for eye openness, mouth opening proxy for yawning
- Simple classifier with hysteresis and cooldown
- Webcam or video file input, on-frame overlays

## Quickstart

```bash
# From project root
python -m venv .venv
. .venv/Scripts/Activate.ps1  # on PowerShell
pip install -r drowsness_detection/requirements.txt
python drowsness_detection/main.py --source 0
```

Use `--source path/to/video.mp4` for a video file. Press `q` to quit.

## Safety and limitations
- This is NOT a safety-certified system. Use as a demo only.
- Lighting, camera angle, occlusions, glasses, and camera quality affect accuracy.
- Tune thresholds in `config.py` for your environment and camera.

## Project layout
- `config.py`: thresholds and smoothing parameters
- `detector.py`: face mesh wrapper and feature extraction
- `classifier.py`: activity classification and smoothing
- `viz.py`: drawing utilities
- `main.py`: CLI entrypoint

## Configuration
Edit `drowsness_detection/config.py` to adjust thresholds:
- `EAR_SLEEP_THRESHOLD`, `EAR_FRAMES_SLEEP`
- `MOUTH_OPEN_THRESHOLD`, `MOUTH_FRAMES_YAWN`
- `SMOOTHING_WINDOW`, `STATUS_COOLDOWN_FRAMES`

## Troubleshooting
- If the camera index `0` fails, try `--source 1` or `2`.
- On Windows, use the PowerShell activation command shown above.
