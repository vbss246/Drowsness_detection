"""Configuration for driver activity detection thresholds and smoothing.

Tune these values for your camera, lighting, and driver population.
"""

# Eye openness threshold (normalized vertical eyelid gap / eye width)
EAR_SLEEP_THRESHOLD: float = 0.22

# Number of consecutive frames below EAR threshold to mark as eye-closed event
EAR_FRAMES_SLEEP: int = 15

# Mouth opening threshold (normalized inner-lip gap / mouth width)
MOUTH_OPEN_THRESHOLD: float = 0.35

# Number of consecutive frames above mouth-open threshold to mark as yawn event
MOUTH_FRAMES_YAWN: int = 15

# Moving average window for smoothing raw EAR/Mouth
SMOOTHING_WINDOW: int = 5

# Cooldown frames after an inactive state before switching back to active
STATUS_COOLDOWN_FRAMES: int = 30

# MediaPipe Face Mesh landmark indices used
# Left eye landmarks
LEFT_EYE_OUTER: int = 33
LEFT_EYE_INNER: int = 133
LEFT_EYE_UPPER: int = 159
LEFT_EYE_LOWER: int = 145

# Right eye landmarks
RIGHT_EYE_OUTER: int = 362
RIGHT_EYE_INNER: int = 263
RIGHT_EYE_UPPER: int = 386
RIGHT_EYE_LOWER: int = 374

# Mouth landmarks (outer corners and inner upper/lower lip)
MOUTH_LEFT_CORNER: int = 78
MOUTH_RIGHT_CORNER: int = 308
MOUTH_UPPER_INNER: int = 13
MOUTH_LOWER_INNER: int = 14


