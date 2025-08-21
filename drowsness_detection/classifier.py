from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import drowsness_detection.config as cfg


@dataclass
class ActivityState:
    status: str  # "Active" or "Inactive"
    reason: str


class ActivityClassifier:
    def __init__(self) -> None:
        self._ear_history: Deque[float] = deque(maxlen=cfg.SMOOTHING_WINDOW)
        self._mouth_history: Deque[float] = deque(maxlen=cfg.SMOOTHING_WINDOW)
        self._closed_eyes_frames: int = 0
        self._yawn_frames: int = 0
        self._cooldown_frames_remaining: int = 0
        self._last_status: str = "Active"

    def _smoothed(self, values: Deque[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def update(self, ear: float, mouth_open: float) -> ActivityState:
        # Smooth inputs
        self._ear_history.append(ear)
        self._mouth_history.append(mouth_open)
        ear_smooth = self._smoothed(self._ear_history)
        mouth_smooth = self._smoothed(self._mouth_history)

        # Update consecutive frame counters
        if ear_smooth < cfg.EAR_SLEEP_THRESHOLD:
            self._closed_eyes_frames += 1
        else:
            self._closed_eyes_frames = 0

        if mouth_smooth > cfg.MOUTH_OPEN_THRESHOLD:
            self._yawn_frames += 1
        else:
            self._yawn_frames = 0

        # Determine state
        reason = ""
        current_status = "Active"

        if self._closed_eyes_frames >= cfg.EAR_FRAMES_SLEEP:
            current_status = "Inactive"
            reason = "Eyes closed"
        elif self._yawn_frames >= cfg.MOUTH_FRAMES_YAWN:
            current_status = "Inactive"
            reason = "Yawning"
        else:
            current_status = "Active"
            reason = "Eyes open"

        # Apply cooldown/hysteresis to reduce flapping
        if current_status == "Inactive":
            self._cooldown_frames_remaining = cfg.STATUS_COOLDOWN_FRAMES
        else:
            if self._cooldown_frames_remaining > 0:
                # remain inactive during cooldown
                current_status = "Inactive"
                reason = f"Cooldown ({reason})"
                self._cooldown_frames_remaining -= 1

        self._last_status = current_status
        return ActivityState(status=current_status, reason=reason)


