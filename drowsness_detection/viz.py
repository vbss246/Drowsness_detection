from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

import drowsness_detection.config as cfg


def _draw_points(frame: np.ndarray, landmarks: np.ndarray, indices: Iterable[int], color: Tuple[int, int, int]) -> None:
    for idx in indices:
        x, y = landmarks[idx]
        cv2.circle(frame, (int(x), int(y)), 1, color, -1, lineType=cv2.LINE_AA)


def draw_overlays(frame: np.ndarray, landmarks: np.ndarray, ear: float, mouth_open: float, status: str, reason: str) -> np.ndarray:
    out = frame
    # Draw eyes and mouth keypoints
    left_eye_idx = [cfg.LEFT_EYE_OUTER, cfg.LEFT_EYE_INNER, cfg.LEFT_EYE_UPPER, cfg.LEFT_EYE_LOWER]
    right_eye_idx = [cfg.RIGHT_EYE_OUTER, cfg.RIGHT_EYE_INNER, cfg.RIGHT_EYE_UPPER, cfg.RIGHT_EYE_LOWER]
    mouth_idx = [cfg.MOUTH_LEFT_CORNER, cfg.MOUTH_RIGHT_CORNER, cfg.MOUTH_UPPER_INNER, cfg.MOUTH_LOWER_INNER]

    _draw_points(out, landmarks, left_eye_idx, (0, 255, 0))
    _draw_points(out, landmarks, right_eye_idx, (0, 255, 0))
    _draw_points(out, landmarks, mouth_idx, (0, 255, 255))

    # Status text
    status_color = (0, 200, 0) if status == "Active" else (0, 0, 255)
    cv2.rectangle(out, (10, 10), (350, 95), (0, 0, 0), -1)
    cv2.putText(out, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"Reason: {reason}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Metrics box
    cv2.rectangle(out, (10, 100), (350, 160), (0, 0, 0), -1)
    cv2.putText(out, f"EAR: {ear:.3f} (thr {cfg.EAR_SLEEP_THRESHOLD:.2f})", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Mouth: {mouth_open:.3f} (thr {cfg.MOUTH_OPEN_THRESHOLD:.2f})", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return out


