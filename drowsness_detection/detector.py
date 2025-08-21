from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import drowsness_detection.config as cfg


@dataclass
class FrameFeatures:
    ear: float
    mouth_open: float
    landmarks_2d: np.ndarray  # shape (468, 2), pixel coordinates


class FaceFeatureExtractor:
    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._face_mesh.close()

    def _extract_landmarks(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        height, width = image_bgr.shape[:2]
        coords = []
        for landmark in face_landmarks.landmark:
            x_px = int(landmark.x * width)
            y_px = int(landmark.y * height)
            coords.append((x_px, y_px))
        return np.asarray(coords, dtype=np.float32)

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)))

    def _compute_ear(self, lm: np.ndarray) -> float:
        # Use vertical eyelid distance normalized by eye width for each eye, then average
        left_vertical = self._euclidean_distance(lm[cfg.LEFT_EYE_UPPER], lm[cfg.LEFT_EYE_LOWER])
        left_width = self._euclidean_distance(lm[cfg.LEFT_EYE_OUTER], lm[cfg.LEFT_EYE_INNER])
        right_vertical = self._euclidean_distance(lm[cfg.RIGHT_EYE_UPPER], lm[cfg.RIGHT_EYE_LOWER])
        right_width = self._euclidean_distance(lm[cfg.RIGHT_EYE_OUTER], lm[cfg.RIGHT_EYE_INNER])

        # Avoid division by zero
        left_ratio = left_vertical / left_width if left_width > 1e-6 else 0.0
        right_ratio = right_vertical / right_width if right_width > 1e-6 else 0.0
        return float((left_ratio + right_ratio) / 2.0)

    def _compute_mouth_open(self, lm: np.ndarray) -> float:
        vertical = self._euclidean_distance(lm[cfg.MOUTH_UPPER_INNER], lm[cfg.MOUTH_LOWER_INNER])
        width = self._euclidean_distance(lm[cfg.MOUTH_LEFT_CORNER], lm[cfg.MOUTH_RIGHT_CORNER])
        return float(vertical / width) if width > 1e-6 else 0.0

    def process(self, frame_bgr: np.ndarray) -> Optional[FrameFeatures]:
        landmarks_2d = self._extract_landmarks(frame_bgr)
        if landmarks_2d is None:
            return None
        ear = self._compute_ear(landmarks_2d)
        mouth_open = self._compute_mouth_open(landmarks_2d)
        return FrameFeatures(ear=ear, mouth_open=mouth_open, landmarks_2d=landmarks_2d)


