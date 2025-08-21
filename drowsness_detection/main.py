from __future__ import annotations

import argparse
from typing import Union

import cv2

from drowsness_detection.classifier import ActivityClassifier
from drowsness_detection.detector import FaceFeatureExtractor
from drowsness_detection.viz import draw_overlays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver activity detection (Active vs Inactive)")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g., 0) or path to video file",
    )
    return parser.parse_args()


def open_capture(source_arg: str) -> cv2.VideoCapture:
    source: Union[int, str]
    if source_arg.isdigit():
        source = int(source_arg)
    else:
        source = source_arg
    cap = cv2.VideoCapture(source)
    return cap


def main() -> None:
    args = parse_args()
    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    extractor = FaceFeatureExtractor()
    classifier = ActivityClassifier()

    window_name = "Driver Activity"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            features = extractor.process(frame)
            if features is None:
                cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
            else:
                state = classifier.update(features.ear, features.mouth_open)
                annotated = draw_overlays(frame, features.landmarks_2d, features.ear, features.mouth_open, state.status, state.reason)
                cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


