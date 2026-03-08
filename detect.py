"""
Helmet Detection - Detection Script
Run inference using the trained YOLOv8 model on images, videos, or webcam.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Helmet Detection Inference")
    parser.add_argument("--source", type=str, default="0",
                        help="Image/video path, or '0' for webcam")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to runs/detect/")
    parser.add_argument("--show", action="store_true", default=True,
                        help="Show results in a window")
    args = parser.parse_args()

    # Load the trained model
    model = YOLO(args.model)

    # Run detection
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        show=args.show,
        device=0,
        stream=True,  # Stream results for video/webcam
    )

    # Process results (needed when stream=True)
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls]
                print(f"Detected: {label} ({conf:.2f})")


if __name__ == "__main__":
    main()
