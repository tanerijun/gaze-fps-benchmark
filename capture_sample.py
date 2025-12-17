"""
Capture Sample Image for FPS Benchmark

A simple script to capture a frame from your webcam and save it as a sample image
for use with the fps-exp.py benchmark script.

Usage:
    python capture_sample.py                    # Capture from default webcam
    python capture_sample.py --output assets/sample.png
    python capture_sample.py --camera 1         # Use camera ID 1
"""

import argparse
import sys

import cv2


def capture_sample_image(camera_id: int = 0, output_path: str = "assets/sample.png"):
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        print("Available camera IDs to try: 0, 1, 2")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 360

    print("Camera opened successfully!")
    print("\nInstructions:")
    print("  - Position your face in front of the camera")
    print("  - Press SPACE to capture the image")
    print("  - Press ESC or Q to quit without capturing")
    print("\nShowing preview...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break

        display_frame = cv2.flip(frame, 1)

        cv2.putText(
            display_frame,
            "Press SPACE to capture, ESC/Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Capture Sample Image", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):  # Space key
            cv2.imwrite(output_path, display_frame)
            print(f"\n✓ Image captured and saved to: {output_path}")
            print(
                f"  Image size: {display_frame.shape[1]}x{display_frame.shape[0]} pixels"
            )
            break
        elif key == 27 or key == ord("q"):
            print("\nCapture cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    print("\nYou can now run the benchmark with:")
    print(f"  python fps-exp.py --image {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture a sample image from webcam for FPS benchmarking"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/sample.png",
        help="Output path for the captured image (default: assets/sample.png)",
    )

    args = parser.parse_args()

    capture_sample_image(camera_id=args.camera, output_path=args.output)


if __name__ == "__main__":
    main()
