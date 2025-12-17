"""
FPS Benchmark Script for Gaze Estimation Model

Usage:
    uv run fps-exp.py
"""

import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.se = nn.Identity()
        self.activation = nn.ReLU()
        self.reparam_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.se(self.reparam_conv(x)))


class MobileOneS1(nn.Module):
    def __init__(self):
        super().__init__()
        num_blocks = [2, 8, 10, 1]
        width_multipliers = [1.5, 1.5, 2.0, 2.5]
        self.in_planes = min(64, int(64 * width_multipliers[0]))

        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            use_se=False,
        )

        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks[0])
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks[1])
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks[2])
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks[3])

        self.out_channels = self.in_planes

    def _make_stage(
        self, planes: int, num_blocks: int, use_se: bool = False
    ) -> nn.Sequential:
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []

        for stride in strides:
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    use_se=use_se,
                )
            )
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    use_se=use_se,
                )
            )
            self.in_planes = planes

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class GazeHead(nn.Module):
    def __init__(self, in_channels: int, num_bins: int = 90, dropout_rate: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_pitch = nn.Linear(in_channels, num_bins)
        self.fc_yaw = nn.Linear(in_channels, num_bins)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        pitch_logits = self.fc_pitch(x)
        yaw_logits = self.fc_yaw(x)
        return pitch_logits, yaw_logits


class GazeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileOneS1()
        self.head = GazeHead(in_channels=self.backbone.out_channels, num_bins=90)

    def forward(self, x):
        features = self.backbone(x)
        pitch_logits, yaw_logits = self.head(features)
        return pitch_logits, yaw_logits


class GazePipeline:
    def __init__(
        self,
        weights_path: str,
        face_detector_path: str,
        device: str = "auto",
        image_size: int = 224,
    ):
        self.image_size = image_size
        self.device = self._setup_device(device)
        self.num_bins = 90
        self.bin_width = 360.0 / self.num_bins

        # Setup face detector
        self._setup_face_detector(face_detector_path)

        # Setup gaze model
        self.model: nn.Module = GazeModel()
        self._load_model_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

        # JIT compile for faster inference
        self._compile_model()

        print(f"Gaze pipeline initialized on device: {self.device}")

    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _setup_face_detector(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face detector model not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        print("Face detector initialized")

    def _load_model_weights(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        print(f"Model weights loaded from {weights_path}")

    def _compile_model(self):
        try:
            dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(
                self.device
            )
            # Type ignore to avoid type checker issues with JIT
            self.model = torch.jit.trace(self.model, dummy_input)  # type: ignore
            print("Model successfully JIT compiled with torch.jit.trace")
        except Exception as e:
            print(f"JIT compilation failed: {e}")
            print("Continuing with standard PyTorch model")

    def _decode_predictions(self, predictions):
        pitch_logits, yaw_logits = predictions

        # Get bin predictions
        pitch_bins = torch.argmax(pitch_logits, dim=1).cpu().numpy()
        yaw_bins = torch.argmax(yaw_logits, dim=1).cpu().numpy()

        # Convert bins to angles (center of bin)
        pitch_angles = (pitch_bins * self.bin_width) - 180.0 + (self.bin_width / 2.0)
        yaw_angles = (yaw_bins * self.bin_width) - 180.0 + (self.bin_width / 2.0)

        return pitch_angles, yaw_angles

    def __call__(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.face_detector.detect(mp_image)
        detections = detection_result.detections

        if not detections:
            return []

        results = []

        for detection in detections:
            bbox = detection.bounding_box
            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = int(x1 + bbox.width)
            y2 = int(y1 + bbox.height)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop = cv2.resize(face_crop, (self.image_size, self.image_size))

            face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
            # Normalize using ImageNet mean and std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            face_tensor = (face_tensor - mean) / std
            face_tensor = face_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(face_tensor)

            pitch_angles, yaw_angles = self._decode_predictions(predictions)

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "gaze": {
                        "pitch": float(pitch_angles[0]),
                        "yaw": float(yaw_angles[0]),
                    },
                }
            )

        return results


def run_benchmark(pipeline: GazePipeline, frame: np.ndarray, num_frames: int = 100):
    print(f"\n{'=' * 60}")
    print(f"Starting benchmark with {num_frames} iterations...")
    print(f"{'=' * 60}")

    # Warmup (run a few times to ensure GPU is ready)
    print("Warming up...")
    for i in range(25):
        _ = pipeline(frame)
    print("Warmup complete.\n")

    # Actual benchmark
    start_time = time.monotonic()
    face_counts = []

    for i in range(num_frames):
        results = pipeline(frame)
        face_counts.append(len(results))
        print(f"Processing frame {i + 1}/{num_frames}...", end="\r")

    duration = time.monotonic() - start_time
    fps = num_frames / duration

    print(f"\n\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Total frames processed: {num_frames}")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average time per frame: {(duration / num_frames) * 1000:.2f} ms")
    print(f"Faces detected per frame: {face_counts[0]} (from first frame)")
    print(f"Device: {pipeline.device}")
    print(f"{'=' * 60}\n")

    return {
        "fps": fps,
        "total_frames": num_frames,
        "duration": duration,
        "avg_frame_time_ms": (duration / num_frames) * 1000,
        "device": str(pipeline.device),
        "faces_detected": face_counts[0],
    }


def main():
    weights_path = "assets/model.pth"
    detector_path = "assets/blaze_face_short_range.tflite"
    image_path = "assets/sample.png"
    num_frames = 200
    device = "cpu"

    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}")
        print("Please place your trained model weights in the assets/ directory")
        sys.exit(1)

    if not os.path.exists(detector_path):
        print(f"Error: Face detector model not found at {detector_path}")
        print("Please ensure blaze_face_short_range.tflite is in the assets/ directory")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: Sample image not found at {image_path}")
        print("Please place a sample image (e.g., sample.png) in the assets/ directory")
        sys.exit(1)

    print(f"Loading image from {image_path}...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Failed to load image from {image_path}")
        sys.exit(1)

    print(f"Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels")

    print("\nInitializing gaze estimation pipeline...")
    pipeline = GazePipeline(
        weights_path=weights_path,
        face_detector_path=detector_path,
        device=device,
    )

    _ = run_benchmark(pipeline, frame, num_frames=num_frames)

    print("Running final inference to display gaze prediction...")
    final_results = pipeline(frame)
    if final_results:
        print("\nSample Gaze Prediction:")
        for i, result in enumerate(final_results):
            print(f"  Face {i + 1}:")
            print(f"    Bounding Box: {result['bbox']}")
            print(f"    Pitch: {result['gaze']['pitch']:.2f}°")
            print(f"    Yaw: {result['gaze']['yaw']:.2f}°")
    else:
        print("\nNo faces detected in the sample image.")

    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    main()
