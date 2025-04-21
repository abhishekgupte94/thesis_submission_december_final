import numpy as np
import cv2
import torch.nn as nn
import torch

class VideoPreprocessorAVDF:
    def __init__(self, frame_size=224):
        super().__init__()
        self.frame_interval = 0.04  # 40ms intervals as specified
        self.frame_size = frame_size

    def process_video(self, video_path):
        """
        Process video frames at 4ms intervals.
        Returns: frames xv ∈ R^(3×Tv×H×W)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = max(1, round(fps * self.frame_interval))

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0  # Normalize to [0, 1]
                frames.append(frame)

            frame_count += 1

        cap.release()

        if not frames:
            return None

        frames = np.stack(frames)  # [Tv, H, W, 3]
        frames = torch.from_numpy(frames).float()

        # Permute each frame: [H, W, 3] → [3, H, W]
        frames = frames.permute(0, 3, 1, 2)  # [Tv, 3, H, W]
        frames = frames.permute(1, 0, 2, 3)  # [3, Tv, H, W]

        return frames
