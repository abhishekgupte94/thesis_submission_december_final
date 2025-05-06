import os
import gc
import cv2
import torch
import face_alignment
import numpy as np
import torch.multiprocessing as mp
from pathlib import Path
import time  # ensure this is imported at the top level

class VideoPreprocessor_FANET:
    """
    Distributed video lip-extraction using FaceAlignment and OpenCV.
    Saves lip-only videos to a common output directory across all GPUs.
    """
    def __init__(
        self,
        batch_size: int ,
        output_base_dir: str = None,
        device: str = 'cuda'
    ):
        self.batch_size = batch_size
        self.output_base_dir = output_base_dir
        self.device = device

        os.makedirs(self.output_base_dir, exist_ok=True)

        # Initialize FaceAlignment once per worker
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self.device,
            face_detector='sfd',
            flip_input=False
        )

        torch.backends.cudnn.benchmark = True

    def __call__(self, video_paths: list[str]) -> list[str]:
        """
        Process a list of video paths sequentially in this process,
        returning the list of lip-only output paths.
        """
        processed = []
        for path in video_paths:
            out = self.process_video(path)
            if out:
                processed.append(out)
        return processed

    def process_video(self, video_path: str) -> str:
        """
        Reads a video, extracts lip regions in batches, writes a new video file of lip crops,
        and returns the output video path.
        """
        if not os.path.exists(video_path):
            print(f"❌ Video path does not exist: {video_path}")

        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)

        out_path = os.path.join(self.output_base_dir, f"{video_name}_lips_only.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
        if not out.isOpened():
            raise Exception("❌ VideoWriter failed to open. Check codec and file path.")

        buffer = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            buffer.append(frame)
            if len(buffer) >= self.batch_size:
                self._process_batch(buffer, out)
                buffer.clear()

        if buffer:
            self._process_batch(buffer, out)

        cap.release()
        out.release()
        del cap, out, buffer
        gc.collect()
        torch.cuda.empty_cache()

        return out_path

    def _process_batch(self, frame_batch, out_writer):
        """
        Given a list of BGR frames, extracts lip crops and writes each to out_writer.
        """
        landmarks_batch = self.fa.get_landmarks_from_batch(frame_batch)
        for frame, landmarks in zip(frame_batch, landmarks_batch):
            try:
                lip_crop, (x_min, y_min, x_max, y_max) = self.extract_lip_segment(frame, landmarks)
                frame[y_min:y_max, x_min:x_max] = lip_crop
                out_writer.write(frame)
            except Exception as e:
                print(f"⚠️ Lip extraction error: {e}")
        del frame_batch, landmarks_batch
        gc.collect()
        torch.cuda.empty_cache()

    def extract_lip_segment(self, frame, landmarks):
        """Extract lip region based on mouth landmarks"""
        lip_landmarks = landmarks[48:]
        x_coords = lip_landmarks[:, 0].astype(int)
        y_coords = lip_landmarks[:, 1].astype(int)
        x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
        y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)
        if x_max <= x_min or y_max <= y_min:
            # invalid crop, return original frame region
            return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
        lip_crop = frame[y_min:y_max, x_min:x_max]
        return lip_crop, (x_min, y_min, x_max, y_max)

    def parallel_main(self, video_paths: list[str]) -> list[str]:
        """
        Extract lip-only videos for all paths in parallel across GPUs,
        saving outputs under `output_base_dir` and returning their paths.
        """
        world_size = torch.cuda.device_count() or 1
        manager = mp.Manager()
        return_dict = manager.dict()
        chunks = [video_paths[i::world_size] for i in range(world_size)]
        mp.spawn(
            worker_process,
            args=(chunks, self.batch_size, self.output_base_dir, return_dict),
            nprocs=world_size,
            join=True
        )

        all_outputs = []
        for rank in range(world_size):
            all_outputs.extend(return_dict.get(rank, []))

        print(f"✅ Extracted {len(all_outputs)} lip-only videos to '{self.output_base_dir}'.")
        return all_outputs

# Worker process must be module-level for mp.spawn


def worker_process(rank, chunks, batch_size, output_dir, return_dict):
    torch.cuda.set_device(rank)
    device_str = f'cuda:{rank}'
    processor = VideoPreprocessor_FANET(
        batch_size=batch_size,
        output_base_dir=output_dir,
        device=device_str
    )

    assigned_videos = chunks[rank]
    num_videos = len(assigned_videos)

    print(f"[GPU {rank}] Starting {num_videos} videos.")

    start_time = time.time()
    processed_videos = []

    for idx, video_path in enumerate(assigned_videos):
        output = processor.process_video(video_path)
        if output:
            processed_videos.append(output)

        if (idx + 1) % 10 == 0 or (idx + 1) == num_videos:
            elapsed = time.time() - start_time
            avg_time_per_video = elapsed / (idx + 1)
            eta_seconds = avg_time_per_video * (num_videos - idx - 1)
            print(f"[GPU {rank}] {idx + 1}/{num_videos} videos done. ETA: {eta_seconds/60:.2f} minutes.")

    return_dict[rank] = processed_videos

    del processor
    gc.collect()
    torch.cuda.empty_cache()
