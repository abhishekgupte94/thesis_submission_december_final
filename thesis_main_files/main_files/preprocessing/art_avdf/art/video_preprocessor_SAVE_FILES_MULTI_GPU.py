import os
import gc
import cv2
import torch
import face_alignment
import numpy as np
import torch.multiprocessing as mp
from pathlib import Path
import time


class VideoPreprocessor_FANET:
    """
    Distributed video lip-extraction using FaceAlignment.
    Decouples GPU landmark detection and disk video writing by saving crops as .npy files.
    """

    def __init__(self, batch_size: int, output_base_dir: str = None, device: str = 'cuda', rank: int = 0):
        self.batch_size = batch_size
        self.output_base_dir = output_base_dir
        self.device = device
        self.rank = rank

        os.makedirs(self.output_base_dir, exist_ok=True)

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self.device,
            face_detector='sfd',
            flip_input=False
        )

        torch.backends.cudnn.benchmark = True

    def __call__(self, video_paths: list[str]) -> list[str]:
        processed = []
        for path in video_paths:
            out = self.process_video(path)
            if out:
                processed.append(out)
        return processed

    def process_video(self, video_path: str) -> str:
        if not os.path.exists(video_path):
            print(f"❌ Video path does not exist: {video_path}")
            return None

        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return None

        out_path = os.path.join(self.output_base_dir, f"{video_name}_lips_only.npy")
        print(f"[INFO] [GPU {self.rank}] Saving intermediate output to {out_path}")

        buffer = []
        all_crops = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            buffer.append(frame)
            if len(buffer) >= self.batch_size:
                crops = self._process_batch(buffer)
                all_crops.extend(crops)
                buffer.clear()

        if buffer:
            crops = self._process_batch(buffer)
            all_crops.extend(crops)

        cap.release()

        if all_crops:
            np.save(out_path, np.array(all_crops, dtype=np.uint8))
            print(f"[GPU {self.rank}] Saved {len(all_crops)} crops to {out_path}")
        else:
            print(f"[GPU {self.rank}] No valid crops extracted from {video_path}")
            return None

        del cap, buffer, all_crops
        gc.collect()
        torch.cuda.empty_cache()

        return out_path

    def _process_batch(self, frame_batch):
        crops = []
        try:
            frame_batch_tensor = torch.stack(
                [torch.from_numpy(frame).permute(2, 0, 1) for frame in frame_batch],
                dim=0
            ).float().to(self.device)

            landmarks_batch = self.fa.get_landmarks_from_batch(frame_batch_tensor)

            for frame_tensor, landmarks in zip(frame_batch_tensor, landmarks_batch):
                try:
                    if landmarks is None:
                        print(f"⚠️ Skipping frame: no face detected.")
                        continue

                    frame = frame_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    lip_crop, _ = self.extract_lip_segment(frame, landmarks)

                    if lip_crop is None:
                        print(f"⚠️ Skipping frame: invalid lip crop.")
                        continue

                    resized_crop = cv2.resize(lip_crop, (224, 224), interpolation=cv2.INTER_CUBIC)

                    if not isinstance(resized_crop, np.ndarray) or resized_crop.ndim != 3 or resized_crop.dtype != np.uint8:
                        print(f"⚠️ Invalid frame for saving. Skipping.")
                        continue

                    crops.append(resized_crop)

                except Exception as e:
                    print(f"⚠️ Error during frame processing: {e}")

        except Exception as e:
            print(f"❌ Batch processing failed: {e}")

        finally:
            del frame_batch, frame_batch_tensor, landmarks_batch
            gc.collect()
            torch.cuda.empty_cache()

        return crops

    def extract_lip_segment(self, frame, landmarks):
        if landmarks is None:
            return None, (0, 0, 0, 0)

        lip_landmarks = landmarks[48:]
        x_coords = lip_landmarks[:, 0].astype(int)
        y_coords = lip_landmarks[:, 1].astype(int)

        x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
        y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)

        if x_max <= x_min or y_max <= y_min or (x_max - x_min) < 10 or (y_max - y_min) < 10:
            return None, (x_min, y_min, x_max, y_max)

        lip_crop = frame[y_min:y_max, x_min:x_max]
        return lip_crop, (x_min, y_min, x_max, y_max)

    def parallel_main(self, video_paths: list[str]) -> list[str]:
        available_gpus = torch.cuda.device_count()
        world_size = available_gpus  # Use all available GPUs
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

        print(f"✅ Extracted {len(all_outputs)} lip-only .npy arrays to '{self.output_base_dir}'.")
        return all_outputs


def worker_process(rank, chunks, batch_size, output_dir, return_dict):
    torch.cuda.set_device(rank)
    device_str = f'cuda:{rank}'

    processor = VideoPreprocessor_FANET(
        batch_size=batch_size,
        output_base_dir=output_dir,
        device=device_str,
        rank=rank
    )

    assigned_videos = chunks[rank]
    num_videos = len(assigned_videos)

    print(f"[GPU {rank}] Starting {num_videos} videos.")

    start_time = time.time()
    processed_videos = []

    for idx, video_path in enumerate(assigned_videos):
        try:
            output = processor.process_video(video_path)
            if output:
                processed_videos.append(output)
        except Exception as e:
            print(f"⚠️ [GPU {rank}] Error processing {video_path}: {e}")

        if (idx + 1) % 10 == 0 or (idx + 1) == num_videos:
            elapsed = time.time() - start_time
            avg_time_per_video = elapsed / (idx + 1)
            eta_seconds = avg_time_per_video * (num_videos - idx - 1)
            print(f"[GPU {rank}] {idx + 1}/{num_videos} videos done. ETA: {eta_seconds / 60:.2f} minutes.")

    return_dict[rank] = processed_videos

    del processor
    gc.collect()
    torch.cuda.empty_cache()


# Helper for converting .npy to .mp4 (run separately)
def npy_to_mp4(npy_path, fps=25, output_size=(224, 224)):
    frames = np.load(npy_path)
    out_path = str(Path(npy_path).with_suffix('.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, output_size)

    for frame in frames:
        out.write(frame)
    out.release()
    print(f"[CPU] Saved MP4: {out_path}")

# import os
# import gc
# import cv2
# import torch
# import face_alignment
# import numpy as np
# import torch.multiprocessing as mp
# from pathlib import Path
# import threading
# import queue
# import time
# from tqdm import tqdm
#
#
# class FramePrefetcher:
#     """
#     Prefetch frames from a video capture in a separate thread to hide I/O latency.
#     """
#     def __init__(self, cap, maxsize=512):
#         self.cap = cap
#         self.queue = queue.Queue(maxsize)
#         self.stop_signal = False
#         self.thread = threading.Thread(target=self._reader)
#         self.thread.start()
#
#     def _reader(self):
#         """Continuously read frames and push to queue until video ends."""
#         while not self.stop_signal:
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.queue.put(None)
#                 break
#             self.queue.put(frame)
#
#     def read(self):
#         """Read a frame from the queue."""
#         return self.queue.get()
#
#     def stop(self):
#         """Stop the prefetch thread."""
#         self.stop_signal = True
#         self.thread.join()


# class VideoPreprocessor_FANET:
#     """
#     Distributed video lip-extraction using FaceAlignment and OpenCV for video saving.
#     Includes frame prefetching, optional half-precision, and GPU-optimized memory transfer.
#     """
#     def __init__(self, batch_size: int, output_base_dir: str, device: str = 'cuda', rank: int = 0, use_fp16: bool = False):
#         self.batch_size = batch_size
#         self.output_base_dir = output_base_dir
#         self.device = device
#         self.rank = rank
#         self.use_fp16 = use_fp16
#
#         os.makedirs(self.output_base_dir, exist_ok=True)
#
#         self.fa = face_alignment.FaceAlignment(
#             face_alignment.LandmarksType.TWO_D,
#             device=self.device,
#             face_detector='sfd',
#             flip_input=False
#         )
#
#         torch.backends.cudnn.benchmark = True
#
#     def __call__(self, video_paths: list[str]) -> list[str]:
#         """Process multiple videos sequentially."""
#         processed = []
#         for path in tqdm(video_paths, desc=f"[GPU {self.rank}] Processing videos", position=self.rank):
#             out = self.process_video(path)
#             if out:
#                 processed.append(out)
#         return processed
#
#     def process_video(self, video_path: str) -> str:
#         """Process a single video: extract lips and save new video."""
#         if not os.path.exists(video_path):
#             print(f"❌ Video path does not exist: {video_path}")
#             return None
#
#         video_name = Path(video_path).stem
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print(f"❌ Error opening video: {video_path}")
#             return None
#
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         out_path = os.path.join(self.output_base_dir, f"{video_name}_lips_only.mp4")
#
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(out_path, fourcc, fps, (224, 224))
#
#         print(f"[INFO] [GPU {self.rank}] Saving output to {out_path}")
#
#         prefetcher = FramePrefetcher(cap)
#         buffer = []
#
#         while True:
#             frame = prefetcher.read()
#             if frame is None:
#                 break
#             buffer.append(frame)
#
#             if len(buffer) >= self.batch_size:
#                 success = self._process_batch(buffer, out)
#                 if not success:
#                     break
#                 buffer.clear()
#
#         if buffer:
#             self._process_batch(buffer, out)
#
#         prefetcher.stop()
#         cap.release()
#         out.release()
#
#         if os.path.exists(out_path):
#             size_mb = os.path.getsize(out_path) / (1024 * 1024)
#             print(f"[GPU {self.rank}] Saved {out_path} ({size_mb:.2f} MB)")
#
#         del cap, out, buffer, prefetcher
#         gc.collect()
#         torch.cuda.empty_cache()
#
#         return out_path
#
#     def _process_batch(self, frame_batch, out_writer):
#         """Process a batch of frames and write output crops."""
#         try:
#             frame_batch_tensor = torch.stack(
#                 [torch.from_numpy(frame).permute(2, 0, 1).pin_memory() for frame in frame_batch],
#                 dim=0
#             ).float()
#
#             if self.use_fp16:
#                 frame_batch_tensor = frame_batch_tensor.half()
#
#             frame_batch_tensor = frame_batch_tensor.to(self.device, non_blocking=True)
#
#             landmarks_batch = self.fa.get_landmarks_from_batch(frame_batch_tensor)
#
#             frames_to_write = []
#
#             for frame_tensor, landmarks in zip(frame_batch_tensor, landmarks_batch):
#                 if landmarks is None:
#                     print(f"⚠️ Skipping frame: no face detected.")
#                     continue
#
#                 frame = frame_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
#
#                 lip_crop, _ = self.extract_lip_segment(frame, landmarks)
#
#                 if lip_crop is None:
#                     print(f"⚠️ Skipping frame: invalid lip crop.")
#                     continue
#
#                 resized_crop = cv2.resize(lip_crop, (224, 224), interpolation=cv2.INTER_CUBIC)
#
#                 if resized_crop.ndim != 3 or resized_crop.dtype != np.uint8:
#                     print(f"⚠️ Invalid frame format. Skipping.")
#                     continue
#
#                 frames_to_write.append(resized_crop)
#
#             # Write all collected frames
#             for f in frames_to_write:
#                 out_writer.write(f)
#
#             return True
#
#         except Exception as e:
#             print(f"❌ Batch processing failed: {e}")
#             return False
#
#         finally:
#             del frame_batch, frame_batch_tensor
#             gc.collect()
#             torch.cuda.empty_cache()
#
#     def extract_lip_segment(self, frame, landmarks):
#         """Extract lips region from a frame based on landmarks."""
#         if landmarks is None:
#             return None, (0, 0, 0, 0)
#
#         lip_landmarks = landmarks[48:]
#
#         x_coords = lip_landmarks[:, 0].astype(int)
#         y_coords = lip_landmarks[:, 1].astype(int)
#
#         x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
#         y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)
#
#         if x_max <= x_min or y_max <= y_min:
#             return None, (x_min, y_min, x_max, y_max)
#
#         lip_crop = frame[y_min:y_max, x_min:x_max]
#         return lip_crop, (x_min, y_min, x_max, y_max)
#
#     def parallel_main(self, video_paths: list[str]) -> list[str]:
#         """Parallel entry point: distribute videos across GPUs."""
#         available_gpus = torch.cuda.device_count()
#         world_size = min(available_gpus, 4)  # Only use 4 GPUs max
#         manager = mp.Manager()
#         return_dict = manager.dict()
#         chunks = [video_paths[i::world_size] for i in range(world_size)]
#
#         mp.spawn(
#             worker_process,
#             args=(chunks, self.batch_size, self.output_base_dir, return_dict, self.use_fp16),
#             nprocs=world_size,
#             join=True
#         )
#
#         all_outputs = []
#         for rank in range(world_size):
#             all_outputs.extend(return_dict.get(rank, []))
#
#         print(f"✅ Extracted {len(all_outputs)} lip-only videos to '{self.output_base_dir}'.")
#         return all_outputs
#
#
# def worker_process(rank, chunks, batch_size, output_dir, return_dict, use_fp16):
#     """Each GPU worker."""
#     torch.cuda.set_device(rank)
#     device_str = f'cuda:{rank}'
#
#     processor = VideoPreprocessor_FANET(
#         batch_size=batch_size,
#         output_base_dir=output_dir,
#         device=device_str,
#         rank=rank,
#         use_fp16=use_fp16
#     )
#
#     assigned_videos = chunks[rank]
#     processed_videos = []
#
#     print(f"[GPU {rank}] Starting {len(assigned_videos)} videos.")
#
#     for idx, video_path in enumerate(assigned_videos):
#         try:
#             output = processor.process_video(video_path)
#             if output:
#                 processed_videos.append(output)
#         except Exception as e:
#             print(f"⚠️ [GPU {rank}] Error processing {video_path}: {e}")
#
#     return_dict[rank] = processed_videos
#
#     del processor
#     gc.collect()
#     torch.cuda.empty_cache()
