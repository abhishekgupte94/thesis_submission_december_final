

import os
import gc
import cv2
import torch
import psutil
import face_alignment
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.multiprocessing as tmp
import subprocess

torch.backends.cudnn.benchmark = True

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

class VideoPreprocessor_FANET:
    def __init__(self, batch_size=32, output_base_dir_real=None, real_output_txt_path=None):
        self.batch_size = batch_size
        self.output_base_dir_real = output_base_dir_real
        self.real_output_txt_path = real_output_txt_path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self.device,
            face_detector='sfd',
            flip_input=False
        )

        os.makedirs(self.output_base_dir_real, exist_ok=True)

    def process_video(self, video_path):
        self.frames_written = 0
        video_name = os.path.basename(video_path).split('.')[0]
        output_video_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.mp4")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Error opening video: {video_path}")
                return None


            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (self.width, self.height)

            # Define output path and codec (MP4 + H.264)
            mp4_output_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' for H.264; fallback 'mp4v' if needed
            out = cv2.VideoWriter(mp4_output_path, fourcc, fps, frame_size)

            # Optional: Check if VideoWriter opened successfully
            if not out.isOpened():
                raise Exception("VideoWriter failed to open. Check codec and file path.")

            if not out.isOpened():
                raise RuntimeError("❌ VideoWriter failed to open. Use XVID and .avi")

            frame_buffer, original_frames = [], []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                original_frames.append(frame)

                if len(original_frames) >= self.batch_size:
                    self._process_batch(original_frames, out)
                    original_frames.clear()

            if original_frames:
                self._process_batch(original_frames, out)

            cap.release()
            out.release()

            # 💥 Aggressively clear memory
            del cap, out, frame_buffer, original_frames
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            # mp4_output_path = avi_output_path.replace(".avi", ".mp4")
            # subprocess.run([
            #     "ffmpeg", "-y",
            #     "-i", avi_output_path,
            #     "-vcodec", "libx264",
            #     "-pix_fmt", "yuv420p",
            #     mp4_output_path
            # ])

            # os.remove(avi_output_path)
            print(f"📸 Total frames written: {self.frames_written}")

            return output_video_path if self.frames_written > 0 else None

        except Exception as e:
            print(f"❌ Error processing {video_path}: {str(e)}")
            return None

    def _process_batch(self, original_batch, out_writer):
        try:
            batch_tensor = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float()
                for img in original_batch
            ]).to(self.device)

            landmarks_batch = self.fa.get_landmarks_from_batch(batch_tensor)

            # 💥 Free tensor from memory
            del batch_tensor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ Batch landmark error: {str(e)}")
            return

        for orig_frame, landmarks_per_frame in zip(original_batch, landmarks_batch or []):
            if landmarks_per_frame is None or len(landmarks_per_frame) == 0:
                print("⚠️ No face detected in frame.")
                continue

            try:
                single_face_landmarks = landmarks_per_frame
                lip_segment, _ = self.extract_lip_segment(orig_frame, single_face_landmarks)

                if lip_segment.size == 0:
                    continue

                lip_resized = cv2.resize(lip_segment, (self.width, self.height))
                out_writer.write(lip_resized)
                self.frames_written += 1

            except Exception as e:
                print(f"⚠️ Lip extraction error: {str(e)}")
                continue

        # 💥 Free batch memory
        del original_batch, landmarks_batch
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def extract_lip_segment(self, frame, landmarks):
        lip_landmarks = landmarks[48:]
        x_coords = lip_landmarks[:, 0].astype(int)
        y_coords = lip_landmarks[:, 1].astype(int)

        x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
        y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)

        if x_max <= x_min or y_max <= y_min:
            print(f"⚠️ Invalid lip crop: x({x_min}:{x_max}), y({y_min}:{y_max})")
            return np.array([]), (x_min, y_min, x_max, y_max)

        lip_crop = frame[y_min:y_max, x_min:x_max]
        return lip_crop, (x_min, y_min, x_max, y_max)

    @staticmethod
    def _worker_process_static(rank, video_paths, args, return_dict):
        print(f"🚀 Worker {rank} processing {len(video_paths)} videos")

        try:
            instance = VideoPreprocessor_FANET(*args)
            results = []

            for video_path in video_paths:
                result = instance.process_video(video_path)
                if result:
                    results.append(result)

            return_dict[rank] = results

        except Exception as e:
            print(f"❌ Worker {rank} error: {e}")
            return_dict[rank] = []

        finally:
            del instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"🧹 Worker {rank} memory cleaned up.")

    def parallel_main(self, video_paths, num_workers=6):
        print(f"⚙️ Launching GPU-parallel processing with {num_workers} workers")

        ctx = tmp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()

        chunks = [video_paths[i::num_workers] for i in range(num_workers)]
        f = (self.batch_size, self.output_base_dir_real, None)

        processes = []
        for i, chunk in enumerate(chunks):
            p = ctx.Process(target=self._worker_process_static, args=(i, chunk, args, return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        all_processed = []
        for paths in return_dict.values():
            all_processed.extend(paths)

        if self.real_output_txt_path:
            with open(self.real_output_txt_path, 'w') as f:
                for path in all_processed:
                    f.write(f"{os.path.basename(path)} 0\n")

        print(f"✅ Parallel processing done. Total: {len(all_processed)} videos.")
        return all_processed
