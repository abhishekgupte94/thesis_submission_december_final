#
# import os
# import gc
# import cv2
# import torch
# import psutil
# import face_alignment
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# torch.backends.cudnn.benchmark = True
#
# import multiprocessing as mp
# from torch.multiprocessing import spawn
#
# def get_memory_usage():
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024 / 1024  # MB
#
#
# class VideoPreprocessor_FANET:
#     def __init__(self, batch_size=32, output_base_dir_real=None, real_output_txt_path=None):
#         self.batch_size = batch_size
#         self.output_base_dir_real = output_base_dir_real
#         self.real_output_txt_path = real_output_txt_path
#
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#         self.fa = face_alignment.FaceAlignment(
#             face_alignment.LandmarksType.TWO_D,
#             device=self.device,
#             face_detector='sfd',
#             flip_input=False
#         )
#
#         os.makedirs(self.output_base_dir_real, exist_ok=True)
#
#     def process_video(self, video_path):
#         self.frames_written = 0
#
#         video_name = os.path.basename(video_path).split('.')[0]
#         output_video_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.mp4")
#
#         try:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"❌ Error opening video: {video_path}")
#                 return None
#
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             frame_size = (width, height)
#
#             # Use .mp4v or fallback to .XVID
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
#
#             if not out.isOpened():
#                 print(f"❌ Error creating MP4 output: {output_video_path}")
#                 return None
#
#             frame_buffer, original_frames = [], []
#
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#
#                 # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 # frame_buffer.append(rgb_frame)
#                 original_frames.append(frame)
#
#                 if len(frame_buffer) >= self.batch_size:
#                     self._process_batch(frame_buffer, original_frames, out)
#                     frame_buffer.clear()
#                     original_frames.clear()
#
#             if frame_buffer:
#                 self._process_batch(frame_buffer, original_frames, out)
#
#             cap.release()
#             out.release()
#             del cap, out
#
#             print(f"📸 Total frames written: {self.frames_written}")
#             return output_video_path if self.frames_written > 0 else None
#
#         except Exception as e:
#             print(f"❌ Error processing {video_path}: {str(e)}")
#             return None
#
#     def _process_batch(self, rgb_batch, original_batch, out_writer):
#         try:
#             batch_tensor = torch.stack([
#                 torch.from_numpy(img).permute(2, 0, 1).float()# / 255.0
#                 for img in original_batch
#             ]).to(self.device)
#             print(f"🧪 Batch tensor shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}")
#
#             # landmarks_batch = self.fa.get_landmarks_from_batch(batch_tensor)
#             landmarks_batch = self.fa.get_landmarks_from_batch(batch_tensor)
#
#             print(f"➡️ Landmarks detected: {len(landmarks_batch or [])}")
#
#             print(f"🧠 Type of landmarks_batch: {type(landmarks_batch)}")
#             # for i, landmarks in enumerate(landmarks_batch or []):
#             #     print(f"   Frame {i} → Type: {type(landmarks)}, Value: {landmarks}")
#         except Exception as e:
#             print(f"⚠️ Batch landmark error: {str(e)}")
#             return
#
#
#         for orig_frame, landmarks_per_frame in zip(original_batch, landmarks_batch or []):
#             if landmarks_per_frame is None or len(
#                     landmarks_per_frame) == 0:
#                 print("⚠️ No face detected in frame.")
#                 continue
#
#             try:
#                 # Always take the first detected face landmarks
#                 # ✅ Correct:
#                 single_face_landmarks = landmarks_per_frame
#
#                 # Now try lip extraction
#                 lip_segment, _ = self.extract_lip_segment(orig_frame, single_face_landmarks)
#
#                 print(f"Lip segment size {lip_segment.size}")
#                 if  lip_segment.size == 0:
#                     continue
#
#                 lip_resized = cv2.resize(lip_segment, (224, 224), interpolation=cv2.INTER_CUBIC)
#                 # lip_resized_bgr = cv2.cvtColor(lip_resized, cv2.COLOR_RGB2BGR)
#                 out_writer.write(lip_resized)
#
#                 self.frames_written += 1  # ✅ Count successful frame writes
#                 print(f"🖼️ Frame written. Total so far: {self.frames_written}")
#
#             except Exception as e:
#                 print(f"⚠️ Lip extraction error: {str(e)}")
#                 continue
#
#     def extract_lip_segment(self, frame, landmarks):
#         # Use 68-point landmarks → mouth is from index 48 to 67
#         lip_landmarks = landmarks[48:]
#
#         # Get x and y coordinates
#         x_coords = lip_landmarks[:, 0].astype(int)
#         y_coords = lip_landmarks[:, 1].astype(int)
#
#         # Get bounding box of lip landmarks
#         x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
#         y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)
#
#         # Simple check to avoid bad crops
#         if x_max <= x_min or y_max <= y_min:
#             print(f"⚠️ Invalid lip crop: x({x_min}:{x_max}), y({y_min}:{y_max})")
#             return np.array([]), (x_min, y_min, x_max, y_max)
#
#         # Extract the region from the original frame
#         lip_crop = frame[y_min:y_max, x_min:x_max]
#         print(f"✅ Lip segment shape: {lip_crop.shape}")
#
#         return lip_crop, (x_min, y_min, x_max, y_max)
#     #
#     # def main_parallel(self, video_paths, max_workers=2):
#     #     processed_paths = []
#     #     print(f"🧵 Starting parallel video processing with {max_workers} workers...")
#     #
#     #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#     #         futures = {executor.submit(self.process_video, vp): vp for vp in video_paths}
#     #
#     #         for future in tqdm(as_completed(futures), total=len(video_paths), desc="📦 Processing videos"):
#     #             result = future.result()
#     #             if result:
#     #                 processed_paths.append(result)
#     #
#     #     if self.real_output_txt_path and processed_paths:
#     #         with open(self.real_output_txt_path, 'w') as f:
#     #             for path in processed_paths:
#     #                 f.write(f"{os.path.basename(path)} 0\n")
#     #
#     #     print(f"✅ Processed {len(processed_paths)} videos.")
#     #     return processed_paths
#     #
#     # def main_single(self, real_video_single):
#     #     processed_paths = []
#     #
#     #     print(f"\nProcessing single video")
#     #     print(f"Start memory: {get_memory_usage():.2f} MB")
#     #
#     #     result = self.process_video(real_video_single)
#     #     if result:
#     #         with open(self.real_output_txt_path, 'w') as f:
#     #             f.write(f"{os.path.basename(result)} 0\n")
#     #         processed_paths.append(result)
#     #
#     #     gc.collect()
#     #     if self.device == 'cuda':
#     #         torch.cuda.empty_cache()
#     #
#     #     print(f"Post-cleanup memory: {get_memory_usage():.2f} MB")
#     #     return processed_paths
#
#     def main(self, video_paths):
#         processed_paths = []
#
#         for video_path in video_paths:
#             print(f"🎞️ Processing: {video_path}")
#             result = self.process_video(video_path)
#
#             if result:
#                 with open(self.real_output_txt_path, 'w') as f:
#                     f.write(f"{os.path.basename(result)} 0\n")  # Label '0' for real
#                 processed_paths.append(result)
#             else:
#                 print(f"❌ Skipped or failed: {video_path}")
#
#         print(f"✅ Finished processing {len(processed_paths)} out of {len(video_paths)} videos.")
#         return processed_paths
#
#
#     def worker_process(rank, video_paths, output_dir, output_txt, batch_size):
#         print(f"🚀 Worker {rank} started with {len(video_paths)} videos.")
#
#         processor = VideoPreprocessor_FANET(
#             batch_size=batch_size,
#             output_base_dir_real=output_dir,
#             real_output_txt_path=None  # Avoid write conflict
#         )
#
#         results = []
#         for video_path in video_paths:
#             result = processor.process_video(video_path)
#             if result:
#                 results.append(result)
#
#         return results
#
#     def parallel_main(self, video_paths, num_workers=8):
#         print(f"⚙️ Launching GPU-aware parallel processing with {num_workers} workers...")
#
#         # Split work evenly
#         chunks = [video_paths[i::num_workers] for i in range(num_workers)]
#         ctx = mp.get_context("spawn")
#         with ctx.Pool(num_workers) as pool:
#             results = pool.starmap(
#                 self.worker_process,
#                 [(i, chunk, self.output_base_dir_real, self.real_output_txt_path, self.batch_size)
#                  for i, chunk in enumerate(chunks)]
#             )
#
#         # Flatten and optionally save
#         flattened = [item for sublist in results for item in sublist if item]
#         if self.real_output_txt_path:
#             with open(self.real_output_txt_path, 'w') as f:
#                 for path in flattened:
#                     f.write(f"{os.path.basename(path)} 0\n")
#
#         print(f"✅ Parallel processing done. Processed: {len(flattened)} videos.")
#         return flattened


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

    def parallel_main(self, video_paths, num_workers=12):
        print(f"⚙️ Launching GPU-parallel processing with {num_workers} workers")

        ctx = tmp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()

        chunks = [video_paths[i::num_workers] for i in range(num_workers)]
        args = (self.batch_size, self.output_base_dir_real, None)

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
# import os
# import gc
# import cv2
# import torch
# import psutil
# import face_alignment
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# import torch.multiprocessing as tmp
#
# torch.backends.cudnn.benchmark = True
#
# # Shared face_alignment model (lazy init)
# _fa_instance = None
# def get_face_alignment(device='cuda'):
#     global _fa_instance
#     if _fa_instance is None:
#         _fa_instance = face_alignment.FaceAlignment(
#             face_alignment.LandmarksType.TWO_D,
#             device=device,
#             face_detector='sfd',
#             flip_input=False
#         )
#     return _fa_instance
#
# class VideoPreprocessor_FANET:
#     def __init__(self, batch_size=24, output_base_dir_real=None, real_output_txt_path=None):
#         self.batch_size = batch_size
#         self.output_base_dir_real = output_base_dir_real
#         self.real_output_txt_path = real_output_txt_path
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         os.makedirs(self.output_base_dir_real, exist_ok=True)
#
#     def process_video(self, video_path):
#         self.frames_written = 0
#         video_name = os.path.basename(video_path).split('.')[0]
#         output_video_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.mp4")
#
#         try:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"❌ Error opening video: {video_path}")
#                 return None
#
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             frame_size = (self.width, self.height)
#
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
#
#             if not out.isOpened():
#                 raise RuntimeError("❌ VideoWriter failed to open. Check codec or path.")
#
#             original_frames = []
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 original_frames.append(frame)
#                 if len(original_frames) >= self.batch_size:
#                     self._process_batch(original_frames, out)
#                     original_frames.clear()
#
#             if original_frames:
#                 self._process_batch(original_frames, out)
#
#             cap.release()
#             out.release()
#             del cap, out, original_frames
#             gc.collect()
#             if self.device == 'cuda':
#                 torch.cuda.empty_cache()
#
#             print(f"📸 Total frames written: {self.frames_written}")
#             return output_video_path if self.frames_written > 0 else None
#
#         except Exception as e:
#             print(f"❌ Error processing {video_path}: {str(e)}")
#             return None
#
#     def _process_batch(self, original_batch, out_writer):
#         try:
#             batch_tensor = torch.stack([
#                 torch.from_numpy(img).permute(2, 0, 1).float()
#                 for img in original_batch
#             ])
#             batch_tensor = batch_tensor.to(self.device, non_blocking=True)
#
#             fa = get_face_alignment(self.device)
#             landmarks_batch = fa.get_landmarks_from_batch(batch_tensor)
#
#             allocated = torch.cuda.memory_allocated() / 1024 / 1024
#             print(f"🚀 GPU mem used this batch: {allocated:.2f} MB")
#
#             del batch_tensor
#             torch.cuda.empty_cache()
#
#         except Exception as e:
#             print(f"⚠️ Landmark batch error: {str(e)}")
#             return
#
#         for orig_frame, landmarks_per_frame in zip(original_batch, landmarks_batch or []):
#             if landmarks_per_frame is None or len(landmarks_per_frame) == 0:
#                 print("⚠️ No face detected in frame.")
#                 continue
#
#             try:
#                 lip_segment, _ = self.extract_lip_segment(orig_frame, landmarks_per_frame)
#                 if lip_segment.size == 0:
#                     continue
#                 lip_resized = cv2.resize(lip_segment, (self.width, self.height))
#                 out_writer.write(lip_resized)
#                 self.frames_written += 1
#             except Exception as e:
#                 print(f"⚠️ Lip extraction error: {str(e)}")
#                 continue
#
#         del original_batch, landmarks_batch
#         gc.collect()
#         if self.device == 'cuda':
#             torch.cuda.empty_cache()
#
#     def extract_lip_segment(self, frame, landmarks):
#         lip_landmarks = landmarks[48:]
#         x_coords = lip_landmarks[:, 0].astype(int)
#         y_coords = lip_landmarks[:, 1].astype(int)
#         x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
#         y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)
#         if x_max <= x_min or y_max <= y_min:
#             return np.array([]), (x_min, y_min, x_max, y_max)
#         return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
#
#     @staticmethod
#     def _worker_process_static(rank, video_paths, args, return_dict):
#         print(f"🚀 Worker {rank} processing {len(video_paths)} videos")
#         try:
#             instance = VideoPreprocessor_FANET(*args)
#             results = [instance.process_video(vp) for vp in video_paths if vp]
#             return_dict[rank] = [r for r in results if r]
#         except Exception as e:
#             print(f"❌ Worker {rank} error: {e}")
#             return_dict[rank] = []
#         finally:
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#     def parallel_main(self, video_paths, num_workers=8):
#         print(f"⚙️ Launching GPU-parallel processing with {num_workers} workers")
#         ctx = tmp.get_context("spawn")  # Or use 'forkserver' if local Linux
#         manager = ctx.Manager()
#         return_dict = manager.dict()
#         chunks = [video_paths[i::num_workers] for i in range(num_workers)]
#         args = (self.batch_size, self.output_base_dir_real, None)
#         processes = []
#         for i, chunk in enumerate(chunks):
#             p = ctx.Process(target=self._worker_process_static, args=(i, chunk, args, return_dict))
#             p.start()
#             processes.append(p)
#         for p in processes:
#             p.join()
#
#         all_processed = []
#         for paths in return_dict.values():
#             all_processed.extend(paths)
#         if self.real_output_txt_path:
#             with open(self.real_output_txt_path, 'w') as f:
#                 for path in all_processed:
#                     f.write(f"{os.path.basename(path)} 0\n")
#         print(f"✅ Parallel processing done. Total: {len(all_processed)} videos.")
#         return all_processed
