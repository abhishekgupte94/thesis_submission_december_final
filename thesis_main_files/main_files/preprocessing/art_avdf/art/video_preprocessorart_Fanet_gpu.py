# import os
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# import cv2
# import numpy as np
# import face_alignment
# import torch
# import gc
# import psutil
# from pathlib import Path
#
# # from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# torch.backends.cudnn.benchmark = True
# from tqdm import tqdm
#
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
#         # 🔧 MODIFIED: Explicit device setup for single GPU
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#         # 🔧 MODIFIED: GPU-based face alignment initialized once
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
#         video_name = os.path.basename(video_path).split('.')[0]
#         output_video_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.mp4")
#
#         try:
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 print(f"Error opening video: {video_path}")
#                 return None
#
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             lip_video_size = (224, 224)
#             # Save intermediate as .avi to avoid codec issues in Colab
#             avi_output_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.avi")
#             fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#             out = cv2.VideoWriter(avi_output_path, fourcc, fps, lip_video_size)
#
#             if not out.isOpened():
#                 print(f"❌ Error creating AVI output: {avi_output_path}")
#                 return None
#
#             processed_frames = 0
#             # print(f"Processing {video_path} | Initial memory: {get_memory_usage():.2f} MB")
#
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#
#                 # frame_mem = get_memory_usage()
#                 result = self.process_frame(frame, out)
#                 # if result:
#                     # processed_frames += 1
#
#                 del frame
#                 # if processed_frames % 10 == 0:
#                 #     gc.collect()
#                 #     if self.device == 'cuda':
#                 #         torch.cuda.empty_cache()
#
#                 # print(f"Frame {processed_frames} | Δ Memory: {get_memory_usage() - frame_mem:.2f} MB")
#
#             cap.release()
#             out.release()
#             # Convert to .mp4 using ffmpeg if needed
#
#
#             del cap, out
#             output_video_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.mp4")
#             conversion_cmd = f"ffmpeg -y -i \"{avi_output_path}\" -vcodec libx264 \"{output_video_path}\""
#
#             print(f"📦 Converting AVI to MP4...")
#             os.system(conversion_cmd)
#
#             # Optional: delete the .avi to save space
#             os.remove(avi_output_path)
#
#             return output_video_path if processed_frames > 0 else None
#
#         except Exception as e:
#             print(f"Error processing {video_path}: {str(e)}")
#             return None
#
#     def process_frame(self, frame, out_writer):
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         try:
#             landmarks = self.fa.get_landmarks(rgb_frame)
#             if not landmarks:
#                 return False
#
#             lip_segment, _ = self.extract_lip_segment(frame, landmarks[0])
#             if lip_segment.size == 0:
#                 return False
#
#             lip_resized = cv2.resize(lip_segment, (224, 224), interpolation=cv2.INTER_CUBIC)
#             out_writer.write(lip_resized)
#
#             del rgb_frame, landmarks, lip_segment, lip_resized
#             return True
#
#         except Exception as e:
#             print(f"Frame processing error: {str(e)}")
#             return False
#
#     def extract_lip_segment(self, frame, landmarks):
#         lip_landmarks = landmarks[48:]
#         x_coords = lip_landmarks[:, 0].astype(int)
#         y_coords = lip_landmarks[:, 1].astype(int)
#
#         x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
#         y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)
#
#         return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
#
#     def main(self, video_paths):
#         processed_paths = []
#
#         # ❌ REMOVED: joblib.Parallel for multi-video processing
#         # 🔧 MODIFIED: Sequential single-GPU video processing
#         for video_path in video_paths:
#             print("video done!")
#             result = self.process_video(video_path)
#             if result:
#                 with open(self.real_output_txt_path, 'a') as f:
#                     f.write(f"{os.path.basename(result)} 0\n")
#                 processed_paths.append(result)
#
#         print(f"Processed {len(processed_paths)} videos.")
#         return
#
#
#     def main_parallel(self, video_paths, max_workers=2):
#         processed_paths = []
#
#         print(f"🧵 Starting parallel video processing with {max_workers} workers...")
#
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = {executor.submit(self.process_video, vp): vp for vp in video_paths}
#
#             for future in tqdm(as_completed(futures), total=len(video_paths), desc="📦 Processing videos"):
#                 result = future.result()
#                 if result:
#                     processed_paths.append(result)
#         print(len(processed_paths))
#         # ✅ Now write to the label file once all processing is done
#         if self.real_output_txt_path and processed_paths:
#             with open(self.real_output_txt_path, 'w') as f:
#                 for path in processed_paths:
#                     f.write(f"{os.path.basename(path)} 0\n")
#
#         print(f"✅ Processed {len(processed_paths)} videos.")
#         return processed_paths
#
#     def main_single(self, real_video_single):
#         processed_paths = []
#
#         print(f"\nProcessing single video")
#         print(f"Start memory: {get_memory_usage():.2f} MB")
#
#         result = self.process_video(real_video_single)
#         if result:
#             with open(self.real_output_txt_path, 'w') as f:
#                 f.write(f"{os.path.basename(result)} 0\n")
#             processed_paths.append(result)
#
#         # 🔧 MODIFIED: Force GC and GPU memory cleanup per video
#         gc.collect()
#         if self.device == 'cuda':
#             torch.cuda.empty_cache()
#
#         print(f"Post-cleanup memory: {get_memory_usage():.2f} MB")
#         return processed_paths


import os
import gc
import cv2
import torch
import psutil
import face_alignment
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        video_name = os.path.basename(video_path).split('.')[0]
        output_video_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.mp4")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video: {video_path}")
                return None

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            lip_video_size = (224, 224)
            avi_output_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(avi_output_path, fourcc, fps, lip_video_size)

            if not out.isOpened():
                print(f"❌ Error creating AVI output: {avi_output_path}")
                return None

            frame_buffer = []
            original_frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(rgb_frame)
                original_frames.append(frame)

                if len(frame_buffer) >= self.batch_size:
                    self._process_batch(frame_buffer, original_frames, out)
                    frame_buffer = []
                    original_frames = []

            if frame_buffer:
                self._process_batch(frame_buffer, original_frames, out)

            cap.release()
            out.release()
            del cap, out

            conversion_cmd = f"ffmpeg -y -i \"{avi_output_path}\" -vcodec libx264 \"{output_video_path}\""
            print(f"📦 Converting AVI to MP4...")
            os.system(conversion_cmd)
            os.remove(avi_output_path)

            return output_video_path

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None

    def _process_batch(self, rgb_batch, original_batch, out_writer):
        try:
            landmarks_batch = self.fa.get_landmarks_from_batch(rgb_batch)
        except Exception as e:
            print(f"⚠️ Batch landmark error: {str(e)}")
            return

        for orig_frame, landmarks in zip(original_batch, landmarks_batch or []):
            if landmarks is None:
                continue
            try:
                lip_segment, _ = self.extract_lip_segment(orig_frame, landmarks)
                if not isinstance(lip_segment, np.ndarray) or lip_segment.size == 0:
                    continue

                lip_resized = cv2.resize(lip_segment, (224, 224), interpolation=cv2.INTER_CUBIC)
                out_writer.write(lip_resized)
            except Exception as e:
                print(f"⚠️ Lip extraction error: {str(e)}")
                continue

    def extract_lip_segment(self, frame, landmarks):
        lip_landmarks = landmarks[48:]
        x_coords = lip_landmarks[:, 0].astype(int)
        y_coords = lip_landmarks[:, 1].astype(int)

        x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
        y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)

        return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

    def main(self, video_paths):
        processed_paths = []

        for video_path in video_paths:
            result = self.process_video(video_path)
            if result:
                with open(self.real_output_txt_path, 'w') as f:
                    f.write(f"{os.path.basename(result)} 0\n")
                processed_paths.append(result)

        print(f"✅ Processed {len(processed_paths)} videos.")
        return

    def main_parallel(self, video_paths, max_workers=2):
        processed_paths = []
        print(f"🧵 Starting parallel video processing with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_video, vp): vp for vp in video_paths}

            for future in tqdm(as_completed(futures), total=len(video_paths), desc="📦 Processing videos"):
                result = future.result()
                if result:
                    processed_paths.append(result)

        if self.real_output_txt_path and processed_paths:
            with open(self.real_output_txt_path, 'w') as f:
                for path in processed_paths:
                    f.write(f"{os.path.basename(path)} 0\n")

        print(f"✅ Processed {len(processed_paths)} videos.")
        return processed_paths

    def main_single(self, real_video_single):
        processed_paths = []

        print(f"\nProcessing single video")
        print(f"Start memory: {get_memory_usage():.2f} MB")

        result = self.process_video(real_video_single)
        if result:
            with open(self.real_output_txt_path, 'w') as f:
                f.write(f"{os.path.basename(result)} 0\n")
            processed_paths.append(result)

        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        print(f"Post-cleanup memory: {get_memory_usage():.2f} MB")
        return processed_paths
