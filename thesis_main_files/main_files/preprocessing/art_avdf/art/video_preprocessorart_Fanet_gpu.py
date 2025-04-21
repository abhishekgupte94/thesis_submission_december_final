# ============================
# Parallel Video Preprocessing (GPU-accelerated)
# ============================

import os
import cv2
import numpy as np
import face_alignment
import torch
import gc
import psutil
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed  # NEW: added for parallel processing
from multiprocessing import cpu_count  # NEW: used to determine number of parallel jobs


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
            face_detector='sfd'
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
            fourcc = cv2.VideoWriter_fourcc(*'H264')

            try:
                out = cv2.VideoWriter(output_video_path, fourcc, fps, lip_video_size)
                if not out.isOpened():
                    print(f"Error creating output video: {output_video_path}")
                    return None

                processed_frames = 0
                print(f"Processing {video_path} | Initial memory: {get_memory_usage():.2f} MB")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_mem = get_memory_usage()
                    result = self.process_frame(frame, out)
                    if result:
                        processed_frames += 1

                    del frame
                    if processed_frames % 10 == 0:
                        gc.collect()
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()

                    print(f"Frame {processed_frames} | Δ Memory: {get_memory_usage() - frame_mem:.2f} MB")

                return output_video_path if processed_frames > 0 else None

            finally:
                cap.release()
                out.release()
                del cap, out

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None

    def process_frame(self, frame, out_writer):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            landmarks = self.fa.get_landmarks(rgb_frame)
            if not landmarks:
                return False

            lip_segment, _ = self.extract_lip_segment(frame, landmarks[0])
            if lip_segment.size == 0:
                return False

            lip_resized = cv2.resize(lip_segment, (224, 224), interpolation=cv2.INTER_CUBIC)
            out_writer.write(lip_resized)

            del rgb_frame, landmarks, lip_segment, lip_resized
            return True

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return False

    def extract_lip_segment(self, frame, landmarks):
        lip_landmarks = landmarks[48:]
        x_coords = lip_landmarks[:, 0].astype(int)
        y_coords = lip_landmarks[:, 1].astype(int)

        x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
        y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)

        return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

    def main(self, video_paths):
        processed_paths = []

        def process_and_log(video_path):
            result = self.process_video(video_path)
            return result

        # Parallel processing of videos using joblib
        num_jobs = min(cpu_count(), len(video_paths))  # NEW: determine optimal number of parallel jobs
        results = Parallel(n_jobs=num_jobs)(delayed(process_and_log)(vp) for vp in video_paths)  # NEW: parallel processing

        with open(self.real_output_txt_path, 'w') as f:
            for result in results:
                if result:
                    f.write(f"{os.path.basename(result)} 0\n")
                    processed_paths.append(result)

        print(f"File {self.real_output_txt_path} has been overwritten with new values.")
        return processed_paths

    def main_single(self, real_video_single):
        processed_paths = []

        print(f"\nProcessing single video")
        print(f"Start memory: {get_memory_usage():.2f} MB")

        result = self.process_video(real_video_single)
        if result:
            with open(self.real_output_txt_path, 'a') as f:
                f.write(f"{os.path.basename(result)} 0\n")
            processed_paths.append(result)

        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        print(f"Post-cleanup memory: {get_memory_usage():.2f} MB")

        return processed_paths
