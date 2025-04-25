import os
import gc
import cv2
import torch
import numpy as np
import multiprocessing as tmp
import torchlm
from torchlm.models import pipnet
from torchlm.tools import faceboxesv2

class VideoPreprocessor_PIPNet:
    def __init__(self, output_base_dir_real=None, real_output_txt_path=None):
        self.output_base_dir_real = output_base_dir_real
        self.real_output_txt_path = real_output_txt_path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.face_detector = faceboxesv2(device=self.device)
        self.landmark_detector = pipnet(
            backbone="resnet18",
            pretrained=True,
            num_nb=10,
            num_lms=98,
            net_stride=32,
            input_size=256,
            meanface_type="wflw",
            map_location=self.device,
            checkpoint=None
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

            fps = cap.get(cv2.CAP_PROP_FPS)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (self.width, self.height)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

            if not out.isOpened():
                raise Exception("VideoWriter failed to open. Check codec and file path.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self._process_single_frame(frame, out)

            cap.release()
            out.release()

            del cap, out
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            print(f"📸 Total frames written: {self.frames_written}")
            return output_video_path if self.frames_written > 0 else None

        except Exception as e:
            print(f"❌ Error processing {video_path}: {str(e)}")
            return None

    def _process_single_frame(self, frame, out_writer):
        try:
            bboxes = self.face_detector(frame)

            if len(bboxes) == 0:
                print("⚠️ No face detected in frame.")
                return

            landmarks = self.landmark_detector(frame, bboxes=bboxes)

            if landmarks is None or len(landmarks) == 0:
                print("⚠️ No landmarks found.")
                return

            single_face_landmarks = landmarks[0]
            lip_segment, _ = self.extract_lip_segment(frame, single_face_landmarks)

            if lip_segment.size == 0:
                return

            lip_resized = cv2.resize(lip_segment, (self.width, self.height))
            out_writer.write(lip_resized)
            self.frames_written += 1

        except Exception as e:
            print(f"⚠️ Frame processing error: {str(e)}")

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
            instance = VideoPreprocessor_PIPNet(*args)
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
        args = (self.output_base_dir_real, None)

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
