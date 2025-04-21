import os
import cv2
import numpy as np
import face_alignment
import torch
import gc

class VideoPreprocessor_FANET:
    def __init__(self, batch_size=32, output_base_dir_real=None, real_output_txt_path=None):
        """
        Initialize the video preprocessor.
        Uses face_alignment for landmark detection, supports GPU if available.
        """
        self.batch_size = batch_size
        self.output_base_dir_real = output_base_dir_real
        self.real_output_txt_path = real_output_txt_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_gpu = self.device == 'cuda'

        # Initialize face alignment model using GPU (or CPU fallback)
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self.device,
            face_detector='sfd'  # Memory-efficient and accurate
        )

        # Ensure the output directory exists
        os.makedirs(self.output_base_dir_real, exist_ok=True)

    def process_video(self, video_path):
        """
        Extracts lip region from video frame-by-frame and writes a new video with lip crops.
        """
        video_name = os.path.basename(video_path).split('.')[0]
        output_video_path = os.path.join(self.output_base_dir_real, f"{video_name}_lips_only.mp4")

        try:
            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Error opening video: {video_path}")
                return None

            # Setup video writer
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            lip_video_size = (224, 224)
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, lip_video_size)

            if not out.isOpened():
                print(f"❌ Error creating output video: {output_video_path}")
                return None

            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the current frame and write to output
                if self.process_frame(frame, out):
                    processed_frames += 1

                # Clean up memory to avoid GPU overflow
                del frame
                if processed_frames % 10 == 0:
                    gc.collect()
                    if self.use_gpu:
                        torch.cuda.empty_cache()

            # Release resources after processing
            cap.release()
            out.release()

            # Return output path only if successful
            return output_video_path if processed_frames > 0 else None

        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            return None

    def process_frame(self, frame, out_writer):
        """
        Detects landmarks and writes lip crop to output.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            landmarks = self.fa.get_landmarks(rgb_frame)
            if not landmarks:
                return False

            # Extract lips and resize
            lip_segment, _ = self.extract_lip_segment(frame, landmarks[0])
            if lip_segment.size == 0:
                return False

            lip_resized = cv2.resize(lip_segment, (224, 224), interpolation=cv2.INTER_CUBIC)
            out_writer.write(lip_resized)

            # Explicitly free memory used in the frame
            del rgb_frame, landmarks, lip_segment, lip_resized
            return True

        except Exception as e:
            print(f"⚠️ Frame processing error: {e}")
            return False

    def extract_lip_segment(self, frame, landmarks):
        """
        Crops the lip region based on 68-point facial landmarks (index 48–67).
        """
        lip_landmarks = landmarks[48:]
        x_coords = lip_landmarks[:, 0].astype(int)
        y_coords = lip_landmarks[:, 1].astype(int)

        # Clamp to avoid going out of image bounds
        x_min, x_max = np.clip([x_coords.min(), x_coords.max()], 0, frame.shape[1] - 1)
        y_min, y_max = np.clip([y_coords.min(), y_coords.max()], 0, frame.shape[0] - 1)

        return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

    def main(self, video_paths):
        """
        Processes a batch of videos and writes their cropped outputs.
        Overwrites the output TXT file with new paths.
        """
        processed_paths = []

        # Open output TXT file for writing video paths
        with open(self.real_output_txt_path, 'w') as f:
            for video_path in video_paths:
                print(f"📂 Processing: {os.path.basename(video_path)}")

                result = self.process_video(video_path)
                if result:
                    f.write(f"{os.path.basename(result)} 0\n")
                    processed_paths.append(result)

                # Cleanup between videos
                gc.collect()
                if self.use_gpu:
                    torch.cuda.empty_cache()

        print(f"✅ Written output to: {self.real_output_txt_path}")
        return processed_paths

    def main_single(self, real_video_single):
        """
        Processes a single video and appends to output TXT file.
        """
        processed_paths = []
        print(f"📂 Processing single video: {real_video_single}")

        result = self.process_video(real_video_single)
        if result:
            with open(self.real_output_txt_path, 'a') as f:
                f.write(f"{os.path.basename(result)} 0\n")
            processed_paths.append(result)

        gc.collect()
        if self.use_gpu:
            torch.cuda.empty_cache()

        return processed_paths
