import cv2
import numpy as np
import mmcv
import os

# Set video path
video_path = "../thesis_main_files/datasets/processed/lav_df/train/000473.mp4"

# Open video file with FFMPEG backend
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"🚨 ERROR: Could not open video file: {video_path}")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"📽 Video Info: Frames={frame_count}, FPS={fps}, Resolution={width}x{height}")

# Define new resize dimensions
new_w, new_h = 112, 112  # Modify as needed

# Create debug folder
debug_folder = "debug_resized_frames"
os.makedirs(debug_folder, exist_ok=True)

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    # Convert frame explicitly to a valid NumPy array
    frame = np.ascontiguousarray(frame,dtype = 'uint8')
    if not ret:
        print("🎬 Video ended or frame could not be read.")
        break

    print(f"\n🔍 Debugging Resize for Frame {frame_index}/{frame_count}")

    # 🛠 Check if frame is None or empty
    # if frame is None or not isinstance(frame, np.ndarray):
    #     print(f"🚨 ERROR: Frame {frame_index} is None or not a valid NumPy array! Skipping...")
    #     frame_index += 1
    #     continue

    if frame.size == 0:
        print(f"🚨 ERROR: Frame {frame_index} is empty! Skipping...")
        frame_index += 1
        continue

    # Print frame properties
    try:
        print(f"🟡 Frame {frame_index} - Shape: {frame.shape}, Dtype: {frame.dtype}")
    except Exception as e:
        print(f"🚨 ERROR: Unable to get frame properties: {e}")
        frame_index += 1
        continue

    # Normalize float images and convert to uint8
    if frame.dtype in [np.float32, np.float64]:
        print(f"⚠️ WARNING: Frame {frame_index} is float. Normalizing to uint8...")
        frame = (frame * 255).clip(0, 255).astype(np.uint8)

    # Handle grayscale frames (ensure correct shape)
    if len(frame.shape) == 2:  # If grayscale, convert to 3-channel
        print(f"⚠️ WARNING: Frame {frame_index} is grayscale. Expanding to 3-channel image.")
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:  # If single-channel in 3D format
        print(f"⚠️ WARNING: Frame {frame_index} has a single-channel dimension. Removing extra axis.")
        frame = np.squeeze(frame, axis=-1)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)



    # 🔹 Resize using OpenCV
    try:
        resized_cv2 = cv2.resize(frame, (112,112), interpolation=cv2.INTER_LINEAR)
        print(f"🟢 OpenCV resized Frame {frame_index} to {112}x{112}")
        # Uncomment to save resized images
        # cv2.imwrite(os.path.join(debug_folder, f"resized_cv2_{frame_index}.png"), resized_cv2)
    except Exception as e:
        print(f"🚨 OpenCV Resize Error for Frame {frame_index}: {e}")

    # 🔹 Resize using mmcv
    try:
        resized_mmcv = mmcv.imresize(frame, (112,112))
        print(f"🟢 mmcv resized Frame {frame_index} to {112}x{112}")
        # Uncomment to save resized images
        # cv2.imwrite(os.path.join(debug_folder, f"resized_mmcv_{frame_index}.png"), resized_mmcv)
    except Exception as e:
        print(f"🚨 mmcv Resize Error for Frame {frame_index}: {e}")

    frame_index += 1  # Ensure the index moves forward

cap.release()
print("✅ Video resize debugging complete.")
