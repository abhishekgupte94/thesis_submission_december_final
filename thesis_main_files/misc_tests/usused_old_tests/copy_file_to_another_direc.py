import os
import shutil

def move_lips_only_videos(source_dir, target_dir, recursive=False):
    """
    Moves all files ending with '_lips_only.mp4' from source_dir to target_dir.

    Parameters:
    - source_dir (str): The path of the source directory to search in.
    - target_dir (str): The path of the target directory to move files to.
    - recursive (bool): If True, will search subdirectories recursively.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if recursive:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('_lips_only.mp4'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_dir, file)
                    shutil.move(src_path, dst_path)
                    print(f"Moved: {src_path} -> {dst_path}")
    else:
        for file in os.listdir(source_dir):
            if file.endswith('_lips_only.mp4'):
                src_path = os.path.join(source_dir, file)
                dst_path = os.path.join(target_dir, file)
                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} -> {dst_path}")

source_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/data/processed_files/lav_df/train"
target_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/Video-Swin-Transformer/data/train/real"
move_lips_only_videos(source_path,target_path)