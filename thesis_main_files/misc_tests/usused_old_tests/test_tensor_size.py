import torch


def display_tensor(tensor = None, video_tensor_path = None):
# Replace this with the actual path to your .pt or .pth file
#     video_tensor_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/thesis_main_files/test_files/test_2_video_embeddings/batch_features_lips.pt"
    video_tensor = None
    if tensor is None:
    # Load the video tensor
        video_tensor = torch.load(video_tensor_path)


        print("Loaded video tensor:")
        print(video_tensor)

        # Optionally, print shape and dtype
        print(f"Shape: {video_tensor.shape}")
        print(f"Dtype: {video_tensor.dtype}")
    else:
        print("Loaded video tensor:")


        print(tensor)

        # Optionally, print shape and dtype
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")

    # Print the loaded tensor
