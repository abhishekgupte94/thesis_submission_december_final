import torch

file_path = "/Users/abhishekgupte_macbookpro/PycharmProjects/Video-Swin-Transformer/batch_features_videos.pt"
extracted_features = torch.load(file_path)


print(extracted_features.size())