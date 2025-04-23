# from torch.utils.data import DataLoader, TensorDataset
# import torch
# from
# def build_audio_video_deep_feature_dataset(
#     dataset: Dataset,
#     model_inference_fn,
#     binary_label_fn,
#     batch_size: int = 64,
#     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
# ) -> TensorDataset:
#     """
#     Runs inference using a model that accepts audio and video features and outputs two enhanced tensors.
#     Concatenates those outputs and stores them with binary labels.
#
#     Args:
#         dataset (Dataset): AudioVideoFeatureDataset.
#         model_inference_fn (function): (audio_feats, video_feats) -> (feat1, feat2)
#         binary_label_fn (function): label -> binary_label
#         batch_size (int): For DataLoader.
#         device (str): 'cuda' or 'cpu'.
#
#     Returns:
#         TensorDataset of concatenated features and binary labels.
#     """
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     all_features, all_labels = [], []
#
#     with torch.no_grad():
#         for audio_feats, video_feats, labels in loader:
#             audio_feats = audio_feats.to(device)
#             video_feats = video_feats.to(device)
#
#             feat1, feat2 = model_inference_fn(audio_feats, video_feats)  # Model output
#             combined_features = torch.cat([feat1, feat2], dim=1)  # Final features: [B, D1+D2]
#
#             binary_labels = torch.tensor([binary_label_fn(lbl.item()) for lbl in labels], dtype=torch.long)
#             all_features.append(combined_features.cpu())
#             all_labels.append(binary_labels)
#
#     return TensorDataset(torch.cat(all_features), torch.cat(all_labels))
