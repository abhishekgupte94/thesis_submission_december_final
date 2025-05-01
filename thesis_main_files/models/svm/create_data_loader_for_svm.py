import torch
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm
from thesis_main_files.utils.files_imp import create_manifest_from_selected_files

class FeatureBuilder:
    @staticmethod
    def get_encoded_features_for_svm(
        dataloader,
        model,
        feature_processor,
        output_txt_path,
        binary_label_fn,
        device=None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
            model = torch.nn.DataParallel(model)

        model = model.to(device)
        model.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for audio_paths, video_paths, labels in tqdm(dataloader, desc="Encoding features for SVM"):
                create_manifest_from_selected_files(video_paths, output_txt_path)

                audio_feats, video_feats = feature_processor.create_datasubset(
                    csv_path=None,
                    use_preprocessed=False,
                    video_paths=video_paths
                )

                if audio_feats is None or video_feats is None:
                    print("⚠️ Skipping batch due to feature extraction failure.")
                    continue

                audio_feats = audio_feats.to(device)
                video_feats = video_feats.to(device)

                f_art, f_lip = model(audio_features=audio_feats, video_features=video_feats)
                combined_feats = torch.cat([f_art, f_lip], dim=1).cpu()  # (B x 512)

                binary_labels = torch.tensor(
                    [binary_label_fn(lbl) for lbl in labels],
                    dtype=torch.long
                )

                all_features.append(combined_feats)
                all_labels.append(binary_labels)

        if not all_features:
            raise RuntimeError("❌ No valid features extracted for SVM.")

        features_tensor = torch.cat(all_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        return TensorDataset(features_tensor, labels_tensor)

    @staticmethod
    def encode_and_save_dataset_npz(
        dataloader,
        model,
        feature_processor,
        output_txt_path,
        binary_label_fn,
        save_path_npz,
        device=None
    ):
        """
        Wrapper that encodes features using get_encoded_features_for_svm(...)
        and saves the resulting TensorDataset as a .npz file.

        Returns:
            TensorDataset
        """
        dataset = FeatureBuilder.get_encoded_features_for_svm(
            dataloader=dataloader,
            model=model,
            feature_processor=feature_processor,
            output_txt_path=output_txt_path,
            binary_label_fn=binary_label_fn,
            device=device
        )

        features = dataset.tensors[0].numpy()
        labels = dataset.tensors[1].numpy()
        np.savez_compressed(save_path_npz, features=features, labels=labels)
        print(f"✅ Saved SVM dataset to: {save_path_npz} | Shape: {features.shape}")

        return dataset
