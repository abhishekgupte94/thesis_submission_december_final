# mvit_adapter.py
from typing import List, Optional
import torch

from thesis_main_files.main_files.feature_extraction.new_file_setups.Video_Feature_Extraction.mvitv2_torchvision.extract_video_features_mvitv2_time_dimension import MViTv2FeatureExtractor

class MViTVideoFeatureExtractor:
    """
    Thin adapter that owns and drives MvitFeatureExtractor.
    - Builds the core extractor in __init__ (no external wiring).
    - Returns a batched tensor ready for the trainer.
    """
    def __init__(self,
                 device: Optional[torch.device] = None,
                 amp: bool = True,
                 strict_temporal: bool = False,
                 save_video_feats: bool = False,
                 save_dir: Optional[str] = None,
                 preserve_temporal: bool = True,
                 temporal_pool: bool = True,
                 aggregate: str = "mean"):
        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                device = torch.device("cpu")
        self.device = device
        self.strict_temporal = strict_temporal
        self.save_video_feats = save_video_feats
        self.save_dir = save_dir

        # Build the core extractor here (per your request)
        self.mvit_core = MViTv2FeatureExtractor(
            device=str(device),
            preserve_temporal=preserve_temporal,     # keep temporal richness
            temporal_pool=temporal_pool,             # produce (T', D); we can pool if mismatch
            aggregate=aggregate,                      # "none" (don’t pool) or "mean"
            dtype=(torch.float16 if amp else torch.float32),
            verbose=False,
            default_save_dir=save_dir
        )

    def execute(self, video_paths: List[str]) -> torch.Tensor:
        """
        Extract features for a list of video paths and return a batch:
          - (B, D) if per-sample vectors, or
          - (B, T', D) if temporal features and all T' match,
          - else falls back to time-mean → (B, D).
        """
        results = self.mvit_core.extract_from_paths(
            video_paths,
            save=self.save_video_feats,
            save_dir=self.save_dir,
            overwrite=False
        )

        feats = [r["features"] for r in results]  # list of CPU tensors
        shapes = {tuple(f.shape) for f in feats}
        print("Vid Feat Execute stage reached!")
        if len(shapes) == 1:
            batch = torch.stack(feats, dim=0)  # (B, D) or (B, T', D)
        else:
            if self.strict_temporal:
                raise ValueError(f"Temporal lengths differ in batch: {shapes}")
            # fallback: pool time dimension when shapes differ
            pooled = [f.mean(dim=0) if f.dim() == 2 else f for f in feats]
            batch = torch.stack(pooled, dim=0)  # (B, D)

        # Move to trainer’s device
        return batch
