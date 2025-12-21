# training_backbones_init.py
# ============================================================
# Example: Backbone construction for training
# ============================================================

from scripts.feature_extraction.SWIN.main.build_swin3d import (
    build_swin3d_backbone,
    BuildSwin3DConfig,
)

from scripts.feature_extraction.SWIN.main.build_swin2d import (
    build_swin2d_backbone,
    BuildSwin2DConfig,
)

import torch


def build_backbones_for_training():
    """
    Builds video (Swin-3D) and audio (Swin-2D) backbones.
    No forward pass, no device placement, no freezing.
    """

    # --------------------------------------------------------
    # Video backbone (Swin-3D)
    # --------------------------------------------------------
    swin3d_cfg = BuildSwin3DConfig(
        pretrained=None,          # or path to .pth
        pretrained2d=True,
        patch_size=(4, 4, 4),
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(2, 7, 7),
        drop_path_rate=0.2,
        use_checkpoint=False,
        out="5d",                 # "5d" or "bd"
    )

    video_backbone = build_swin3d_backbone(swin3d_cfg)

    # --------------------------------------------------------
    # Audio backbone (Swin-2D)
    # --------------------------------------------------------
    swin2d_cfg = BuildSwin2DConfig(
        yaml_path=None,           # optional Swin YAML
        img_size=(224, 224),
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        use_checkpoint=False,
    )

    audio_backbone = build_swin2d_backbone(swin2d_cfg)

    return video_backbone, audio_backbone


# # ------------------------------------------------------------
# # Optional local sanity check (safe to delete)
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     video_backbone, audio_backbone = build_backbones_for_training()
#
#     print("[OK] Video backbone:", type(video_backbone))
#     print("[OK] Audio backbone:", type(audio_backbone))
