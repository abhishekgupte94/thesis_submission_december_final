
"""
custom_swin_backbone_wrapper.py

Purpose:
    Provide a clean, Swin-repo-compatible way to build *your* customised
    Swin backbone that:
      - is trained FROM SCRATCH (or with whatever pretrained you choose),
      - is used as a FEATURE EXTRACTOR inside a larger architecture,
      - already returns (logits, features) in its `forward()` method,
      - exposes the same weight-decay skip APIs as the official Swin model,
        so that `optimizer.build_optimizer(config, model)` still works.

This file DOES NOT:
      - define a training loop,
      - define any loss,
      - parse CLI args,
      - know about the big architecture’s logic.

Your external trainer is responsible for:
      - constructing `config`,
      - creating DataLoaders,
      - calling `build_custom_swin_backbone(config)`,
      - wiring logits/features into the big architecture and loss,
      - stepping the optimizer, scheduler, etc.
"""

from typing import Optional

import torch
from torch import nn

# This should point to YOUR modified backbone file.
# If you already replaced the SwinTransformer in swin_transformer.py,
# import that here. Otherwise, adjust the import to your actual class.
from swin_transformer import SwinTransformer  # <-- your customised version


class CustomSwinBackbone(nn.Module):
    """
    Thin wrapper around your customised Swin backbone.

    Assumptions:
      - The underlying Swin model's forward(...) ALREADY returns:
            logits, features = model(x)
      - You want to use both for downstream logic:
          * logits:   e.g. for supervised pretext or debugging
          * features: for the big self-supervised architecture's encoder space

    This wrapper:
      - Only standardises the __init__ interface around the Swin config.
      - Delegates no_weight_decay / no_weight_decay_keywords so that the
        existing optimizer and param grouping logic from the Swin repo
        continues to work unchanged.
      - Does NOT define any loss or training code.
    """

    def __init__(self, config, num_classes: Optional[int] = None):
        """
        Args:
            config: Swin-style config node.
                    We expect fields like:
                        config.MODEL.SWIN.IMG_SIZE
                        config.MODEL.SWIN.PATCH_SIZE
                        config.MODEL.SWIN.IN_CHANS
                        config.MODEL.SWIN.EMBED_DIM
                        config.MODEL.SWIN.DEPTHS
                        config.MODEL.SWIN.NUM_HEADS
                        config.MODEL.SWIN.WINDOW_SIZE
                        config.MODEL.SWIN.MLP_RATIO
                        config.MODEL.SWIN.QKV_BIAS
                        config.MODEL.SWIN.QK_SCALE
                        config.MODEL.DROP_RATE
                        config.MODEL.ATTN_DROP_RATE
                        config.MODEL.DROP_PATH_RATE
                        config.MODEL.SWIN.APE
                        config.MODEL.SWIN.PATCH_NORM
                        config.TRAIN.USE_CHECKPOINT

            num_classes: if provided, overrides config.MODEL.NUM_CLASSES.
                         This is the dimension of the `logits` output.
                         For pure feature-extraction / SSL, you can set
                         this to 0 (or ignore logits entirely later).
        """
        super().__init__()

        if num_classes is None:
            num_classes = config.MODEL.NUM_CLASSES

        # Instantiate your modified Swin backbone.
        # IMPORTANT:
        #   - Your SwinTransformer.forward(...) MUST return (logits, features).
        #   - If you've kept the original API, you can add that behaviour there.
        self.backbone = SwinTransformer(
            img_size=config.MODEL.SWIN.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=num_classes,  # <- logits dim
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            attn_drop_rate=config.MODEL.ATTN_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

        # Expose feature dimensionality for heads / big-arch plugins.
        # In the official Swin code this is `num_features`.
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Delegates directly to your customised Swin backbone, which is
        expected to return:

            logits, features = self.backbone(x)

        Returns:
            logits:   [B, num_classes]
            features: [B, D]         (D = self.feature_dim)
        """
        logits, features = self.backbone(x)
        return logits, features

    # ------------------------------------------------------------------
    # Delegation for optimizer weight-decay handling
    # ------------------------------------------------------------------
    def no_weight_decay(self):
        """
        Delegate to underlying backbone if it defines a skip list.
        This keeps compatibility with optimizer.build_optimizer(...)
        from the Swin repo.
        """
        if hasattr(self.backbone, "no_weight_decay"):
            return self.backbone.no_weight_decay()
        return set()

    def no_weight_decay_keywords(self):
        """
        Delegate to underlying backbone if it defines skip keywords.
        This is used by the Swin optimizer logic to group parameters.
        """
        if hasattr(self.backbone, "no_weight_decay_keywords"):
            return self.backbone.no_weight_decay_keywords()
        return set()


def build_custom_swin_backbone(config, num_classes: Optional[int] = None) -> CustomSwinBackbone:
    """
    Helper function to mirror the style of models.build_model(config),
    but explicitly build your CustomSwinBackbone.

    Typical usage in your OUT-OF-SCRIPT trainer:

        from custom_swin_backbone_wrapper import build_custom_swin_backbone
        from optimizer import build_optimizer
        from lr_scheduler import build_scheduler

        backbone = build_custom_swin_backbone(config)
        backbone.cuda()

        optimizer = build_optimizer(config, backbone)
        scheduler = build_scheduler(config, optimizer, steps_per_epoch)

        # training loop:
        for batch in train_loader:
            x, y = ...
            logits, features = backbone(x)
            loss = my_big_arch_loss_fn(logits, features, batch, config)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    Args:
        config:       Swin config node.
        num_classes:  Optional override for logits dimension.

    Returns:
        CustomSwinBackbone instance, ready to be passed into your trainer.
    """
    model = CustomSwinBackbone(config, num_classes=num_classes)
    return model