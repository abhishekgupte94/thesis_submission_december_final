"""
system_pretrain.py
==================

Self-supervised pretraining system:

    Architecture: Swin + A + B
    Loss: dummy L2 alignment (replace with your NPV-style SSL loss)
"""

from typing import Dict, Any
import torch
import pytorch_lightning as pl
from scripts.feature_extraction.main.main_feature_extraction_wrapper import DualSwinBackbone
from core.NPVForensics.VACL_block.main.vacl_wrapper import VACLProjectionHead
from core.NPVForensics.common_projection.main.common_projection_head_module_wrapper import FaceAudioCommonSpaceWrapper

from training_systems.architectures.pretrain_architecture import AVPretrainArchitecture


class AVPretrainSystem(pl.LightningModule):
    """
    [NEW] Self-supervised pretraining system using:
        Architecture: Swin + A + B
        Data: dict batches from AVSegmentTokenDataModule.
    """

    def __init__(
        self,
        swin_backbone: DualSwinBackbone,
        d_swin: int,      # expected fused dim (Da + Dv, or whatever _combine_av outputs)
        d_a: int,
        d_b: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["swin_backbone"])

        # Heads
        self.module_vacl = VACLProjectionHead(d_v,d_a,seq_len,k,out_dim,mu = 0.5,input_layout = "bsd",pool= "mean")

        self.module_common_projection = FaceAudioCommonSpaceWrapper(d_a,d_f,d_fa, temperature = 0.1,
                        loss_weight= 1.0)

        # Architecture: Swin + A + B
        self.arch = AVPretrainArchitecture(
            swin_backbone=swin_backbone,
            module_a=self.module_vacl,
            module_b=self.module_common_projection,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward through pretraining architecture.
        """
        return self.arch(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        [MODIFIED] Strategy A:
          - batch is dict (no Swin in DataLoader).
        """
        outputs = self.forward(batch)

        z_a = outputs["module_vacl_out"]  # [B, d_a]
        z_b = outputs["module_projection_head_out"]  # [B, d_b]

        # Dummy SSL loss: L2 alignment (replace with real loss)
        if z_a.shape != z_b.shape:
            raise ValueError("For this dummy SSL loss, z_a and z_b must have same shape.")

        loss = ((z_a - z_b) ** 2).mean()

        self.log(
            "train_pretrain_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        outputs = self.forward(batch)
        z_a = outputs["module_vacl_out"]
        z_b = outputs["module_projection_head_out"]

        if z_a.shape != z_b.shape:
            raise ValueError("For this dummy SSL loss, z_a and z_b must have same shape.")

        loss = ((z_a - z_b) ** 2).mean()
        self.log(
            "val_pretrain_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        """
        Optimizer for Swin + A + B during pretraining.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer is not None else 100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_pretrain_loss",
            },
        }
