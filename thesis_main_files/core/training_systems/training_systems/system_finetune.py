"""
system_finetune.py
==================

Supervised fine-tuning system:

    Architecture: Swin (frozen) + A + B + C
    Data: dict batches with 'label'
"""

from typing import Dict, Any
import torch
from torch import nn
import pytorch_lightning as pl

from dual_swin_backbone import DualSwinBackbone
from heads_av import ModuleA, ModuleB, ModuleC
from architectures_av import AVFinetuneArchitecture


class AVFinetuneSystem(pl.LightningModule):
    """
    [NEW] Supervised fine-tuning system using:
        Architecture: Swin + A + B + C
        Data: dict batches (must include 'label').
    """

    def __init__(
        self,
        swin_backbone: DualSwinBackbone,
        d_swin: int,
        d_a: int,
        d_b: int,
        num_classes: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["swin_backbone"])

        # Heads
        self.module_a = ModuleA(in_dim=d_swin, out_dim=d_a)
        self.module_b = ModuleB(in_dim=d_swin, out_dim=d_b)
        self.module_c = ModuleC(in_dim=d_b, num_classes=num_classes)

        # Architecture: Swin + A + B + C
        self.arch = AVFinetuneArchitecture(
            swin_backbone=swin_backbone,
            module_a=self.module_a,
            module_b=self.module_b,
            module_c=self.module_c,
        )

        # 🔒 [FREEZE_BACKBONE] — freeze Swin for fine-tuning
        self.arch.swin_backbone.freeze()

        # Optional sanity counts
        n_backbone = sum(p.numel() for p in self.arch.swin_backbone.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Finetune] Frozen Swin params: {n_backbone}")
        print(f"[Finetune] Trainable params (A/B/C + others): {n_trainable}")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward through fine-tuning architecture.
        """
        return self.arch(batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        [MODIFIED] Strategy A, supervised:
          - batch dict includes 'label'
          - use module_c_out for loss
        """
        outputs = self.forward(batch)
        logits = outputs["module_c_out"]  # [B, num_classes]
        labels = batch["label"]           # [B]

        loss = self.ce_loss(logits, labels)

        self.log(
            "train_finetune_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        outputs = self.forward(batch)
        logits = outputs["module_c_out"]
        labels = batch["label"]

        loss = self.ce_loss(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()

        self.log("val_finetune_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_finetune_acc", acc,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Optimizer for Swin + A + B + C during fine-tuning.
        Swin is frozen (requires_grad=False), so only A/B/C are updated.
        """
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer is not None else 50,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_finetune_loss",
            },
        }
