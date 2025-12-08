from typing import Dict, Any
import torch
from torch import nn


class AVFinetuneArchitecture(nn.Module):
    """
    [NEW] Architecture used ONLY for fine-tuning:
        Swin + Module A + Module B + Module C

    Module C is attached on top of Module B's output.
    """

    def __init__(
        self,
        swin_backbone: nn.Module,  # DualSwinBackbone
        module_a: nn.Module,
        module_b: nn.Module,
        module_c: nn.Module,
    ) -> None:
        super().__init__()
        self.swin_backbone = swin_backbone
        self.module_a = module_a
        self.module_b = module_b
        self.module_c = module_c

    def _combine_av(
        self,
        feats_a: torch.Tensor,
        feats_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        [NEW] Same AV fusion logic as pretrain architecture.
        """
        if feats_a.dim() == 3:
            feats_a = feats_a.mean(dim=1)
        if feats_v.dim() == 3:
            feats_v = feats_v.mean(dim=1)
        swin_features = torch.cat([feats_a, feats_v], dim=-1)
        return swin_features

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_tokens: torch.Tensor = batch["audio_tokens"]
        video_tokens: torch.Tensor = batch["video_tokens"]

        feats_a, feats_v = self.swin_backbone(audio_tokens, video_tokens)
        swin_features = self._combine_av(feats_a, feats_v)

        z_a = self.module_a(swin_features)       # [B, d_a]
        z_b = self.module_b(swin_features)       # [B, d_b]
        logits = self.module_c(z_b)              # [B, num_classes]

        out: Dict[str, Any] = {
            "swin_features": swin_features,
            "module_a_out": z_a,
            "module_b_out": z_b,
            "module_c_out": logits,
            "batch_metadata": {
                k: v
                for k, v in batch.items()
                if k not in ("audio_tokens", "video_tokens")
            },
        }
        return out
