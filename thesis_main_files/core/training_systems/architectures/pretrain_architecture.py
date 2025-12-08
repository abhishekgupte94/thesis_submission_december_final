"""
architectures_av.py
===================

Two distinct architectures:

    1) AVPretrainArchitecture : Swin + A + B
    2) AVFinetuneArchitecture : Swin + A + B + C (C on top of B)

Both expect:
    - swin_backbone : DualSwinBackbone (audio+video)
    - module_a      : ModuleA
    - module_b      : ModuleB
    - module_c      : ModuleC (only for finetune)
"""

from typing import Dict, Any
import torch
from torch import nn
from scripts.feature_extraction.main.main_feature_extraction_wrapper import DualSwinBackbone
from core.NPVForensics.VACL_block.main.vacl_wrapper import VACLProjectionHead
from core.NPVForensics.common_projection.main.common_projection_head_module_wrapper import FaceAudioCommonSpaceWrapper
class AVPretrainArchitecture(nn.Module):
    """
    [NEW] Architecture used ONLY for pretraining:
        Swin + Module A + Module B

    Consumes dict-style batches and returns:
        - swin_features   (fused AV representation)
        - module_a_out
        - module_b_out
        - batch_metadata
    """

    def __init__(
        self,
        swin_backbone: nn.Module,  # DualSwinBackbone
        module_common_projection: nn.Module,
        module_vacl_projection: nn.Module,
    ) -> None:
        super().__init__()
        self.swin_backbone = swin_backbone(audio_cfg_path,
                 video_cfg_path,
                 audio_opts=None,
                 video_opts=None)
        self.module_common_projection = module_common_projection(d_a,d_f,d_fa, temperature = 0.1,
                        loss_weight= 1.0)
        self.module_vacl_projection = module_vacl_projection(d_v,d_a,seq_len,k,out_dim,mu = 0.5,input_layout = "bsd",pool= "mean")

    def _combine_av(
        self,
        feats_a: torch.Tensor,
        feats_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        [NEW] Simple AV fusion example:
            - pool each over time if needed
            - concatenate along feature dimension

        You can replace this with your real AV fusion / projector.
        """
        if feats_a.dim() == 3:
            feats_a = feats_a.mean(dim=1)  # [B, T_a, Da] -> [B, Da]
        if feats_v.dim() == 3:
            feats_v = feats_v.mean(dim=1)  # [B, T_v, Dv] -> [B, Dv]
        swin_features = torch.cat([feats_a, feats_v], dim=-1)  # [B, Da+Dv]
        return swin_features

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected batch keys:
          - "audio_tokens": [B, N_tokens_a, D_a]
          - "video_tokens": [B, N_tokens_v, D_v]
        """
        audio_tokens: torch.Tensor = batch["audio_tokens"]
        video_tokens: torch.Tensor = batch["video_tokens"]

        feats_a, feats_v = self.swin_backbone(audio_tokens, video_tokens)
        swin_features = self._combine_av(feats_a, feats_v)

        z_a = self.module_a(swin_features)  # [B, d_a]
        z_b = self.module_b(swin_features)  # [B, d_b]

        out: Dict[str, Any] = {
            "swin_features": swin_features,
            "module_a_out": z_a,
            "module_b_out": z_b,
            "batch_metadata": {
                k: v
                for k, v in batch.items()
                if k not in ("audio_tokens", "video_tokens")
            },
        }
        return out


