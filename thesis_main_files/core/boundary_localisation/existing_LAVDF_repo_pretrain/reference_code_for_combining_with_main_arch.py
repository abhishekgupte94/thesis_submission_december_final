# file: your_main_model.py

import torch
from torch import nn

from batfdplus_prb_wrapper import BatfdPlusPRBExtractor


class YourBoundaryModel(nn.Module):
    """
    Your main architecture that:
      - uses your SSL encoder to produce Zsv, Zsa
      - has its own PRB / boundary pipeline
      - optionally consults the BA-TFD+ teacher for PRB outputs
    """

    def __init__(self, teacher_ckpt: str | None = None, device: str = "cuda"):
        super().__init__()

        self.device = torch.device(device)

        # --- Your components ---
        # self.ssl_encoder = ...
        # self.your_prb_module = ...
        # self.heads, losses, etc.

        # --- Optional BA-TFD+ teacher (until PRB) ---
        self.teacher = None
        if teacher_ckpt is not None:
            self.teacher = BatfdPlusPRBExtractor(
                checkpoint_path=teacher_ckpt,
                device=device,
            )

    # ------------------------------------------------
    # 1) YOUR forward path
    # ------------------------------------------------
    def forward_your_path(self, batch):
        """
        Your original forward logic.
        This is just a stub; adapt to how your code actually looks.
        """
        # Example structure:
        # video_feats, audio_feats = self.ssl_encoder(batch)
        # prb_v, prb_a = self.your_prb_module(video_feats, audio_feats)
        # logits = self.classifier_head(prb_v, prb_a, ...)
        # return {
        #   "logits": logits,
        #   "prb_v": prb_v,
        #   "prb_a": prb_a,
        # }
        raise NotImplementedError("Implement your main forward path here.")

    # ------------------------------------------------
    # 2) TEACHER path: BA-TFD+ until PRB
    # ------------------------------------------------
    def get_teacher_prb_outputs(self, batch):
        """
        Run the official BA-TFD+ model in parallel up to the PRB stage
        and return the PRB outputs. If no teacher is configured, returns None.
        """
        if self.teacher is None:
            return None

        # IMPORTANT:
        # The 'batch' format must match what BatfdPlus expects:
        #   {
        #     "video": (B, C, T, H, W),
        #     "audio": (B, 1, n_mels, T),
        #     ...
        #   }
        # If your main batch is different, adapt/construct a teacher_batch here.
        teacher_batch = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        prb_teacher = self.teacher.forward_until_prb(teacher_batch)
        return prb_teacher

    # ------------------------------------------------
    # 3) Combined "parallel" call: your path + teacher PRB
    # ------------------------------------------------
    def forward_with_teacher_prb(self, batch):
        """
        Convenience function:
          - runs your own forward path
          - runs the BA-TFD+ teacher in parallel (if available)
          - returns a dict bundling both outputs
        """
        # 1) Your main path
        your_out = self.forward_your_path(batch)

        # 2) Teacher PRB path
        teacher_prb = self.get_teacher_prb_outputs(batch)

        # 3) Bundle
        return {
            "your": your_out,
            "teacher_prb": teacher_prb,
        }
