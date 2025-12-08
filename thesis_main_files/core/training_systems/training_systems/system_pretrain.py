# system_pretrain.py
from typing import Dict, Any

import torch
from torch import nn
import pytorch_lightning as pl

# [ADDED IMPORT]
# fvcore provides a FLOPs counter implementation from Meta.
from fvcore.nn import FlopCountAnalysis


class AVPretrainSystem(pl.LightningModule):
    def __init__(
        self,
        architecture: nn.Module,
        lr: float = 1e-4,
        # [ADDED FIELD]
        # This flag allows you to turn FLOPs profiling on/off from your config.
        enable_flop_profile: bool = True,
        # ... your other arguments ...
    ):
        super().__init__()

        # [UNCHANGED] your existing assignments
        self.architecture = architecture
        self.lr = lr
        # ...

        # [ADDED FIELD]
        # Internal flag to ensure we only run the FLOP profiler once
        # (on the first training batch).
        self.enable_flop_profile = enable_flop_profile
        self._flop_profiled = False

    # ------------------------------------------------------------------
    # [ADDED METHOD] One-shot FLOPs profiler
    # ------------------------------------------------------------------
    def _profile_flops_once(self, batch: Dict[str, Any]) -> None:
        """
        Run FLOPs analysis on the underlying architecture using a single,
        tiny batch (usually just 1 sample). We call this from the *first*
        training_step and set a flag so it never runs again.

        This gives you:
        - flops_forward: FLOPs for forward pass per sample
        - flops_train_step_est: approximate FLOPs for forward + backward + update
        """

        # If we've already profiled, or the user disabled profiling, do nothing.
        if self._flop_profiled or not self.enable_flop_profile:
            return

        self._flop_profiled = True  # Mark as done to avoid repeated profiling.

        # [ADDED] Construct a minimal batch with 1 sample on the current device.
        # IMPORTANT: Adjust the keys here to match your actual batch structure.
        sample = {}

        # Example for dict batches:
        # batch["audio_tokens"]: [B, Sa, Da]
        # batch["video_tokens"]: [B, Sv, Dv]
        if "audio_tokens" in batch:
            sample["audio_tokens"] = batch["audio_tokens"][:1].to(self.device)
        if "video_tokens" in batch:
            sample["video_tokens"] = batch["video_tokens"][:1].to(self.device)

        # [ADDED] Run FLOPs analysis on the architecture with your sample batch.
        # The architecture should accept a dict and return its usual outputs.
        fca = FlopCountAnalysis(self.architecture, sample)

        # [ADDED] total() gives you FLOPs for a single forward pass.
        total_flops = fca.total()

        # [ADDED] Rough estimate: Backward & optimizer are ≈ 2× forward FLOPs.
        # So train step ≈ 3× forward FLOPs. You can change the factor if you want.
        train_step_flops = total_flops * 3.0

        # [ADDED] Log to Lightning: you can inspect these in TensorBoard/W&B.
        # They are logged ONCE at the first step.
        self.log(
            "flops_forward",
            float(total_flops),
            prog_bar=True,
            rank_zero_only=True,
        )
        self.log(
            "flops_train_step_est",
            float(train_step_flops),
            prog_bar=True,
            rank_zero_only=True,
        )

        # Optional debug print – only on rank 0 in DDP.
        if self.trainer is not None and self.trainer.is_global_zero:
            print(f"[FLOP PROFILER] Forward FLOPs: {total_flops:,.0f}")
            print(
                f"[FLOP PROFILER] Train-step FLOPs (≈forward+backward): "
                f"{train_step_flops:,.0f}"
            )

    # ------------------------------------------------------------------
    # [MODIFIED] training_step – now calls the FLOPs profiler once
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Standard Lightning training_step. We've added a single line
        that triggers the FLOPs measurement once on the first batch.
        """

        # [ADDED LINE] Run one-shot FLOP profiling on the first batch.
        self._profile_flops_once(batch)

        # [UNCHANGED LOGIC] This should match your existing training code.
        # Adjust names like "logits" / dict keys / loss type as needed.

        # Example assumption:
        # out = self.architecture(batch) returns a dict with "logits" key
        out = self.architecture(batch)

        # Extract logits from your architecture's output
        logits = out["logits"]  # <-- change key if your dict is different

        # Labels from batch (adjust key if needed)
        labels = batch["label"].to(self.device)

        # Standard classification loss
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # Log loss as usual
        self.log("train_loss", loss, prog_bar=True)

        return loss

    # ------------------------------------------------------------------
    # [UNCHANGED] rest of your Lightning methods: configure_optimizers, etc.
    # ------------------------------------------------------------------
    # def configure_optimizers(self):
    #     ...
