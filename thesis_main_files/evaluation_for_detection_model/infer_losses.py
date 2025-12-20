# inference/min_eval_ssl_losses.py
from __future__ import annotations

"""
============================================================
min_eval_ssl_losses.py

ROLE
----
Minimal SSL evaluation script that:
- does NOT assume offline_root / batch_name
- reuses YOUR existing dataloader / feature pipeline
- runs model forward passes (SSL loss-only)
- supports single GPU or DDP (torchrun) if you return a Dataset

You must implement ONLY:
1) get_eval_loader_or_dataset(...)
2) batch_to_model_inputs(...)

Everything else is stable and copy-paste ready.
============================================================
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

import torch

from evaluation_for_detection_model.build_model import build_model, load_weights_into_model, BuildModelArgs
from evaluation_for_detection_model.ddp_multi_gpu_prep import rank, is_dist, allreduce_mean, distributed_eval_loader


# ============================================================
# [STUB 1] Plug in your existing loader OR dataset
# ============================================================
def get_eval_loader_or_dataset(args) -> Union[torch.utils.data.DataLoader, torch.utils.data.Dataset]:
    """
    Return either:
      A) a torch.utils.data.DataLoader  (recommended if you already have one), OR
      B) a torch.utils.data.Dataset     (recommended for DDP; script will wrap it)

    Examples:
    ---------
    - If your main training script already builds a datamodule/loader:
        from main_trainer_pretrain import build_eval_loader
        return build_eval_loader(limit_samples=args.n_samples)

    - If you have a Dataset and want DDP sharding here:
        return MyDataset(...)

    REQUIRED:
    ---------
    The batch produced must contain (or be convertible into):
      - audio tensor  shape ~ (B, n_mels, T) or (B, 1, n_mels, T)
      - video tensor  shape ~ (B, C, T, H, W)  (typically uint8)
    """
    raise NotImplementedError("Implement get_eval_loader_or_dataset(args) to return your loader/dataset.")


# ============================================================
# [STUB 2] Define how a batch is fed into your model
# ============================================================
def batch_to_model_inputs(batch: Any, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert your batch to (video, audio) inputs expected by AVPretrainArchitecture forward().

    Expected by your architecture (based on training code):
      - audio_in: (B, 1, n_mels, T)
      - video_in: (B, C, T, H, W) in float32 with 0..1 range

    Modify ONLY this function if your batch keys/shapes differ.
    """

    # ---------------------------
    # [DEFAULT ASSUMPTION]
    # ---------------------------
    # If your batch is a dict like:
    #   batch["audio"]         -> (B, n_mels, T)
    #   batch["video_u8_cthw"] -> (B, C, T, H, W)  uint8
    #
    # Adjust these keys to your actual batch keys.
    audio = batch["audio"].to(device, non_blocking=True)
    video = batch["video_u8_cthw"].to(device, non_blocking=True)

    # Ensure audio has channel dim (B,1,n_mels,T)
    if audio.dim() == 3:
        audio = audio.unsqueeze(1)

    # Normalize video to float [0,1]
    if video.dtype == torch.uint8:
        video = video.float().div_(255.0)
    else:
        video = video.float()

    return video, audio


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights-pt", type=str, default="checkpoints/best_weights.pt")
    p.add_argument("--out", type=str, default="ssl_eval_report.txt")

    # Optional knobs (useful if your stub supports them)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=500, help="Your intended eval budget")
    p.add_argument("--limit-batches", type=int, default=0, help="0 = no limit")

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------- Build + load model --------------------
    model = build_model(BuildModelArgs(device=device, freeze_backbones=True))
    epoch, gstep = load_weights_into_model(model, args.weights_pt, strict=True)

    if rank() == 0:
        mode = "DDP" if is_dist() else "SingleGPU/CPU"
        print(f"[Loaded] {args.weights_pt} (epoch={epoch}, global_step={gstep}) | mode={mode}")

    # -------------------- Get loader or dataset from your pipeline --------------------
    obj = get_eval_loader_or_dataset(args)

    # If user returns a Dataset, wrap it with a DDP-safe loader here.
    if isinstance(obj, torch.utils.data.Dataset):
        loader = distributed_eval_loader(
            obj,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
    else:
        # user already returned a loader; we assume it's correct.
        # NOTE: if this loader is not using DistributedSampler under DDP,
        # each rank may process the same samples (duplication).
        loader = obj

    # -------------------- SSL evaluation loop --------------------
    sum_total = torch.tensor(0.0, device=device)
    sum_vacl  = torch.tensor(0.0, device=device)
    sum_cpe   = torch.tensor(0.0, device=device)
    n = torch.tensor(0.0, device=device)

    for bi, batch in enumerate(loader):
        if args.limit_batches and bi >= args.limit_batches:
            break

        video_in, audio_in = batch_to_model_inputs(batch, device=device)
        out: Dict[str, torch.Tensor] = model(video_in=video_in, audio_in=audio_in)

        sum_total += out["loss_total"]
        sum_vacl  += out["loss_vacl"]
        sum_cpe   += out["loss_cpe"]
        n += 1.0

    mean_total = allreduce_mean(sum_total / torch.clamp(n, min=1.0))
    mean_vacl  = allreduce_mean(sum_vacl  / torch.clamp(n, min=1.0))
    mean_cpe   = allreduce_mean(sum_cpe   / torch.clamp(n, min=1.0))

    if rank() == 0:
        report = (
            f"SSL Eval Report\n"
            f"==============\n"
            f"weights_pt: {args.weights_pt}\n"
            f"epoch: {epoch}\n"
            f"global_step: {gstep}\n"
            f"batches_seen: {int(n.item())}\n"
            f"\n"
            f"mean_loss_total: {float(mean_total.cpu()):.6f}\n"
            f"mean_loss_vacl : {float(mean_vacl.cpu()):.6f}\n"
            f"mean_loss_cpe  : {float(mean_cpe.cpu()):.6f}\n"
        )

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)

        print(report)
        print(f"[Wrote] {out_path}")


if __name__ == "__main__":
    main()



