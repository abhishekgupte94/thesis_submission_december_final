# evaluation_for_detection_model/infer_losses.py
from __future__ import annotations

"""
============================================================
infer_losses.py

ROLE (SSL GENERALIZATION CHECK)
------------------------------
Runs the SSL model on a chosen split (train/val) and reports:
- loss_total
- loss_vacl
- loss_cpe

This is "evaluation_for_detection_model" for SSL:
✓ objective generalization (val vs train)
✗ not classification
============================================================
"""

import argparse
from pathlib import Path
import csv
import torch

from dataloader import SegmentDataModule
from evaluation_for_detection_model.build_model import build_model, load_weights_into_model, BuildModelArgs


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--offline-root", type=str, required=True)
    p.add_argument("--batch-name", type=str, required=True)
    p.add_argument("--weights-pt", type=str, default="checkpoints/best_weights.pt")

    p.add_argument("--split", type=str, choices=["train", "val"], default="val")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit-batches", type=int, default=0, help="0 = no limit")
    p.add_argument("--out-csv", type=str, default="ssl_inference_losses.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------
    # [DATA] Reuse your existing DM exactly
    # ------------------------------------------------------------
    dm = SegmentDataModule(
        offline_root=Path(args.offline_root),
        batch_name=args.batch_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        map_location="cpu",
        seed=123,
        val_split=0.05,
        val_batch_size=args.batch_size,
    )
    dm.setup(stage="fit")
    loader = dm.val_dataloader() if args.split == "val" else dm.train_dataloader()

    # ------------------------------------------------------------
    # [MODEL] Build topology, then load trained weights
    # ------------------------------------------------------------
    model = build_model(BuildModelArgs(device=device, freeze_backbones=True))
    epoch, gstep = load_weights_into_model(model, args.weights_pt, strict=True)
    print(f"[Loaded] {args.weights_pt} (epoch={epoch}, global_step={gstep})")

    # ------------------------------------------------------------
    # [SSL INFERENCE LOOP]
    # ------------------------------------------------------------
    sum_total = 0.0
    sum_vacl = 0.0
    sum_cpe = 0.0
    n = 0

    rows = []

    for bi, batch in enumerate(loader):
        if args.limit_batches and bi >= args.limit_batches:
            break

        # NOTE: these keys match the ones we assumed earlier.
        # If your dataloader uses different keys, change ONLY these lines.
        audio = batch["audio"].to(device, non_blocking=True).unsqueeze(1)          # (B,1,n_mels,T)
        video = batch["video_u8_cthw"].to(device, non_blocking=True).float().div_(255.0)

        out = model(video_in=video, audio_in=audio)

        lt = float(out["loss_total"].detach().cpu())
        lv = float(out["loss_vacl"].detach().cpu())
        lc = float(out["loss_cpe"].detach().cpu())

        sum_total += lt
        sum_vacl += lv
        sum_cpe += lc
        n += 1

        rows.append({"batch_idx": bi, "loss_total": lt, "loss_vacl": lv, "loss_cpe": lc})

    mean_total = sum_total / max(n, 1)
    mean_vacl = sum_vacl / max(n, 1)
    mean_cpe = sum_cpe / max(n, 1)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch_idx", "loss_total", "loss_vacl", "loss_cpe"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({"batch_idx": "MEAN", "loss_total": mean_total, "loss_vacl": mean_vacl, "loss_cpe": mean_cpe})

    print(f"[Wrote] {out_path}")
    print(f"[MEAN] total={mean_total:.6f}  vacl={mean_vacl:.6f}  cpe={mean_cpe:.6f}")


if __name__ == "__main__":
    main()
