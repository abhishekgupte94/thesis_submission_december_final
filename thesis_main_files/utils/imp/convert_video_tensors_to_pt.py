from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import torch


def _to_cthw_uint8(frames_thwc_uint8: torch.Tensor) -> torch.Tensor:
    """
    (T,H,W,C) uint8 -> (C,T,H,W) uint8
    """
    return frames_thwc_uint8.to(torch.uint8).permute(3, 0, 1, 2).contiguous()


def export_mp4_segments_to_sibling_video_pt_dir(
    root_dir: str | Path,
    *,
    resize_hw: Optional[Tuple[int, int]] = None,  # (H,W) optional
    overwrite: bool = False,
) -> None:
    """
    Walk root_dir recursively, find *.mp4 segments, and save:

        <mp4_parent>/video_pt/<mp4_stem>.pt

    Example:
        .../seg_0001/seg_0001.mp4
        -> .../seg_0001/video_pt/seg_0001.pt

    Saves:
        {
          "video_uint8_cthw": (3,T,H,W) uint8,
          "orig_T": int,
          "src_mp4": str,
        }

    NO temporal padding, NO sampling.
    """

    root_dir = Path(root_dir)

    try:
        from torchvision.io import read_video
    except Exception as e:
        raise RuntimeError(
            "torchvision video IO is required. "
            "If unavailable, we can switch to OpenCV."
        ) from e

    if resize_hw is not None:
        import torch.nn.functional as F

    mp4s = list(root_dir.rglob("*.mp4"))
    written = 0

    for mp4_path in mp4s:
        out_dir = mp4_path.parent / "video_pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pt = out_dir / f"{mp4_path.stem}.pt"

        if out_pt.exists() and not overwrite:
            continue

        # Decode
        frames, _, _ = read_video(str(mp4_path), pts_unit="sec")  # (T,H,W,C)
        if frames.numel() == 0:
            continue

        frames = frames.to(torch.uint8)
        orig_T = int(frames.shape[0])

        # # Optional resize (spatial only)
        # if resize_hw is not None:
        #     Ht, Wt = resize_hw
        #     x = frames.permute(0, 3, 1, 2).float()  # (T,C,H,W)
        #     x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
        #     frames = (
        #         x.round()
        #          .clamp(0, 255)
        #          .to(torch.uint8)
        #          .permute(0, 2, 3, 1)
        #          .contiguous()
        #     )

        # payload = {
        #     "video_uint8_cthw": _to_cthw_uint8(frames),  # (3,T,H,W)
        #     "orig_T": orig_T,
        #     "src_mp4": str(mp4_path),
        # }

        torch.save(_to_cthw_uint8(frames), out_pt)
        written += 1

    print(f"[export] found={len(mp4s)} mp4s, written={written} pt files")
