from __future__ import annotations

import argparse
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
        torch.save((3,T,H,W) uint8, out_pt)

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
        import torch.nn.functional as F  # noqa: F401

    mp4s = list(root_dir.rglob("*.mp4"))
    written = 0

    for mp4_path in mp4s:
        out_dir = mp4_path.parent / "video_pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pt = out_dir / f"{mp4_path.stem}.pt"

        if out_pt.exists() and not overwrite:
            continue

        frames, _, _ = read_video(str(mp4_path), pts_unit="sec")  # (T,H,W,C)
        if frames.numel() == 0:
            continue

        frames = frames.to(torch.uint8)

        # NOTE: keeping your current behavior: save ONLY the tensor
        torch.save(_to_cthw_uint8(frames), out_pt)
        written += 1

    print(f"[export] root_dir={root_dir}")
    print(f"[export] found={len(mp4s)} mp4s, written={written} pt files")


if __name__ == "__main__":
    # =========================================================================
    # Repo-root relationship ideology (as requested)
    #
    # Expectation:
    # - This file is inside repo somewhere like:
    #     thesis_main_files/<something>/<this_script>.py
    # - So parents[1] resolves to thesis_main_files/
    # =========================================================================

    parser = argparse.ArgumentParser(
        description="Export *.mp4 segments to sibling video_pt/*.pt next to each segment folder"
    )

    # This is the *relative* root under the base directory.
    # Example: "AVSpeech/AVSpeech_offline_training_files/avspeech_video_stage1/video_face_crops"
    # or "LAVDF/.../video_face_crops"
    parser.add_argument(
        "--file_root",
        type=str,
        required=True,
        help="Path RELATIVE to the chosen base directory (e.g. 'AVSpeech/.../video_face_crops').",
    )

    # Base directory under repo data/processed where file_root lives.
    # You can keep this default and override per dataset if needed.
    parser.add_argument(
        "--base_subdir",
        type=str,
        default="video_files",
        help="Subdirectory under data/processed that contains file_root (default: video_files).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files if they already exist.",
    )

    args = parser.parse_args()

    REPO_ROOT = Path(__file__).resolve().parents[2]

    # Build the target root directory under thesis_main_files/
    root_dir = (REPO_ROOT / "data" / "processed" / args.base_subdir / args.file_root).resolve()
    print(f"{root_dir} is the root dir")
    if not root_dir.exists():
        raise FileNotFoundError(
            f"[export] Resolved root_dir does not exist:\n"
            f"  root_dir={root_dir}\n"
            f"  REPO_ROOT={REPO_ROOT}\n"
            f"  base_subdir={args.base_subdir}\n"
            f"  file_root={args.file_root}\n"
        )

    print(f"[export] REPO_ROOT={REPO_ROOT}")
    print(f"[export] root_dir={root_dir}")
    export_mp4_segments_to_sibling_video_pt_dir(root_dir=root_dir, overwrite=args.overwrite)
