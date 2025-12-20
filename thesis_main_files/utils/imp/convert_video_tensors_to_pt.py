from __future__ import annotations

import argparse
import time
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
    debug_every: int = 50,          # [ADDED] heartbeat cadence
    preview_first: int = 3,         # [ADDED] print first K mp4s
    max_files: Optional[int] = None # [ADDED] cap for quick smoke test
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

    # [ADDED] timing + counters
    t0 = time.time()
    n_exists_skip = 0
    n_empty_skip = 0
    n_read_fail = 0
    written = 0
    processed = 0

    try:
        from torchvision.io import read_video
    except Exception as e:
        raise RuntimeError(
            "torchvision video IO is required. "
            "If unavailable, we can switch to OpenCV."
        ) from e

    if resize_hw is not None:
        import torch.nn.functional as F  # noqa: F401

    # [MODIFIED] iterator-based discovery (no upfront list(root_dir.rglob(...)))
    mp4_iter = root_dir.rglob("*.mp4")

    print("\n" + "=" * 80)
    print("[export][debug] starting export_mp4_segments_to_sibling_video_pt_dir")
    print(f"[export][debug] root_dir={root_dir}")
    print(f"[export][debug] overwrite={overwrite} resize_hw={resize_hw}")
    if max_files is not None:
        print(f"[export][debug] max_files={max_files}")
    print("=" * 80 + "\n")

    # [ADDED] preview printing state (now prints as discovered)
    preview_printed = 0

    # [MODIFIED] main loop now streams mp4 paths
    for idx, mp4_path in enumerate(mp4_iter, start=1):
        # [ADDED] stop early for smoke tests
        if max_files is not None and idx > max_files:
            print(f"[export][debug] reached max_files={max_files}, stopping early")
            break

        processed += 1

        # [MODIFIED] preview first few mp4 paths (streaming)
        if preview_first > 0 and preview_printed < preview_first:
            preview_printed += 1
            if preview_printed == 1:
                print(f"[export][debug] showing first {preview_first} discovered mp4 paths:")
            print(f"  [{preview_printed}] {mp4_path}")
            if preview_printed == preview_first:
                print("")

        out_dir = mp4_path.parent / "video_pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pt = out_dir / f"{mp4_path.stem}.pt"

        if out_pt.exists() and not overwrite:
            n_exists_skip += 1
            # [MODIFIED] no len(mp4s) available in iterator mode
            if debug_every > 0 and (idx % debug_every == 0):
                elapsed = time.time() - t0
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[export][debug] idx={idx} "
                    f"written={written} exists_skip={n_exists_skip} empty_skip={n_empty_skip} read_fail={n_read_fail} "
                    f"elapsed_sec={elapsed:.1f} rate={rate:.2f} files/s "
                    f"(latest skipped exists) out_pt={out_pt}"
                )
            continue

        t_read0 = time.time()
        try:
            frames, _, _ = read_video(str(mp4_path), pts_unit="sec")  # (T,H,W,C)
        except Exception as e:
            n_read_fail += 1
            print(f"[export][ERROR] read_video failed: mp4={mp4_path}")
            print(f"[export][ERROR] exception={type(e).__name__}: {e}")
            continue
        t_read1 = time.time()

        if frames.numel() == 0:
            n_empty_skip += 1
            if debug_every > 0 and (idx % debug_every == 0):
                print(f"[export][warn] empty frames: mp4={mp4_path}")
            continue

        frames = frames.to(torch.uint8)

        # NOTE: keeping your current behavior: save ONLY the tensor
        t_save0 = time.time()
        torch.save(_to_cthw_uint8(frames), out_pt)
        t_save1 = time.time()

        written += 1

        # [MODIFIED] heartbeat progress (no len(mp4s))
        if debug_every > 0 and (idx % debug_every == 0 or idx == 1):
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            try:
                T, H, W, C = frames.shape
            except Exception:
                T = H = W = C = -1

            print(
                f"[export][debug] idx={idx} "
                f"written={written} exists_skip={n_exists_skip} empty_skip={n_empty_skip} read_fail={n_read_fail} "
                f"elapsed_sec={elapsed:.1f} rate={rate:.2f} files/s "
                f"read_sec={(t_read1 - t_read0):.2f} save_sec={(t_save1 - t_save0):.2f} "
                f"frames_shape=(T={T},H={H},W={W},C={C}) "
                f"mp4={mp4_path.name}"
            )

    # [ADDED] final summary
    total = time.time() - t0
    print("\n" + "-" * 80)
    print(f"[export] root_dir={root_dir}")
    print(f"[export] processed={processed} mp4s")
    print(f"[export] written={written} pt files")
    print(f"[export] skipped_exists={n_exists_skip} skipped_empty={n_empty_skip} read_fail={n_read_fail}")
    print(f"[export] total_time_sec={total:.1f} avg_rate={(processed/total if total>0 else 0.0):.2f} files/s")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export *.mp4 segments to sibling video_pt/*.pt next to each segment folder"
    )

    parser.add_argument(
        "--file_root",
        type=str,
        required=True,
        help="Absolute OR repo-relative path to the folder to scan (we resolve it below).",
    )

    parser.add_argument(
        "--base_subdir",
        type=str,
        default="video_files",
        help="(kept for compatibility) Not used in the current root_dir resolution.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files if they already exist.",
    )

    parser.add_argument(
        "--debug_every",
        type=int,
        default=50,
        help="Print a heartbeat line every N mp4s (default: 50). Use 1 for very verbose.",
    )
    parser.add_argument(
        "--preview_first",
        type=int,
        default=3,
        help="Print the first K discovered mp4 paths (default: 3).",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Process only the first N mp4s (for quick smoke tests).",
    )

    args = parser.parse_args()

    REPO_ROOT = Path(__file__).resolve().parents[2]
    print(f"[export][debug] REPO_ROOT={REPO_ROOT}")

    file_root_path = Path(args.file_root)
    root_dir = file_root_path if file_root_path.is_absolute() else (REPO_ROOT / file_root_path)
    root_dir = root_dir.resolve()

    print(f"[export][debug] root_dir={root_dir}")

    if not root_dir.exists():
        raise FileNotFoundError(
            f"[export] Resolved root_dir does not exist:\n"
            f"  root_dir={root_dir}\n"
            f"  REPO_ROOT={REPO_ROOT}\n"
            f"  file_root={args.file_root}\n"
        )

    export_mp4_segments_to_sibling_video_pt_dir(
        root_dir=root_dir,
        overwrite=args.overwrite,
        debug_every=args.debug_every,
        preview_first=args.preview_first,
        max_files=args.max_files,
    )
