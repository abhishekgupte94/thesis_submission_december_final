from __future__ import annotations

import argparse
import os  # [ADDED] for env-based sharding defaults (torchrun)
import time
import zlib  # [ADDED] for deterministic path hashing
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
    debug_every: int = 50,
    preview_first: int = 3,
    max_files: Optional[int] = None,
    shard_rank: int = 0,            # [ADDED] DDP-style sharding rank
    shard_world_size: int = 1,      # [ADDED] DDP-style sharding world size
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

    # [ADDED] discovery vs owned counters (important with sharding + iterator)
    discovered = 0          # how many mp4s we have *seen* in traversal
    owned_seen = 0          # how many mp4s belong to this shard (incl. skips/errors)
    owned_previewed = 0     # how many "preview_first" printed

    try:
        from torchvision.io import read_video
    except Exception as e:
        raise RuntimeError(
            "torchvision video IO is required. "
            "If unavailable, we can switch to OpenCV."
        ) from e

    if resize_hw is not None:
        import torch.nn.functional as F  # noqa: F401

    # [MODIFIED] streaming iterator (no upfront list() scan)
    mp4_iter = root_dir.rglob("*.mp4")

    print("\n" + "=" * 80)
    print("[export][debug] starting export_mp4_segments_to_sibling_video_pt_dir")
    print(f"[export][debug] root_dir={root_dir}")
    print(f"[export][debug] overwrite={overwrite} resize_hw={resize_hw}")
    print(f"[export][debug] shard_rank={shard_rank} shard_world_size={shard_world_size}")
    if max_files is not None:
        print(f"[export][debug] max_files(per-rank)={max_files}")
    print("=" * 80 + "\n")

    for mp4_path in mp4_iter:
        discovered += 1

        # [ADDED] DDP-style deterministic sharding by path hash (order-independent)
        if shard_world_size > 1:
            key = zlib.crc32(str(mp4_path).encode("utf-8")) % shard_world_size
            if key != shard_rank:
                continue

        # From here onward, this mp4 belongs to THIS rank
        owned_seen += 1

        # [ADDED] cap is per-rank owned files (prevents early stop after skips)
        if max_files is not None and owned_seen > max_files:
            print(f"[export][debug] reached max_files(per-rank)={max_files}, stopping early")
            break

        # [MODIFIED] preview first few OWNED mp4 paths (streaming)
        if preview_first > 0 and owned_previewed < preview_first:
            owned_previewed += 1
            if owned_previewed == 1:
                print(f"[export][debug] showing first {preview_first} OWNED mp4 paths for this shard:")
            print(f"  [{owned_previewed}] {mp4_path}")
            if owned_previewed == preview_first:
                print("")

        out_dir = mp4_path.parent / "video_pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pt = out_dir / f"{mp4_path.stem}.pt"

        if out_pt.exists() and not overwrite:
            n_exists_skip += 1
            if debug_every > 0 and (owned_seen % debug_every == 0):
                elapsed = time.time() - t0
                rate = owned_seen / elapsed if elapsed > 0 else 0.0
                print(
                    f"[export][debug] owned_idx={owned_seen} discovered={discovered} "
                    f"written={written} exists_skip={n_exists_skip} empty_skip={n_empty_skip} read_fail={n_read_fail} "
                    f"elapsed_sec={elapsed:.1f} rate={rate:.2f} owned_files/s "
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
            if debug_every > 0 and (owned_seen % debug_every == 0):
                print(f"[export][warn] empty frames: mp4={mp4_path}")
            continue

        frames = frames.to(torch.uint8)

        # NOTE: keeping your current behavior: save ONLY the tensor
        t_save0 = time.time()
        torch.save(_to_cthw_uint8(frames), out_pt)
        t_save1 = time.time()

        written += 1

        if debug_every > 0 and (owned_seen % debug_every == 0 or owned_seen == 1):
            elapsed = time.time() - t0
            rate = owned_seen / elapsed if elapsed > 0 else 0.0
            try:
                T, H, W, C = frames.shape
            except Exception:
                T = H = W = C = -1

            print(
                f"[export][debug] owned_idx={owned_seen} discovered={discovered} "
                f"written={written} exists_skip={n_exists_skip} empty_skip={n_empty_skip} read_fail={n_read_fail} "
                f"elapsed_sec={elapsed:.1f} rate={rate:.2f} owned_files/s "
                f"read_sec={(t_read1 - t_read0):.2f} save_sec={(t_save1 - t_save0):.2f} "
                f"frames_shape=(T={T},H={H},W={W},C={C}) "
                f"mp4={mp4_path.name}"
            )

    total = time.time() - t0
    print("\n" + "-" * 80)
    print(f"[export] root_dir={root_dir}")
    print(f"[export] shard_rank={shard_rank} shard_world_size={shard_world_size}")
    print(f"[export] discovered_total={discovered} (walked)")
    print(f"[export] owned_seen={owned_seen} (this shard)")
    print(f"[export] written={written} pt files (this shard)")
    print(f"[export] skipped_exists={n_exists_skip} skipped_empty={n_empty_skip} read_fail={n_read_fail}")
    print(f"[export] total_time_sec={total:.1f} avg_rate={(owned_seen/total if total>0 else 0.0):.2f} owned_files/s")
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
        help="Print a heartbeat line every N OWNED mp4s per rank (default: 50). Use 1 for very verbose.",
    )
    parser.add_argument(
        "--preview_first",
        type=int,
        default=3,
        help="Print the first K OWNED mp4 paths for this shard (default: 3).",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Process only the first N OWNED mp4s per rank (for quick smoke tests).",
    )

    # [ADDED] Sharding args (torchrun-friendly)
    parser.add_argument(
        "--shard_rank",
        type=int,
        default=None,
        help="Shard rank (defaults to env RANK/LOCAL_RANK).",
    )
    parser.add_argument(
        "--shard_world_size",
        type=int,
        default=None,
        help="Shard world size (defaults to env WORLD_SIZE).",
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

    # [ADDED] resolve shard params from torchrun env if not provided
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    shard_world_size = env_world if args.shard_world_size is None else int(args.shard_world_size)
    shard_rank = env_rank if args.shard_rank is None else int(args.shard_rank)

    export_mp4_segments_to_sibling_video_pt_dir(
        root_dir=root_dir,
        overwrite=args.overwrite,
        debug_every=args.debug_every,
        preview_first=args.preview_first,
        max_files=args.max_files,
        shard_rank=shard_rank,               # [ADDED]
        shard_world_size=shard_world_size,   # [ADDED]
    )
