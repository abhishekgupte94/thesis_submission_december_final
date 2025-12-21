from __future__ import annotations

import argparse
import os
import time
import zlib
from pathlib import Path
from typing import Optional

import torch


def _to_cthw_uint8(frames_thwc_uint8: torch.Tensor) -> torch.Tensor:
    """
    (T,H,W,C) uint8 -> (C,T,H,W) uint8
    """
    return frames_thwc_uint8.to(torch.uint8).permute(3, 0, 1, 2).contiguous()


def _owns_path(path: str | Path, *, shard_rank: int, shard_world_size: int) -> bool:
    """
    Deterministic sharding by CRC32(path) % world_size.
    Order-independent (important on NFS).
    """
    if shard_world_size <= 1:
        return True
    key = zlib.crc32(str(path).encode("utf-8")) % shard_world_size
    return key == shard_rank


def export_mp4_segments_to_sibling_video_pt_dir(
    root_dir: str | Path,
    *,
    overwrite: bool = False,
    debug_every: int = 200,
    max_files: Optional[int] = None,   # per-rank owned cap
    shard_rank: int = 0,
    shard_world_size: int = 1,
) -> None:
    root_dir = Path(root_dir)

    t0 = time.time()
    discovered = 0          # total mp4s encountered during walk
    owned_seen = 0          # mp4s owned by this rank
    written = 0
    exists_skip = 0
    empty_skip = 0
    read_fail = 0

    try:
        from torchvision.io import read_video
    except Exception as e:
        raise RuntimeError(
            "torchvision video IO is required (PyAV/ffmpeg). "
            "If unavailable, switch to OpenCV."
        ) from e

    print(f"[export][r{shard_rank}] root_dir={root_dir}")
    print(f"[export][r{shard_rank}] shard_rank={shard_rank} shard_world_size={shard_world_size}")
    if max_files is not None:
        print(f"[export][r{shard_rank}] max_files(per-rank)={max_files}")

    # FAST traversal (no rglob)
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".mp4"):
                continue

            mp4_path = Path(dirpath) / fname
            discovered += 1

            # [SHARD] only handle files owned by this rank
            if not _owns_path(mp4_path, shard_rank=shard_rank, shard_world_size=shard_world_size):
                continue

            owned_seen += 1
            if max_files is not None and owned_seen > max_files:
                print(f"[export][r{shard_rank}] reached max_files(per-rank)={max_files}, stopping")
                total = time.time() - t0
                print(f"[export][r{shard_rank}] discovered={discovered} owned_seen={owned_seen} written={written} total_sec={total:.1f}")
                return

            out_dir = mp4_path.parent / "video_pt"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_pt = out_dir / f"{mp4_path.stem}.pt"

            if out_pt.exists() and not overwrite:
                exists_skip += 1
                if debug_every > 0 and (owned_seen % debug_every == 0):
                    elapsed = time.time() - t0
                    print(
                        f"[export][r{shard_rank}] owned={owned_seen} discovered={discovered} "
                        f"written={written} exists_skip={exists_skip} empty_skip={empty_skip} read_fail={read_fail} "
                        f"elapsed_sec={elapsed:.1f}"
                    )
                continue

            try:
                frames, _, _ = read_video(str(mp4_path), pts_unit="sec")  # (T,H,W,C)
            except Exception as e:
                read_fail += 1
                print(f"[export][r{shard_rank}][ERROR] read_video failed: {mp4_path} :: {type(e).__name__}: {e}")
                continue

            if frames.numel() == 0:
                empty_skip += 1
                continue

            frames = frames.to(torch.uint8)
            video_u8_cthw = _to_cthw_uint8(frames)  # (3,T,H,W) uint8

            torch.save(video_u8_cthw, out_pt)
            written += 1

            if debug_every > 0 and (owned_seen % debug_every == 0 or owned_seen == 1):
                elapsed = time.time() - t0
                print(
                    f"[export][r{shard_rank}] owned={owned_seen} discovered={discovered} "
                    f"written={written} exists_skip={exists_skip} empty_skip={empty_skip} read_fail={read_fail} "
                    f"elapsed_sec={elapsed:.1f} last={mp4_path.name}"
                )

    total = time.time() - t0
    print(
        f"[export][r{shard_rank}][DONE] discovered={discovered} owned_seen={owned_seen} written={written} "
        f"exists_skip={exists_skip} empty_skip={empty_skip} read_fail={read_fail} total_sec={total:.1f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_root", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug_every", type=int, default=200)
    parser.add_argument("--max_files", type=int, default=None)

    # sharding args (optional; torchrun env vars will auto-fill)
    parser.add_argument("--shard_rank", type=int, default=None)
    parser.add_argument("--shard_world_size", type=int, default=None)

    args = parser.parse_args()

    root_dir = Path(args.file_root).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")

    # Resolve shard params from torchrun env if not provided
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    shard_world_size = env_world if args.shard_world_size is None else int(args.shard_world_size)
    shard_rank = env_rank if args.shard_rank is None else int(args.shard_rank)

    export_mp4_segments_to_sibling_video_pt_dir(
        root_dir=root_dir,
        overwrite=args.overwrite,
        debug_every=args.debug_every,
        max_files=args.max_files,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )
