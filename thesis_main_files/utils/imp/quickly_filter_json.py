#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Set


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict JSON, got: {type(data)}")
    return data


def build_available_keys(video_root: Path, json_keys: Set[str]) -> tuple[Set[str], Counter]:
    """
    Builds a set of keys that are "available on disk", matching the JSON key format,
    and returns (available_keys, match_breakdown).

    Supports:
      A) Flat:   .../000001.mp4
      B) Nested: .../000001/seg_0000/seg_0000.mp4  -> treat '000001.mp4' as available
    """
    avail: Set[str] = set()
    breakdown = Counter()

    # Scan mp4 paths (works for both layouts; costs a bit but you're only at ~15k mp4s)
    for p in video_root.rglob("*.mp4"):
        parts = p.parts

        # Rule A: direct filename matches JSON key (flat layout)
        name = p.name  # e.g. "000001.mp4" or "seg_0000.mp4"
        if name in json_keys:
            avail.add(name)
            breakdown["A: mp4 filename == json key"] += 1

        # Rule B: clip_id directory matches JSON key (nested segments)
        # expected: .../<clip_id>/seg_0000/seg_0000.mp4 => clip_id = parts[-3]
        if len(parts) >= 3:
            clip_id = parts[-3]              # e.g. "000001"
            candidate = f"{clip_id}.mp4"      # e.g. "000001.mp4"
            if candidate in json_keys:
                avail.add(candidate)
                breakdown["B: parent clip_id dir -> <clip_id>.mp4"] += 1

        # Optional extra: sometimes structure may be .../<clip_id>/something/seg_x.mp4
        if len(parts) >= 4:
            clip_id2 = parts[-4]
            candidate2 = f"{clip_id2}.mp4"
            if candidate2 in json_keys:
                avail.add(candidate2)
                breakdown["C: grandparent dir -> <clip_id>.mp4"] += 1

    return avail, breakdown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True, type=Path)
    ap.add_argument("--json-in", required=True, type=Path)
    ap.add_argument("--json-out", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    video_root: Path = args.video_root
    json_in: Path = args.json_in
    json_out: Path = args.json_out

    if not video_root.exists():
        raise FileNotFoundError(f"video-root not found: {video_root}")
    if not json_in.exists():
        raise FileNotFoundError(f"json-in not found: {json_in}")

    data = load_json(json_in)
    json_keys = set(data.keys())

    print(f"[filter] video_root        : {video_root}")
    print(f"[filter] json_in_entries   : {len(data)}")

    avail, breakdown = build_available_keys(video_root, json_keys)
    print(f"[filter] available_keys    : {len(avail)}")
    print(f"[filter] match_breakdown   : {dict(breakdown)}")

    filtered = {k: v for k, v in data.items() if k in avail}
    print(f"[filter] json_out_entries  : {len(filtered)}")
    print(f"[filter] dropped           : {len(data) - len(filtered)}")

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(filtered, f)
    print(f"[filter] wrote             : {json_out} ({json_out.stat().st_size} bytes)")

    # samples
    kept_keys = list(filtered.keys())[: args.samples]
    missing_keys = []
    for k in data.keys():
        if k not in filtered:
            missing_keys.append(k)
            if len(missing_keys) >= args.samples:
                break

    if args.report:
        lines = []
        lines.append("=== filter report ===")
        lines.append(f"video_root       : {video_root}")
        lines.append(f"json_in          : {json_in}")
        lines.append(f"json_out         : {json_out}")
        lines.append(f"json_in_entries  : {len(data)}")
        lines.append(f"available_keys   : {len(avail)}")
        lines.append(f"json_out_entries : {len(filtered)}")
        lines.append(f"dropped          : {len(data) - len(filtered)}")
        lines.append(f"match_breakdown  : {dict(breakdown)}")
        lines.append("")
        lines.append("sample_kept_keys:")
        for k in kept_keys:
            lines.append(f"  {k}")
        lines.append("")
        lines.append("sample_missing_keys:")
        for k in missing_keys:
            lines.append(f"  {k}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines), encoding="utf-8")
        print(f"[filter] report            : {args.report} ({args.report.stat().st_size} bytes)")

    if len(filtered) == 0:
        print(
            "\n[WARN] Output is empty. This means none of the JSON keys matched your disk layout.\n"
            "To diagnose, run:\n"
            "  find <video-root> -type f -name '*.mp4' | head\n"
            "and compare those paths to JSON keys like '000001.mp4'.\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
