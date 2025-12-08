#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Check if tokens vary across video (no hardcoding).")
    ap.add_argument("--input_npz", required=True, help="Path to .npz with per-video arrays of shape (L, D)")
    ap.add_argument("--drop_cls", action="store_true",
                    help="If set, drop the first token (often CLS) before analysis")
    ap.add_argument("--eps", type=float, default=1e-8,
                    help="Threshold to count a token as 'varying' (variance > eps)")
    args = ap.parse_args()

    npz_path = Path(args.input_npz).expanduser()
    data = np.load(npz_path, allow_pickle=False)
    keys = list(data.files)

    # Load arrays and sanity check ranks
    arrays = []
    shapes = {}
    for k in keys:
        arr = np.asarray(data[k])
        if arr.ndim != 2:
            raise ValueError(f"Entry '{k}' must be 2D (L,D). Got shape {arr.shape}.")
        arrays.append(arr)
        shapes[k] = arr.shape

    # Align to the minimum shape over all video (no hardcoding)
    min_L = min(a.shape[0] for a in arrays)
    min_D = min(a.shape[1] for a in arrays)
    arrays = [a[:min_L, :min_D] for a in arrays]          # (L, D)
    X = np.stack(arrays, axis=0).astype(np.float32)       # (N, L, D)
    N, L, D = X.shape

    # Optionally drop the first token (often CLS)
    start_idx = 1 if args.drop_cls and L > 1 else 0
    X_tokens = X[:, start_idx:, :]                        # (N, L_used, D)
    L_used = X_tokens.shape[1]

    # Two very simple “does it vary?” measures per token:
    # 1) Mean feature variance across video: var_over_videos(N) → (L,D) → mean over D => (L,)
    var_over_videos = np.var(X_tokens, axis=0)            # (L_used, D)
    token_var_mean = var_over_videos.mean(axis=1)         # (L_used,)

    # 2) Norm-based token energy per video then variance across video
    token_energy = np.linalg.norm(X_tokens, axis=2)       # (N, L_used)
    token_energy_var = np.var(token_energy, axis=0)       # (L_used,)

    # Summaries
    n_vary_mean = int((token_var_mean > args.eps).sum())
    n_vary_energy = int((token_energy_var > args.eps).sum())

    print(f"[INFO] Videos ({N}): {keys}")
    print(f"[INFO] Original shapes: {shapes}")
    print(f"[INFO] Aligned to (L={L}, D={D}); analyzed L_used={L_used} "
          f"({'dropped first token' if start_idx==1 else 'kept all tokens'})")

    print(f"[RESULT] Tokens with variance > {args.eps:g} (mean-over-features): {n_vary_mean}/{L_used}")
    print(f"[RESULT] Tokens with variance > {args.eps:g} (energy-based)     : {n_vary_energy}/{L_used}")

    # Show the top-5 most varying tokens by each metric (index is relative to analyzed range)
    top5_mean_idx = np.argsort(token_var_mean)[-5:][::-1]
    top5_energy_idx = np.argsort(token_energy_var)[-5:][::-1]

    def fmt_top(indices, values):
        return ", ".join([f"{int(i)}:{float(values[i]):.6g}" for i in indices])

    print(f"[TOP5] mean-over-features variance (token_index:variance): {fmt_top(top5_mean_idx, token_var_mean)}")
    print(f"[TOP5] energy-based variance     (token_index:variance): {fmt_top(top5_energy_idx, token_energy_var)}")

if __name__ == "__main__":
    main()
