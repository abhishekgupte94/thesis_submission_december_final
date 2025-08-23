#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    keys = list(data.files)
    arrays = {k: np.asarray(data[k]) for k in keys}
    return keys, arrays


def align_arrays(arrays: dict[str, np.ndarray]):
    """
    Align all arrays to a common (L, D) by cropping to the minimum
    length/dim found across all arrays. No hardcoding.
    """
    shapes = {k: v.shape for k, v in arrays.items()}
    # Expect rank-2 arrays (L, D). If not, raise a helpful error.
    for k, shp in shapes.items():
        if len(shp) != 2:
            raise ValueError(f"Entry '{k}' has rank {len(shp)} (shape {shp}); "
                             f"expected 2D (L, D).")

    min_L = min(shp[0] for shp in shapes.values())
    min_D = min(shp[1] for shp in shapes.values())
    aligned = {k: v[:min_L, :min_D] for k, v in arrays.items()}
    return aligned, min_L, min_D, shapes


def compute_variation(X: np.ndarray, video_keys: list[str]):
    """
    X: (N, L, D) stacked array
    Returns a dict of useful metrics without hardcoding.
    """
    N, L, D = X.shape

    # Variance across videos for each (t, d), then summarize:
    # temporal_var_mean[t] = mean over features of var_over_videos(X[:, t, d])
    var_over_videos = np.var(X, axis=0)        # (L, D)
    temporal_var_mean = var_over_videos.mean(axis=1)  # (L,)
    feature_var_mean  = var_over_videos.mean(axis=0)  # (D,)

    # Alternate “energy variance” views (can be insightful):
    # Token-wise L2 energy per video → variance across videos
    token_energy = np.linalg.norm(X, axis=2)   # (N, L)
    temporal_var_energy = np.var(token_energy, axis=0)  # (L,)

    # Feature-wise L2 energy over time per video → variance across videos
    feature_energy = np.linalg.norm(X, axis=1)  # (N, D)
    feature_var_energy = np.var(feature_energy, axis=0)  # (D,)

    # Per-video pooled vectors (mean over tokens)
    per_video_pooled = X.mean(axis=1)          # (N, D)

    # Pairwise cosine similarity between videos
    norms = np.linalg.norm(per_video_pooled, axis=1, keepdims=True) + 1e-12
    pv_normed = per_video_pooled / norms
    cosine_sim = pv_normed @ pv_normed.T       # (N, N)

    return {
        "temporal_var_mean": temporal_var_mean,       # (L,)
        "temporal_var_energy": temporal_var_energy,   # (L,)
        "feature_var_mean": feature_var_mean,         # (D,)
        "feature_var_energy": feature_var_energy,     # (D,)
        "pairwise_cosine": cosine_sim,                # (N, N)
        "per_video_pooled": per_video_pooled,         # (N, D)
        "video_keys": video_keys                      # list of names in order
    }


def save_csv(arr: np.ndarray, path: Path, header: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, delimiter=",", header=header or "", comments="")


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def maybe_plot_temporal(temporal_var_mean: np.ndarray, temporal_var_energy: np.ndarray, out_png: Path):
    if not MATPLOTLIB_OK:
        return
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(temporal_var_mean, label="temporal_var_mean (avg over features)")
    plt.plot(temporal_var_energy, label="temporal_var_energy (norm-based)")
    plt.xlabel("Token index (after model downsampling)")
    plt.ylabel("Variance across videos")
    plt.title("Temporal variation across videos")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def maybe_plot_feature(feature_var_mean: np.ndarray, feature_var_energy: np.ndarray, out_png: Path):
    if not MATPLOTLIB_OK:
        return
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(feature_var_mean, label="feature_var_mean (avg over time)")
    plt.plot(feature_var_energy, label="feature_var_energy (time-norm based)")
    plt.xlabel("Feature dimension index")
    plt.ylabel("Variance across videos")
    plt.title("Feature-dimension variation across videos")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def maybe_plot_cosine_matrix(cosine: np.ndarray, labels: list[str], out_png: Path):
    if not MATPLOTLIB_OK:
        return
    import matplotlib.pyplot as plt
    plt.figure()
    im = plt.imshow(cosine, vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Pairwise cosine similarity (pooled vectors)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze temporal and feature variation across videos in a .npz")
    parser.add_argument("--input_npz", required=True, help="Path to .npz containing per-video arrays (L, D)")
    parser.add_argument("--outdir", required=True, help="Where to save analysis outputs (CSVs, PNGs, JSON)")
    parser.add_argument("--plots", action="store_true", help="Also save PNG plots (requires matplotlib)")
    args = parser.parse_args()

    npz_path = Path(args.input_npz).expanduser()
    outdir   = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {npz_path}")
    video_keys, arrays = load_npz(npz_path)

    print(f"[INFO] Found {len(video_keys)} videos: {video_keys}")
    aligned, L, D, original_shapes = align_arrays(arrays)
    print(f"[INFO] Original shapes: {original_shapes}")
    print(f"[INFO] Aligned to common shape: (L={L}, D={D}) (cropped to min where needed)")

    # Stack into (N, L, D) in the order of video_keys
    X = np.stack([aligned[k] for k in video_keys], axis=0).astype(np.float32)  # (N, L, D)

    metrics = compute_variation(X, video_keys)

    # Save CSVs
    save_csv(metrics["temporal_var_mean"], outdir / "temporal_var_mean.csv",
             header="temporal_index, variance_mean_over_features")
    save_csv(metrics["temporal_var_energy"], outdir / "temporal_var_energy.csv",
             header="temporal_index, variance_over_videos_of_token_energy")
    save_csv(metrics["feature_var_mean"], outdir / "feature_var_mean.csv",
             header="feature_dim_index, variance_mean_over_time")
    save_csv(metrics["feature_var_energy"], outdir / "feature_var_energy.csv",
             header="feature_dim_index, variance_over_videos_of_time_energy")

    # Save pairwise cosine similarities + labels
    save_csv(metrics["pairwise_cosine"], outdir / "pairwise_cosine.csv")
    save_json({"video_keys": metrics["video_keys"]}, outdir / "labels.json")

    # Save a compact summary JSON with a few stats
    summary = {
        "N_videos": X.shape[0],
        "L_tokens": int(L),
        "D_features": int(D),
        "temporal_var_mean": {
            "mean": float(np.mean(metrics["temporal_var_mean"])),
            "std": float(np.std(metrics["temporal_var_mean"])),
            "max": float(np.max(metrics["temporal_var_mean"])),
            "argmax": int(np.argmax(metrics["temporal_var_mean"]))
        },
        "feature_var_mean": {
            "mean": float(np.mean(metrics["feature_var_mean"])),
            "std": float(np.std(metrics["feature_var_mean"])),
            "max": float(np.max(metrics["feature_var_mean"])),
            "argmax": int(np.argmax(metrics["feature_var_mean"]))
        }
    }
    save_json(summary, outdir / "summary.json")

    # Optional plots
    if args.plots:
        if not MATPLOTLIB_OK:
            print("[WARN] matplotlib not available; skipping plots.")
        else:
            maybe_plot_temporal(metrics["temporal_var_mean"],
                                metrics["temporal_var_energy"],
                                outdir / "temporal_variation.png")
            maybe_plot_feature(metrics["feature_var_mean"],
                               metrics["feature_var_energy"],
                               outdir / "feature_variation.png")
            maybe_plot_cosine_matrix(metrics["pairwise_cosine"],
                                     metrics["video_keys"],
                                     outdir / "pairwise_cosine.png")

    print(f"[OK] Saved analysis to: {outdir}")


if __name__ == "__main__":
    main()
