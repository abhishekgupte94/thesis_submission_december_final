from __future__ import annotations

from pathlib import Path

from scripts.dataset_sampler_for_finetune_offline_saving.csv_strategic_sampler import (
    StrategicSampleSpec,
    strategic_sample_and_move,
)

# =============================================================================
# Paths (FINAL)
# =============================================================================

CSV_IN_PATH = Path(
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/data/raw/csv/lav_df/metadata/metadata.csv"
)

CSV_OUT_PATH = Path(
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/data/raw/csv/lav_df/metadata/processed/metadata_train.csv"
)

SRC_DIR = Path(
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/data/processed/video_files/LAV_DF/video"
)

DST_DIR = Path(
    "/Users/abhishekgupte_macbookpro/PycharmProjects/project_combined_repo_clean/"
    "thesis_main_files/data/processed/splits/train/video"
)


def main() -> None:
    spec = StrategicSampleSpec(
        n_total=15000,
        seed=42,
        label_ratio_fake=0.80,
        short_ratio=0.80,
        short_max_seconds=7.5,
        move_files=True,
        overwrite_existing_in_dst=False,
        dry_run=False,   # set True to test without moving
    )

    sampled_df, stats = strategic_sample_and_move(
        csv_in_path=CSV_IN_PATH,
        csv_out_path=CSV_OUT_PATH,
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        duration_col="duration",
        spec=spec,
        filename_col="filename",
        label_col="label",
    )

    print("\n========== SAMPLING COMPLETE ==========")
    for k, v in stats.items():
        print(f"{k}: {v}")

    short = (sampled_df["duration"] <= 7.5).sum()
    total = len(sampled_df)
    print(f"Duration <= 7.5s: {short}/{total} ({short / max(total,1):.3f})")


if __name__ == "__main__":
    main()
