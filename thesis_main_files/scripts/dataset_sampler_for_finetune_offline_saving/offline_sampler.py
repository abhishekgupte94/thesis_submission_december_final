# core/utils/csv_random_sampler.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class CSVSampleSpec:
    """
    Controls how we sample a CSV before feeding rows into your offline audio/video pipelines.

    Typical usage:
      spec = CSVSampleSpec(
          n_total=5000,
          seed=42,
          stratify_col="label",
          per_class_cap={0: 2500, 1: 2500},
          require_unique_col="clip_id",
      )
    """
    n_total: Optional[int] = None               # None => no cap (use all after filtering)
    seed: int = 42
    shuffle: bool = True

    # Optional filtering
    split_col: Optional[str] = None             # e.g. "split"
    allowed_splits: Optional[Sequence[str]] = None  # e.g. ["train"]

    # Optional stratified sampling
    stratify_col: Optional[str] = None          # e.g. "label"
    per_class_cap: Optional[Dict[Any, int]] = None  # e.g. {0: 2500, 1: 2500}

    # Optional uniqueness constraint (common if you have many segments per clip)
    require_unique_col: Optional[str] = None    # e.g. "clip_id" or "file"

    # Optional column selection (keeps CSV lighter when downstream only needs a few cols)
    keep_cols: Optional[Sequence[str]] = None   # e.g. ["clip_id","label","file"]

    # Optional: drop rows where these columns are null/empty
    required_nonempty_cols: Optional[Sequence[str]] = None


# =============================================================================
# Core Sampler
# =============================================================================

def load_csv_random_sample(
    csv_path: Union[str, Path],
    spec: CSVSampleSpec,
    *,
    return_format: str = "dataframe",  # "dataframe" | "records"
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Load CSV -> filter -> (optional) unique-by -> (optional) stratified cap -> (optional) global cap -> shuffle.

    This function does NOT assume any schema beyond the columns you reference in `spec`.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ---------------------------
    # Column subset
    # ---------------------------
    if spec.keep_cols is not None:
        missing = [c for c in spec.keep_cols if c not in df.columns]
        if missing:
            raise KeyError(f"CSV missing keep_cols={missing}. Available={list(df.columns)}")
        df = df.loc[:, list(spec.keep_cols)].copy()

    # ---------------------------
    # Required nonempty columns
    # ---------------------------
    if spec.required_nonempty_cols:
        missing = [c for c in spec.required_nonempty_cols if c not in df.columns]
        if missing:
            raise KeyError(f"CSV missing required_nonempty_cols={missing}. Available={list(df.columns)}")
        for c in spec.required_nonempty_cols:
            df = df[df[c].notna()]
            # also drop empty strings if dtype is object
            if df[c].dtype == "object":
                df = df[df[c].astype(str).str.strip() != ""]

    # ---------------------------
    # Split filtering
    # ---------------------------
    if spec.split_col and spec.allowed_splits is not None:
        if spec.split_col not in df.columns:
            raise KeyError(f"CSV missing split_col='{spec.split_col}'. Available={list(df.columns)}")
        df = df[df[spec.split_col].isin(list(spec.allowed_splits))]

    # ---------------------------
    # Unique constraint
    # ---------------------------
    if spec.require_unique_col:
        if spec.require_unique_col not in df.columns:
            raise KeyError(
                f"CSV missing require_unique_col='{spec.require_unique_col}'. Available={list(df.columns)}"
            )
        # Keep first occurrence; deterministic if you shuffle later with seed.
        df = df.drop_duplicates(subset=[spec.require_unique_col], keep="first")

    # Empty after filtering?
    if len(df) == 0:
        raise ValueError("No rows remain after filtering/constraints. Check your spec and CSV.")

    # ---------------------------
    # Stratified cap (optional)
    # ---------------------------
    if spec.stratify_col and spec.per_class_cap:
        if spec.stratify_col not in df.columns:
            raise KeyError(f"CSV missing stratify_col='{spec.stratify_col}'. Available={list(df.columns)}")

        # sample per class deterministically
        chunks = []
        rng_seed = int(spec.seed)
        for cls_val, cap in spec.per_class_cap.items():
            sub = df[df[spec.stratify_col] == cls_val]
            if len(sub) == 0:
                continue
            if cap is None or cap <= 0:
                continue
            # if cap > len(sub), take all
            take_n = min(int(cap), len(sub))
            if spec.shuffle:
                sub = sub.sample(n=take_n, random_state=rng_seed, replace=False)
            else:
                sub = sub.head(take_n)
            chunks.append(sub)
            rng_seed += 1  # tweak so each class uses different deterministic stream

        if not chunks:
            raise ValueError("Stratified sampling produced zero rows (check per_class_cap vs CSV labels).")

        df = pd.concat(chunks, axis=0, ignore_index=True)

    # ---------------------------
    # Global cap (optional)
    # ---------------------------
    if spec.n_total is not None:
        n = int(spec.n_total)
        if n < 0:
            raise ValueError("n_total must be >= 0 or None.")
        if n == 0:
            df = df.iloc[0:0].copy()
        elif n < len(df):
            if spec.shuffle:
                df = df.sample(n=n, random_state=int(spec.seed), replace=False)
            else:
                df = df.head(n)

    # ---------------------------
    # Final shuffle (optional)
    # ---------------------------
    if spec.shuffle and len(df) > 1:
        df = df.sample(frac=1.0, random_state=int(spec.seed)).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if return_format == "dataframe":
        return df
    elif return_format == "records":
        return df.to_dict(orient="records")
    else:
        raise ValueError("return_format must be 'dataframe' or 'records'")


# =============================================================================
# Small helper to get just ids/labels (handy if your offline systems accept lists)
# =============================================================================

def extract_cols_as_lists(
    df: pd.DataFrame,
    *,
    id_col: str,
    label_col: Optional[str] = None,
) -> Tuple[List[Any], Optional[List[Any]]]:
    if id_col not in df.columns:
        raise KeyError(f"Missing id_col='{id_col}' in df columns: {list(df.columns)}")
    ids = df[id_col].tolist()

    labels = None
    if label_col is not None:
        if label_col not in df.columns:
            raise KeyError(f"Missing label_col='{label_col}' in df columns: {list(df.columns)}")
        labels = df[label_col].tolist()

    return ids, labels



# =============================================================================
# [ADDED] Filename-only sampler (audio + video share filenames)
# =============================================================================

from typing import List, Tuple, Union
from pathlib import Path
import pandas as pd

def sample_filenames_from_csv(
    csv_path: Union[str, Path],
    *,
    filename_col: str,
    spec: CSVSampleSpec,
) -> List[str]:
    """
    Read CSV → sample rows → return filenames.

    Assumption:
      - audio and video filenames are IDENTICAL
      - downstream systems already know their own root dirs

    Returns:
      List[str]: filenames only (e.g. ["000123.mp4", "000456.mp4"])
    """
    df = load_csv_random_sample(
        csv_path=csv_path,
        spec=spec,
        return_format="dataframe",
    )

    if filename_col not in df.columns:
        raise KeyError(
            f"CSV missing filename_col='{filename_col}'. "
            f"Available columns={list(df.columns)}"
        )

    filenames = df[filename_col].astype(str).tolist()
    return filenames
