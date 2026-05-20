"""Stratified 80/10/10 train/val/test splitter."""

from __future__ import annotations

import json
import logging
import warnings

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from backend.shared.path_resolver import get_dataset_path

# Suppress harmless MLflow schema inference warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

log = logging.getLogger(__name__)


class DataSplitter:
    def run(self, dataset_name: str) -> dict:
        """Split generated_raw.csv → train/val/test CSVs.

        Uses StratifiedShuffleSplit which shuffles and stratifies by
        label × scenario. No data leakage — splits are from mutually
        exclusive index sets.

        Split: 80% train / 10% val / 10% test
        """
        dataset_dir = get_dataset_path(dataset_name)
        input_path = dataset_dir / "raw" / "generated_raw.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"generated_raw.csv not found at {input_path}. Run phase_2_generate.py first."
            )

        df = pd.read_csv(input_path)
        log.info(f"Splitting {len(df):,} rows for '{dataset_name}'")

        # Stratification key: label + scenario ensures both are balanced across splits
        df["_strat"] = df["label"].astype(str) + "__" + df["scenario"].astype(str)

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, valtest_idx = next(sss1.split(df, df["_strat"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valtest_df = df.iloc[valtest_idx].reset_index(drop=True)

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_idx, test_idx = next(sss2.split(valtest_df, valtest_df["_strat"]))
        val_df = valtest_df.iloc[val_idx].reset_index(drop=True)
        test_df = valtest_df.iloc[test_idx].reset_index(drop=True)

        for frame in (train_df, val_df, test_df):
            frame.drop(columns=["_strat"], inplace=True)

        train_df.to_csv(dataset_dir / "train.csv", index=False)
        val_df.to_csv(dataset_dir / "val.csv", index=False)
        test_df.to_csv(dataset_dir / "test.csv", index=False)

        log.info(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

        log_to_mlflow(dataset_name, train_df, val_df, test_df)

        stats = build_stats(train_df, val_df, test_df, dataset_name)
        (dataset_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
        print_stats(stats)
        return stats

    @staticmethod
    def _log_to_mlflow(
        dataset_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Private — called internally. Kept with underscore for legacy compat."""
        log_to_mlflow(dataset_name, train_df, val_df, test_df)


def log_to_mlflow(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Log split statistics to MLflow."""
    try:
        mlflow.set_experiment("phase1_data")
        with mlflow.start_run(run_name=f"split_{dataset_name}"):
            mlflow.log_params(
                {
                    "dataset": dataset_name,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "label1_ratio": round(float(train_df["label"].mean()), 3),
                }
            )
            log_mlflow_datasets(dataset_name, train_df, val_df, test_df)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"MLflow logging skipped: {exc}")


def log_mlflow_datasets(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Register DataFrames as MLflow dataset inputs (best-effort)."""
    from_pandas = getattr(getattr(mlflow, "data", None), "from_pandas", None)
    if from_pandas is None:
        return
    for frame, ctx in [(train_df, "training"), (val_df, "validation"), (test_df, "test")]:
        ds = from_pandas(frame, name=f"{dataset_name}-{ctx}", targets="label")
        mlflow.log_input(ds, context=ctx)


def build_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
) -> dict:
    """Build a split statistics summary dict."""

    def info(df: pd.DataFrame) -> dict:
        return {
            "rows": len(df),
            "label_0": int((df["label"] == 0).sum()),
            "label_1": int((df["label"] == 1).sum()),
            "label_1_ratio": round(float(df["label"].mean()), 3),
            "scenarios": df["scenario"].value_counts().to_dict(),
            "languages": df["language"].value_counts().to_dict(),
        }

    return {"dataset_name": name, "train": info(train), "val": info(val), "test": info(test)}


def print_stats(stats: dict) -> None:
    print("\n" + "─" * 60)
    print(f"Dataset: {stats['dataset_name']}")
    for split in ("train", "val", "test"):
        s = stats[split]
        print(
            f"  {split:6s}: {s['rows']:6,}  "
            f"label=0: {s['label_0']:5,}  label=1: {s['label_1']:5,}  "
            f"ratio: {s['label_1_ratio']:.3f}"
        )
    print("─" * 60)
