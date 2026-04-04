from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV = "runs/n30_q127_h3_paperstyle_200k/train_gpu_binary/metrics.csv"
TARGET_COLS = ["train_loss", "train_acc1", "valid_xe_loss", "valid_acc1"]


def plot_metrics(csv_path: str, out_suffix: str) -> None:
    df = pd.read_csv(csv_path)

    # Prefer epoch on x-axis when available.
    x_col = df["epoch"] if "epoch" in df.columns else df.index

    target_cols = [c for c in TARGET_COLS if c in df.columns]
    missing = [c for c in TARGET_COLS if c not in df.columns]

    if missing:
        print(f"[{csv_path}] missing columns: {missing}")
        print(f"[{csv_path}] available columns: {list(df.columns)}")

    if not target_cols:
        print(f"[{csv_path}] no target columns found, skip plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    for i, col in enumerate(target_cols[:4]):
        ax = axes[i]
        ax.plot(x_col, df[col], linewidth=1.8)
        ax.set_title(col)
        ax.grid(alpha=0.3)
        ax.set_ylabel(col)

    # Hide any unused subplot panels.
    for i in range(len(target_cols[:4]), 4):
        axes[i].axis("off")

    x_label = "epoch" if "epoch" in df.columns else "index"
    axes[-1].set_xlabel(x_label)
    axes[-2].set_xlabel(x_label)

    plt.tight_layout()
    out_path = str(Path(csv_path).with_name(out_suffix))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics CSV files into 4-panel PNGs")
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="One or more CSV paths. If omitted, uses the default sample path.",
    )
    parser.add_argument(
        "--out-suffix",
        default="metrics_4panel.png",
        help="Output filename to save next to each input CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = args.csv_paths if args.csv_paths else [DEFAULT_CSV]

    for csv_path in csv_paths:
        plot_metrics(csv_path, args.out_suffix)


if __name__ == "__main__":
    main()
