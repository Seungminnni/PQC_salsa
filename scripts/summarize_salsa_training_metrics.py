#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import re
from pathlib import Path


RUN_NAME_RE = re.compile(r"secret(?P<secret_seed>\d+)_init(?P<init_seed>\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize SALSA training metrics and recovery status.")
    parser.add_argument("--run_glob", type=str, required=True, help="Glob for training run directories.")
    parser.add_argument("--output_csv", type=str, default="", help="Optional summary CSV output path.")
    parser.add_argument("--output_json", type=str, default="", help="Optional summary JSON output path.")
    return parser.parse_args()


def read_metrics_csv(path: Path):
    if not path.is_file():
        return None, None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    final_row = rows[-1]
    best_row = min(rows, key=lambda row: float(row["valid_xe_loss"]) if row["valid_xe_loss"] not in ("", None) else float("inf"))
    return final_row, best_row


def read_recovery(run_dir: Path):
    success_epoch = None
    success_methods = []
    for path in sorted(run_dir.glob("secret_recovery_*.pkl"), key=lambda item: int(item.stem.split("_")[-1])):
        payload = pickle.load(open(path, "rb"))
        methods = payload.get("success", [])
        if methods:
            success_epoch = int(path.stem.split("_")[-1])
            success_methods = list(methods)
            break
    return success_epoch is not None, success_epoch, success_methods


def parse_run_name(name: str):
    match = RUN_NAME_RE.search(name)
    if not match:
        return None, None
    return int(match.group("secret_seed")), int(match.group("init_seed"))


def to_float(value):
    if value in ("", None):
        return None
    return float(value)


def main():
    args = parse_args()
    run_dirs = sorted(Path(".").glob(args.run_glob))
    rows = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        final_row, best_row = read_metrics_csv(run_dir / "metrics.csv")
        secret_recovered, success_epoch, success_methods = read_recovery(run_dir)
        secret_seed, init_seed = parse_run_name(run_dir.name)
        row = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "secret_seed": secret_seed,
            "init_seed": init_seed,
            "final_epoch": int(final_row["epoch"]) if final_row else None,
            "final_train_loss": to_float(final_row["train_loss"]) if final_row else None,
            "final_train_acc1": to_float(final_row["train_acc1"]) if final_row else None,
            "final_train_acc2": to_float(final_row["train_acc2"]) if final_row else None,
            "final_valid_xe_loss": to_float(final_row["valid_xe_loss"]) if final_row else None,
            "final_valid_acc1": to_float(final_row["valid_acc1"]) if final_row else None,
            "final_valid_acc2": to_float(final_row["valid_acc2"]) if final_row else None,
            "best_valid_xe_epoch": int(best_row["epoch"]) if best_row else None,
            "best_valid_xe_loss": to_float(best_row["valid_xe_loss"]) if best_row else None,
            "secret_recovered": secret_recovered,
            "success_epoch": success_epoch,
            "success_methods": json.dumps(success_methods, ensure_ascii=True),
        }
        rows.append(row)

    output_csv = Path(args.output_csv) if args.output_csv else Path("training_summary.csv")
    output_json = Path(args.output_json) if args.output_json else Path("training_summary.json")
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_dir"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    output_json.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote training summary to {output_csv}")
    print(f"Wrote training summary JSON to {output_json}")


if __name__ == "__main__":
    main()
