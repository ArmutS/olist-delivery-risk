"""Command-line interface for audit and reproducible model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_order_level_dataset
from .evaluation import json_safe
from .modeling import train_and_evaluate
from .reporting import write_report_bundle


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="olist-risk")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Build and validate the point-in-time-safe order table")
    audit.add_argument("--data-dir", type=Path, default=project_root() / "data" / "raw")

    run = subparsers.add_parser("run", help="Run the complete chronological modeling workflow")
    run.add_argument("--data-dir", type=Path, default=project_root() / "data" / "raw")
    run.add_argument("--config", type=Path, default=project_root() / "configs" / "project.json")
    run.add_argument("--reports-dir", type=Path, default=project_root() / "reports")
    run.add_argument("--artifacts-dir", type=Path, default=project_root() / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, diagnostics = build_order_level_dataset(args.data_dir)
    if args.command == "audit":
        print(json.dumps(json_safe(diagnostics), indent=2, ensure_ascii=False))
        return

    config = load_config(args.config)
    result = train_and_evaluate(
        frame,
        proportions=config["split_proportions"],
        random_state=int(config["random_state"]),
        high_capacity=float(config["high_risk_capacity"]),
        medium_capacity=float(config["medium_risk_capacity"]),
    )
    summary = write_report_bundle(
        frame,
        result,
        diagnostics,
        config,
        args.reports_dir,
        args.artifacts_dir,
    )
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
