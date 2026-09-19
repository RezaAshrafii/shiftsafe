from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .checks import run_quality_gate
from .contracts import DataContract
from .reporting import to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ShiftSafe quality gate on a CSV file.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--time-column")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    summary = run_quality_gate(
        frame,
        DataContract(target=args.target, time_column=args.time_column),
    )
    payload = summary.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(to_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
