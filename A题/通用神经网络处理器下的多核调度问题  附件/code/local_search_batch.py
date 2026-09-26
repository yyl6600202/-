"""Run bounded Day 3 local search from Day 2 official results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from local_search import objective, search


DEFAULT_CASES = ("case_001", "case_010")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _number(row, key, default=0):
    value = row.get(key, "")
    if value in (None, ""):
        return default
    return int(value)


def _select_start(summary_path, input_dir, case_name, start_rule):
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8-sig", newline="")))
    rows = [row for row in rows if row.get("case") == case_name and row.get("official_ok") == "True"]
    if start_rule != "auto":
        rows = [row for row in rows if row.get("rule") == start_rule]
    if not rows:
        raise ValueError(f"no official Day 2 result for {case_name}, rule={start_rule}")
    row = min(rows, key=lambda item: (_number(item, "makespan", 10**30), _number(item, "added_copy_bytes", 10**30)))
    plan_path = input_dir / case_name / "cores_4" / f"{row['rule']}.json"
    return row["rule"], json.loads(plan_path.read_text(encoding="utf-8"))


def run(cases, data_dir, input_dir, output_dir, start_rule, max_rounds, max_evaluations, proposal_limit):
    data_dir = Path(data_dir)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    summary_path = input_dir / "problem1_summary.csv"
    rows = []
    for case_name in cases:
        rule, start_plan = _select_start(summary_path, input_dir, case_name, start_rule)
        graph = json.loads((data_dir / f"{case_name}.json").read_text(encoding="utf-8"))
        result = search(
            graph,
            start_plan,
            data_dir / "config.txt",
            max_rounds=max_rounds,
            max_evaluations=max_evaluations,
            proposal_limit=proposal_limit,
        )
        case_dir = output_dir / case_name / "cores_4"
        start_path = case_dir / f"start_{rule}.json"
        best_path = case_dir / "best.json"
        best_result_path = case_dir / "best.result.json"
        history_path = case_dir / "history.json"
        _write_json(start_path, start_plan)
        _write_json(best_path, result["plan"])
        _write_json(best_result_path, result["result"])
        _write_json(history_path, result["history"])
        best_movement = result["result"].get("data_movement_bytes", {})
        row = {
            "case": case_name,
            "start_rule": rule,
            "start_makespan": _number(next(
                item for item in csv.DictReader(summary_path.open(encoding="utf-8-sig", newline=""))
                if item.get("case") == case_name and item.get("rule") == rule
            ), "makespan"),
            "best_makespan": result["result"].get("makespan"),
            "best_added_copy_bytes": best_movement.get("added_copy_bytes"),
            "best_spill_added_copy_bytes": best_movement.get("spill_added_copy_bytes"),
            "evaluations": result["evaluations"],
            "cache_entries": result["cache_entries"],
            "start": str(start_path),
            "best": str(best_path),
            "result": str(best_result_path),
            "history": str(history_path),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_out = output_dir / "local_search_summary.csv"
    fields = [
        "case", "start_rule", "start_makespan", "best_makespan",
        "best_added_copy_bytes", "best_spill_added_copy_bytes",
        "evaluations", "cache_entries", "start", "best", "result", "history",
    ]
    with summary_out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Run Day 3 problem-1 local search")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--input-dir", type=Path, default=Path("day2_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("day3_results"))
    parser.add_argument("--start-rule", choices=("auto", "single", "continuous", "wave", "aligned", "components"), default="auto")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--max-evaluations", type=int, default=24)
    parser.add_argument("--proposal-limit", type=int, default=32)
    args = parser.parse_args(argv)
    run(
        args.cases,
        args.data_dir,
        args.input_dir,
        args.output_dir,
        args.start_rule,
        args.max_rounds,
        args.max_evaluations,
        args.proposal_limit,
    )


if __name__ == "__main__":
    _main()
