"""Evaluate Day 2 initial problem-1 partitions on representative cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation_validation import read_evaluation_config
from initial_partition import generate_candidates
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config


DEFAULT_CASES = (
    "case_001",
    "case_010",
    "case_026",
    "case_051",
    "case_074",
    "case_083",
)
RULES = ("single", "continuous", "wave", "aligned", "components")


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _result_summary(result):
    movement = result.get("data_movement_bytes", {})
    return {
        "makespan": result.get("makespan"),
        "num_cores": result.get("num_cores"),
        "original_graph_copy_bytes": movement.get("original_graph_copy_bytes"),
        "scheduled_copy_bytes": movement.get("scheduled_copy_bytes"),
        "added_copy_bytes": movement.get("added_copy_bytes"),
        "partition_added_copy_bytes": movement.get("partition_added_copy_bytes"),
        "spill_added_copy_bytes": movement.get("spill_added_copy_bytes"),
        "cross_task_traffic_bytes": sum(
            item.get("bytes", 0) for item in result.get("cross_task_traffic", [])
        ) if isinstance(result.get("cross_task_traffic"), list) else result.get("cross_task_traffic", 0),
        "task_dependency_count": len(result.get("task_dependencies", [])),
    }


def evaluate_case(graph, candidates, config, waits, case_dir):
    rows = []
    for candidate in candidates:
        rule = candidate["rule"]
        plan_path = case_dir / f"{rule}.json"
        meta_path = case_dir / f"{rule}.meta.json"
        result_path = case_dir / f"{rule}.result.json"
        _write_json(plan_path, candidate["plan"])

        row = {
            "rule": rule,
            "valid": bool(candidate["valid"]),
            "num_blocks": len(candidate["blocks"]),
            "estimated_score": candidate["estimated_score"],
            "estimated_partition_added_copy_bytes": candidate[
                "estimated_partition_added_copy_bytes"
            ],
            "estimated_scheduled_boundary_bytes": candidate[
                "estimated_scheduled_boundary_bytes"
            ],
            "official_ok": False,
            "evaluation_error": "",
            "plan": str(plan_path),
            "result": str(result_path),
        }
        result = None
        if not candidate["valid"]:
            row["evaluation_error"] = candidate.get("validation_error", "invalid plan")
        else:
            try:
                result = evaluate_scene_a(
                    graph,
                    candidate["plan"],
                    config["bandwidth"],
                    config["capacity"],
                    waits["task_cross_core_wait_cycles"],
                    waits["task_same_core_wait_cycles"],
                )
            except Exception as error:  # retain all candidates for diagnosis
                row["evaluation_error"] = f"{type(error).__name__}: {error}"
            else:
                row["official_ok"] = True
                _write_json(result_path, result)
                row.update(_result_summary(result))

        meta = dict(candidate)
        meta["evaluation"] = {
            key: value
            for key, value in row.items()
            if key not in {"plan", "result"}
        }
        _write_json(meta_path, meta)
        rows.append(row)
    return rows


def run(data_dir, output_dir, case_names, num_cores, target_blocks, rules):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    config_path = data_dir / "config.txt"
    config = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)
    rows = []
    for case_name in case_names:
        graph_path = data_dir / f"{case_name}.json"
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        candidates = generate_candidates(
            graph,
            num_cores,
            target_blocks=target_blocks,
            rules=rules,
        )
        case_dir = output_dir / case_name / f"cores_{num_cores}"
        case_rows = evaluate_case(graph, candidates, config, waits, case_dir)
        for row in case_rows:
            row["case"] = case_name
            rows.append(row)
        _write_json(case_dir / "candidates.json", case_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "problem1_summary.csv"
    fieldnames = [
        "case",
        "rule",
        "num_blocks",
        "valid",
        "official_ok",
        "estimated_score",
        "estimated_partition_added_copy_bytes",
        "estimated_scheduled_boundary_bytes",
        "makespan",
        "original_graph_copy_bytes",
        "scheduled_copy_bytes",
        "added_copy_bytes",
        "partition_added_copy_bytes",
        "spill_added_copy_bytes",
        "cross_task_traffic_bytes",
        "task_dependency_count",
        "evaluation_error",
        "plan",
        "result",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return rows


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Day 2 problem-1 initial plans")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("day2_results"))
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--target-blocks", type=int)
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    args = parser.parse_args(argv)
    run(
        args.data_dir,
        args.output_dir,
        args.cases,
        args.num_cores,
        args.target_blocks,
        args.rules,
    )


if __name__ == "__main__":
    _main()
