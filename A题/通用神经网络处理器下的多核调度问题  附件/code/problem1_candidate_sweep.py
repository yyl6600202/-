"""Compare structured problem-1 partition rules with the official evaluator.

This is an offline search helper: it evaluates each requested rule for each
case/core count and writes one CSV row per candidate.  The official makespan,
not the inexpensive partition score, determines the winner.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation_validation import read_evaluation_config
from graph_features import build_op_dag, extract_features
from initial_partition import generate_candidates
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config


ALL_RULES = ("single", "continuous", "wave", "aligned", "components")
FIELDS = (
    "case", "num_cores", "graph_class", "rule", "num_blocks",
    "baseline_makespan", "makespan", "speedup", "added_copy_bytes",
    "spill_added_copy_bytes", "error",
)


def case_names(data_dir):
    return sorted(
        path.stem for path in Path(data_dir).glob("case_*.json")
        if re.fullmatch(r"case_\d{3}", path.stem)
    )


def evaluate_case(case_name, data_dir, num_cores, target_blocks, rules):
    data_dir = Path(data_dir)
    graph = json.loads((data_dir / f"{case_name}.json").read_text(encoding="utf-8"))
    config_path = data_dir / "config.txt"
    config = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)
    graph_class = extract_features(
        graph, build_op_dag(graph), config["capacity"]
    )["graph_class"]
    candidates = generate_candidates(
        graph,
        num_cores,
        target_blocks=target_blocks or num_cores,
        rules=rules,
    )
    evaluated = []
    for candidate in candidates:
        row = {
            "case": case_name,
            "num_cores": num_cores,
            "graph_class": graph_class,
            "rule": candidate["rule"],
            "num_blocks": len(candidate["blocks"]),
            "baseline_makespan": "",
            "makespan": "",
            "speedup": "",
            "added_copy_bytes": "",
            "spill_added_copy_bytes": "",
            "error": "",
        }
        try:
            if not candidate["valid"]:
                raise ValueError(candidate["validation_error"] or "candidate failed plan validation")
            result = evaluate_scene_a(
                graph,
                candidate["plan"],
                config["bandwidth"],
                config["capacity"],
                waits["task_cross_core_wait_cycles"],
                waits["task_same_core_wait_cycles"],
            )
            movement = result.get("data_movement_bytes", {})
            row["makespan"] = result["makespan"]
            row["added_copy_bytes"] = movement.get("added_copy_bytes", 0)
            row["spill_added_copy_bytes"] = movement.get("spill_added_copy_bytes", 0)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        evaluated.append(row)
    # Use the single-subgraph evaluation as the per-case single-core baseline.
    baseline_row = next(row for row in evaluated if row["rule"] == "single")
    baseline = baseline_row["makespan"]
    for row in evaluated:
        row["baseline_makespan"] = baseline
        if baseline and row["makespan"]:
            row["speedup"] = baseline / row["makespan"]
    return evaluated


def run(data_dir, output_path, workers, num_cores, target_blocks,
        selected_cases=None, rules=ALL_RULES, resume=False):
    if "single" not in rules:
        raise ValueError("the candidate sweep must include the 'single' baseline rule")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = case_names(data_dir)
    if selected_cases:
        requested = set(selected_cases)
        cases = [case for case in cases if case in requested]
    existing = []
    completed = set()
    if resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
        completed = {
            (row["case"], int(row["num_cores"]), row["rule"])
            for row in existing
            if not row.get("error")
        }
    pending = [
        case for case in cases
        if not all((case, num_cores, rule) in completed for rule in rules)
    ]
    rows = list(existing)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                evaluate_case, case, data_dir, num_cores, target_blocks, rules,
            ): case
            for case in pending
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                result_rows = future.result()
            except Exception as error:
                result_rows = [{
                    "case": case,
                    "num_cores": num_cores,
                    "graph_class": "",
                    "rule": rule,
                    "num_blocks": "",
                    "baseline_makespan": "",
                    "makespan": "",
                    "speedup": "",
                    "added_copy_bytes": "",
                    "spill_added_copy_bytes": "",
                    "error": f"{type(error).__name__}: {error}",
                } for rule in rules]
            by_key = {
                (row["case"], int(row["num_cores"]), row["rule"]): row
                for row in rows
            }
            by_key.update({
                (row["case"], int(row["num_cores"]), row["rule"]): row
                for row in result_rows
            })
            rows = list(by_key.values())
            for row in result_rows:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(sorted(
                    rows,
                    key=lambda row: (row["case"], int(row["num_cores"]), row["rule"]),
                ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("candidate_sweep.csv"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--target-blocks", type=int)
    parser.add_argument("--case", action="append", dest="cases",
                        help="evaluate only this case; may be repeated")
    parser.add_argument("--rules", nargs="+", choices=ALL_RULES, default=list(ALL_RULES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(
        args.data_dir, args.output, args.workers, args.num_cores,
        args.target_blocks, args.cases, args.rules, args.resume,
    )


if __name__ == "__main__":
    main()
