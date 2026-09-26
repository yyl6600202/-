"""Run a bounded, reproducible official problem-1 evaluation over all 100 cases."""

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


RULE_BY_CLASS = {
    "wide_shallow": "continuous",
    "mixed": "components",
    "high_spill_risk": "components",
    "deep_narrow": "single",
}
ALL_RULES = ("single", "continuous", "wave", "aligned", "components")


def _case_names(data_dir):
    return sorted(
        path.stem
        for path in Path(data_dir).glob("case_*.json")
        if re.fullmatch(r"case_\d{3}", path.stem)
    )


def _evaluate_case(case_name, data_dir, rule_by_class, num_cores=4,
                   target_blocks=None, adaptive_wide=False, plans_dir=None):
    data_dir = Path(data_dir)
    graph = json.loads((data_dir / f"{case_name}.json").read_text(encoding="utf-8"))
    config_path = data_dir / "config.txt"
    config = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)
    profile = extract_features(graph, build_op_dag(graph), config["capacity"])
    graph_class = profile["graph_class"]
    selected_rule = rule_by_class.get(graph_class, "components")
    candidate_rules = (
        ALL_RULES if adaptive_wide and graph_class == "wide_shallow"
        else tuple(dict.fromkeys(("single", selected_rule)))
    )
    candidates = generate_candidates(
        graph,
        num_cores,
        target_blocks=target_blocks or num_cores,
        rules=candidate_rules,
    )
    results = {}
    for candidate in candidates:
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
            results[candidate["rule"]] = {
                "makespan": result["makespan"],
                "added_copy_bytes": movement.get("added_copy_bytes", 0),
                "spill_added_copy_bytes": movement.get("spill_added_copy_bytes", 0),
                "num_blocks": len(candidate["blocks"]),
                "plan": candidate["plan"],
            }
        except Exception as error:
            results[candidate["rule"]] = {"error": f"{type(error).__name__}: {error}"}

    if "single" not in results or "makespan" not in results["single"]:
        raise RuntimeError("official evaluator did not accept the single-core baseline")
    baseline = results["single"]["makespan"]
    successful = {
        rule: result for rule, result in results.items()
        if "makespan" in result
    }
    # Official makespan is the objective; data movement and block count only
    # break ties. Keeping single in the candidate set prevents regressions.
    winner_rule, candidate = min(
        successful.items(),
        key=lambda item: (
            item[1]["makespan"], item[1]["added_copy_bytes"],
            item[1]["num_blocks"], item[0] != "single",
        ),
    )
    if plans_dir is not None:
        case_plans_dir = Path(plans_dir)
        case_plans_dir.mkdir(parents=True, exist_ok=True)
        (case_plans_dir / f"{case_name}.json").write_text(
            json.dumps(candidate["plan"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "case": case_name,
        "graph_class": graph_class,
        "rule": winner_rule,
        "baseline_makespan": baseline,
        "problem1_makespan": candidate["makespan"],
        "speedup": baseline / candidate["makespan"] if candidate["makespan"] else None,
        "baseline_added_copy_bytes": results["single"]["added_copy_bytes"],
        "problem1_added_copy_bytes": candidate["added_copy_bytes"],
        "problem1_spill_added_copy_bytes": candidate["spill_added_copy_bytes"],
        "num_blocks": candidate["num_blocks"],
        "evaluated_candidates": len(successful),
        "rejected_candidates": len(results) - len(successful),
        "selection": "best_official_candidate_with_single_fallback",
        "error": "",
    }


def run(data_dir, output_path, workers, rule_by_class, only_cases=None,
        skip_cases=(), resume=False, num_cores=4, target_blocks=None,
        adaptive_wide=False, plans_dir=None):
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = _case_names(data_dir)
    if only_cases:
        requested = set(only_cases)
        cases = [case for case in cases if case in requested]
    cases = [case for case in cases if case not in set(skip_cases)]
    rows = []
    if resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        # Failed rows remain retryable on resume.
        completed = {row["case"] for row in rows if not row.get("error")}
        cases = [case for case in cases if case not in completed]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _evaluate_case, case_name, str(data_dir), rule_by_class,
                num_cores, target_blocks, adaptive_wide, plans_dir,
            ): case_name
            for case_name in cases
        }
        for future in as_completed(futures):
            case_name = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "case": case_name,
                    "graph_class": "",
                    "rule": rule_by_class.get("", "components"),
                    "baseline_makespan": "",
                    "problem1_makespan": "",
                    "speedup": "",
                    "baseline_added_copy_bytes": "",
                    "problem1_added_copy_bytes": "",
                    "problem1_spill_added_copy_bytes": "",
                    "num_blocks": "",
                    "evaluated_candidates": 0,
                    "rejected_candidates": 0,
                    "selection": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            rows = [row for row in rows if row["case"] != case_name]
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
                fields = [
                    "case", "graph_class", "rule", "baseline_makespan",
                    "problem1_makespan", "speedup", "baseline_added_copy_bytes",
                    "problem1_added_copy_bytes", "problem1_spill_added_copy_bytes",
                    "num_blocks", "evaluated_candidates", "rejected_candidates",
                    "selection", "error",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda item: item["case"]))
    return rows


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a fixed problem-1 rule over all cases")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("full_problem1_summary.csv"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--target-blocks", type=int,
                        help="partition count for continuous/wave/aligned rules; defaults to num-cores")
    parser.add_argument("--case", action="append", dest="only_cases",
                        help="evaluate only this case; may be repeated")
    parser.add_argument("--skip-case", action="append", default=[],
                        help="skip this case; may be repeated")
    parser.add_argument("--resume", action="store_true",
                        help="keep existing output rows and evaluate only missing cases")
    parser.add_argument("--adaptive-wide", action="store_true",
                        help="evaluate all five partition rules for wide_shallow graphs")
    parser.add_argument("--plans-dir", type=Path,
                        help="save each selected official plan as CASE.json")
    args = parser.parse_args(argv)
    run(args.data_dir, args.output, args.workers, RULE_BY_CLASS,
        args.only_cases, args.skip_case, args.resume,
        args.num_cores, args.target_blocks, args.adaptive_wide, args.plans_dir)


if __name__ == "__main__":
    _main()
