"""Search the new multi-group chain rules against the saved 5-core result."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from chain_cluster_partition import generate_chain_candidates
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config


NEW_MODES = ("root_roundrobin", "root_roundrobin_weighted", "root_load")
DOMINANTS = ("bytes", "id")


def _official(graph, plan, config, waits):
    result = evaluate_scene_a(
        graph, plan, config["bandwidth"], config["capacity"],
        waits["task_cross_core_wait_cycles"], waits["task_same_core_wait_cycles"],
    )
    movement = result.get("data_movement_bytes", {})
    return {
        "makespan": result["makespan"],
        "added_copy_bytes": movement.get("added_copy_bytes", 0),
        "spill_added_copy_bytes": movement.get("spill_added_copy_bytes", 0),
        "plan": plan,
    }


def _case_names(data_dir):
    return sorted(path.stem for path in Path(data_dir).glob("case_*.json")
                  if re.fullmatch(r"case_\d{3}", path.stem))


def search_case(case_name, data_dir, current_row, num_cores, plans_dir):
    data_dir = Path(data_dir)
    graph = json.loads((data_dir / f"{case_name}.json").read_text(encoding="utf-8"))
    config_path = data_dir / "config.txt"
    config = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)
    candidates = generate_chain_candidates(
        graph, num_cores, config["bandwidth"], NEW_MODES, DOMINANTS)
    evaluated = []
    seen = set()
    for candidate in candidates:
        if not candidate["valid"]:
            continue
        fingerprint = json.dumps(candidate["plan"], sort_keys=True,
                                 separators=(",", ":"))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        try:
            result = _official(graph, candidate["plan"], config, waits)
        except Exception as exc:
            continue
        evaluated.append({
            "rule": candidate["rule"],
            "makespan": result["makespan"],
            "added_copy_bytes": result["added_copy_bytes"],
            "num_blocks": candidate["num_blocks"],
            "plan": result["plan"],
        })
    if not evaluated:
        return {
            "case": case_name, "improved": False, "error": "no valid new candidate",
            "current_makespan": int(float(current_row["problem1_makespan"])),
        }
    best = min(evaluated, key=lambda item: (
        item["makespan"], item["added_copy_bytes"], item["num_blocks"], item["rule"],
    ))
    current_makespan = int(float(current_row["problem1_makespan"]))
    improved = best["makespan"] < current_makespan
    if improved:
        output_dir = Path(plans_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{case_name}.json").write_text(
            json.dumps(best["plan"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "case": case_name,
        "improved": improved,
        "current_makespan": current_makespan,
        "new_makespan": best["makespan"],
        "new_speedup": float(current_row["baseline_makespan"]) / best["makespan"],
        "rule": best["rule"],
        "added_copy_bytes": best["added_copy_bytes"],
        "num_blocks": best["num_blocks"],
        "evaluated_candidates": len(evaluated),
        "error": "",
        "plan": best["plan"] if improved else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--current-summary", type=Path,
                        default=Path("day5_results/chain_generator_5core_summary.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("day5_results/chain_roundrobin_delta.csv"))
    parser.add_argument("--plans-dir", type=Path,
                        default=Path("day5_results/chain_roundrobin_plans"))
    parser.add_argument("--max-speedup", type=float, default=3.8)
    parser.add_argument("--num-cores", type=int, default=5)
    parser.add_argument("--case", action="append", dest="cases")
    args = parser.parse_args(argv)
    with args.current_summary.open(newline="", encoding="utf-8-sig") as handle:
        current = {row["case"]: row for row in csv.DictReader(handle)}
    cases = args.cases or [
        case for case in _case_names(args.data_dir)
        if case in current and float(current[case]["speedup"]) < args.max_speedup
    ]
    fields = ["case", "improved", "current_makespan", "new_makespan",
              "new_speedup", "rule", "added_copy_bytes", "num_blocks",
              "evaluated_candidates", "error"]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, case_name in enumerate(cases, 1):
        try:
            result = search_case(case_name, args.data_dir, current[case_name],
                                 args.num_cores, args.plans_dir)
        except Exception as exc:
            result = {"case": case_name, "improved": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        rows.append({field: result.get(field, "") for field in fields})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"progress {index}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()
