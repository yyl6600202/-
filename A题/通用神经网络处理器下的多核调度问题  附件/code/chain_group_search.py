"""Search the number of contiguous root groups for difficult 5-core cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from chain_cluster_partition import _execution_order, _make_plan, _roots_and_lineage, _split_contiguous
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from graph_features import build_op_dag


GROUP_COUNTS = (6, 8, 10, 12, 16, 20, 24, 32)


def _assign(groups, mode, num_cores, branch_cycles):
    if mode == "cyclic":
        return {root: index % num_cores for index, group in enumerate(groups) for root in group}
    pattern = list(range(num_cores)) + list(range(num_cores - 2, 0, -1))
    return {root: pattern[index % len(pattern)] for index, group in enumerate(groups) for root in group}


def search_case(case, data_dir, current, plans_dir, num_cores=5):
    data_dir = Path(data_dir)
    graph = json.loads((data_dir / f"{case}.json").read_text(encoding="utf-8"))
    config_path = data_dir / "config.txt"
    config = read_evaluation_config(config_path)
    waits = read_scene_a_config(config_path)
    graph_data = build_op_dag(graph)
    order = _execution_order(graph_data)
    candidates = []
    seen = set()
    for dominant in ("id", "bytes"):
        lineage, roots, branch_cycles, _ = _roots_and_lineage(graph_data, order, dominant)
        for group_count in GROUP_COUNTS:
            for weighted in (False, True):
                groups = _split_contiguous(
                    roots, [branch_cycles[root] for root in roots], group_count,
                    weighted=weighted,
                )
                for assignment in ("cyclic", "snake"):
                    root_to_core = _assign(groups, assignment, num_cores, branch_cycles)
                    plan, blocks = _make_plan(order, lineage, root_to_core, num_cores)
                    fingerprint = json.dumps(plan, sort_keys=True, separators=(",", ":"))
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    try:
                        result = evaluate_scene_a(
                            graph, plan, config["bandwidth"], config["capacity"],
                            waits["task_cross_core_wait_cycles"],
                            waits["task_same_core_wait_cycles"],
                        )
                    except Exception:
                        continue
                    movement = result.get("data_movement_bytes", {})
                    candidates.append({
                        "rule": f"group_{dominant}_{group_count}_{'weighted' if weighted else 'uniform'}_{assignment}",
                        "makespan": result["makespan"],
                        "added_copy_bytes": movement.get("added_copy_bytes", 0),
                        "num_blocks": len(blocks),
                        "plan": plan,
                    })
    if not candidates:
        return {"case": case, "improved": False, "error": "no candidate evaluated"}
    best = min(candidates, key=lambda x: (x["makespan"], x["added_copy_bytes"], x["num_blocks"], x["rule"]))
    current_makespan = int(float(current["problem1_makespan"]))
    improved = best["makespan"] < current_makespan
    if improved:
        output = Path(plans_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{case}.json").write_text(
            json.dumps(best["plan"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {
        "case": case,
        "improved": improved,
        "current_makespan": current_makespan,
        "new_makespan": best["makespan"],
        "new_speedup": float(current["baseline_makespan"]) / best["makespan"],
        "rule": best["rule"],
        "added_copy_bytes": best["added_copy_bytes"],
        "num_blocks": best["num_blocks"],
        "evaluated_candidates": len(candidates),
        "error": "",
        "plan": best["plan"] if improved else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--current-summary", type=Path, default=Path("day5_results/chain_generator_5core_v2_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("day5_results/chain_group_delta.csv"))
    parser.add_argument("--plans-dir", type=Path, default=Path("day5_results/chain_group_plans"))
    parser.add_argument("--case", action="append", dest="cases", required=True)
    args = parser.parse_args(argv)
    with args.current_summary.open(newline="", encoding="utf-8-sig") as handle:
        current = {row["case"]: row for row in csv.DictReader(handle)}
    fields = ["case", "improved", "current_makespan", "new_makespan", "new_speedup", "rule", "added_copy_bytes", "num_blocks", "evaluated_candidates", "error"]
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(args.cases, 1):
        try:
            result = search_case(case, args.data_dir, current[case], args.plans_dir)
        except Exception as exc:
            result = {"case": case, "improved": False, "error": f"{type(exc).__name__}: {exc}"}
        row = {field: result.get(field, "") for field in fields}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"progress {index}/{len(args.cases)}", flush=True)


if __name__ == "__main__":
    main()
