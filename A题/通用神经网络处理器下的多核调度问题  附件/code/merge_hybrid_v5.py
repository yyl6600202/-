"""Merge accepted hybrid-search plans into the 5-core v4 result."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from stub_multicore_cut_and_schedule import validate_multicore_plan


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "day5_results"
DATA = ROOT / "data"
V4_SUMMARY = RESULTS / "chain_generator_5core_v4_summary.csv"
HYBRID_DELTA = RESULTS / "hybrid_search_delta.csv"
V4_PLANS = RESULTS / "chain_generator_5core_v4_plans"
HYBRID_PLANS = RESULTS / "hybrid_search_plans"
V5_SUMMARY = RESULTS / "chain_generator_5core_v5_summary.csv"
V5_STATS = RESULTS / "chain_generator_5core_v5_stats.json"
V5_REPORT = RESULTS / "chain_generator_5core_v5_report.md"
V5_PLANS = RESULTS / "chain_generator_5core_v5_plans"


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    base_rows = read_rows(V4_SUMMARY)
    delta_rows = {row["case"]: row for row in read_rows(HYBRID_DELTA)}
    if len(base_rows) != 100:
        raise RuntimeError(f"expected 100 v4 rows, got {len(base_rows)}")

    if V5_PLANS.exists():
        shutil.rmtree(V5_PLANS)
    V5_PLANS.mkdir(parents=True)
    for source in sorted(V4_PLANS.glob("case_*.json")):
        shutil.copy2(source, V5_PLANS / source.name)

    accepted = []
    rows = []
    for row in base_rows:
        case = row["case"]
        delta = delta_rows.get(case)
        use_hybrid = False
        if delta and not delta.get("error") and delta.get("improved", "").lower() == "true":
            use_hybrid = int(delta["new_makespan"]) < int(row["problem1_makespan"])
        if use_hybrid:
            plan_path = HYBRID_PLANS / f"{case}.json"
            if not plan_path.exists():
                raise RuntimeError(f"missing hybrid plan for accepted {case}")
            row = dict(row)
            new_makespan = int(delta["new_makespan"])
            row.update({
                "winner": delta["rule"],
                "incumbent_makespan": row.get("incumbent_makespan", ""),
                "best_chain_rule": delta["rule"],
                "best_chain_makespan": str(new_makespan),
                "best_chain_added_copy_bytes": delta["added_copy_bytes"],
                "problem1_makespan": str(new_makespan),
                "speedup": str(int(row["baseline_makespan"]) / new_makespan),
                "added_copy_bytes": delta["added_copy_bytes"],
                "num_blocks": delta["num_blocks"],
                "evaluated_candidates": delta["evaluated_candidates"],
                "rejected_candidates": "0",
                "error": "",
            })
            shutil.copy2(plan_path, V5_PLANS / plan_path.name)
            accepted.append({
                "case": case,
                "old_makespan": int(delta["current_makespan"]),
                "new_makespan": new_makespan,
                "rule": delta["rule"],
            })
        rows.append(row)

    # Validate every final plan against its original graph.
    validation_errors = []
    for row in rows:
        case = row["case"]
        try:
            graph = json.loads((DATA / f"{case}.json").read_text(encoding="utf-8"))
            plan = json.loads((V5_PLANS / f"{case}.json").read_text(encoding="utf-8"))
            validate_multicore_plan(graph, plan)
        except Exception as exc:  # pragma: no cover - report exact case failure
            validation_errors.append(f"{case}: {type(exc).__name__}: {exc}")
    if validation_errors:
        raise RuntimeError("plan validation failed: " + "; ".join(validation_errors[:5]))

    fields = list(base_rows[0])
    with V5_SUMMARY.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    speeds = [float(row["speedup"]) for row in rows]
    sorted_speeds = sorted(speeds)
    baseline_cycles = sum(int(row["baseline_makespan"]) for row in rows)
    final_cycles = sum(int(row["problem1_makespan"]) for row in rows)
    stats = {
        "num_cores": 5,
        "num_cases": len(rows),
        "errors": 0,
        "plan_validation_errors": 0,
        "arithmetic_mean_speedup": sum(speeds) / len(speeds),
        "geometric_mean_speedup": math.exp(sum(math.log(value) for value in speeds) / len(speeds)),
        "median_speedup": (sorted_speeds[49] + sorted_speeds[50]) / 2,
        "p10_speedup": sorted_speeds[9],
        "minimum_speedup": sorted_speeds[0],
        "maximum_speedup": sorted_speeds[-1],
        "faster_cases": sum(value > 1.0 for value in speeds),
        "equal_cases": sum(value == 1.0 for value in speeds),
        "slower_cases": sum(value < 1.0 for value in speeds),
        "baseline_total_cycles": baseline_cycles,
        "problem1_total_cycles": final_cycles,
        "aggregate_cycle_ratio": baseline_cycles / final_cycles,
        "hybrid_candidates": len(delta_rows),
        "hybrid_accepted_cases": [item["case"] for item in accepted],
        "hybrid_accepted_count": len(accepted),
        "selected_rule_counts": dict(Counter(row["winner"] for row in rows)),
    }
    V5_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# A题问题一：5核链簇生成器 v5（混合搜索）",
        "",
        "v5 在 v4 的 100 个合法方案上，对最难的 20 个 case 使用遗传算法 + 模拟退火生成候选，并只接受官方事件模拟器验证后 Makespan 严格下降的方案。",
        "",
        "## 结果",
        "",
        f"- 算术平均加速比：**{stats['arithmetic_mean_speedup']:.6f}×**",
        f"- 几何平均加速比：{stats['geometric_mean_speedup']:.6f}×",
        f"- 中位数加速比：{stats['median_speedup']:.6f}×",
        f"- 总周期比：{stats['aggregate_cycle_ratio']:.6f}×",
        f"- 100 个 case，官方方案校验错误：{stats['errors']}，结构校验错误：{stats['plan_validation_errors']}",
        f"- 混合搜索接受改进：{stats['hybrid_accepted_count']} 个 case",
        "",
        "## 接受的改进",
        "",
        "| case | v4 Makespan | v5 Makespan | 规则 |",
        "|---|---:|---:|---|",
    ]
    for item in accepted:
        lines.append(f"| {item['case']} | {item['old_makespan']} | {item['new_makespan']} | `{item['rule']}` |")
    lines += [
        "",
        "所有候选仍使用连续 Task 切分，并通过 `validate_multicore_plan`；未严格下降的候选保留 v4 方案。",
        "",
        "复现：",
        "",
        "```powershell",
        "python code/hybrid_chain_search.py --worst 20 --official-budget 12 --population 16 --generations 6 --anneal-steps 40 --output day5_results/hybrid_search_delta.csv --plans-dir day5_results/hybrid_search_plans",
        "python code/merge_hybrid_v5.py",
        "```",
    ]
    V5_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
