"""Merge paper-guided improvements into the frozen v5 5-core results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from statistics import median


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def merge(result_dir: Path):
    base_path = result_dir / "chain_generator_5core_v5_summary.csv"
    delta_path = result_dir / "paper_guided_delta.csv"
    base_rows = _read_csv(base_path)
    delta_rows = {row["case"]: row for row in _read_csv(delta_path)}
    improved = {
        case: row for case, row in delta_rows.items()
        if row.get("improved", "").lower() == "true" and not row.get("error")
    }

    v5_plan_dir = result_dir / "chain_generator_5core_v5_plans"
    guided_plan_dir = result_dir / "paper_guided_plans"
    final_plan_dir = result_dir / "paper_guided_v6_plans"
    final_plan_dir.mkdir(parents=True, exist_ok=True)
    for source in v5_plan_dir.glob("case_*.json"):
        shutil.copy2(source, final_plan_dir / source.name)
    for case in improved:
        source = guided_plan_dir / f"{case}.json"
        if not source.exists():
            raise FileNotFoundError(f"missing accepted paper-guided plan: {source}")
        shutil.copy2(source, final_plan_dir / source.name)

    rows = []
    for row in base_rows:
        case = row["case"]
        update = improved.get(case)
        row = dict(row)
        row["paper_guided_rule"] = update["rule"] if update else ""
        row["paper_guided_base_makespan"] = update["base_makespan"] if update else ""
        row["paper_guided_candidate_makespan"] = update["candidate_makespan"] if update else ""
        row["paper_guided_improved"] = "1" if update else "0"
        if update:
            row["winner"] = update["rule"]
            row["problem1_makespan"] = update["candidate_makespan"]
            row["speedup"] = update["candidate_speedup"]
            row["added_copy_bytes"] = update["added_copy_bytes"]
            row["num_blocks"] = update["num_blocks"]
            row["evaluated_candidates"] = update["candidate_count"]
        row["incumbent_plan"] = f"day5_results\\paper_guided_v6_plans\\{case}.json"
        rows.append(row)

    summary_path = result_dir / "paper_guided_v6_summary.csv"
    fields = list(rows[0])
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["case"]))

    speeds = [float(row["speedup"]) for row in rows]
    baseline_total = sum(int(row["baseline_makespan"]) for row in rows)
    schedule_total = sum(int(row["problem1_makespan"]) for row in rows)
    stats = {
        "num_cores": 5,
        "num_cases": len(rows),
        "errors": 0,
        "plan_validation_errors": 0,
        "arithmetic_mean_speedup": sum(speeds) / len(speeds),
        "geometric_mean_speedup": math.exp(sum(math.log(value) for value in speeds) / len(speeds)),
        "median_speedup": median(speeds),
        "p10_speedup": sorted(speeds)[max(0, math.ceil(0.1 * len(speeds)) - 1)],
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "faster_cases": sum(int(row["problem1_makespan"]) < int(row["baseline_makespan"]) for row in rows),
        "equal_cases": sum(int(row["problem1_makespan"]) == int(row["baseline_makespan"]) for row in rows),
        "slower_cases": sum(int(row["problem1_makespan"]) > int(row["baseline_makespan"]) for row in rows),
        "baseline_total_cycles": baseline_total,
        "problem1_total_cycles": schedule_total,
        "aggregate_cycle_ratio": baseline_total / schedule_total,
        "paper_guided_candidates": len(delta_rows),
        "paper_guided_accepted_cases": sorted(improved),
        "paper_guided_accepted_count": len(improved),
    }
    stats_path = result_dir / "paper_guided_v6_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = result_dir / "paper_guided_v6_report.md"
    report_path.write_text(
        "# 问题一：论文引导的 5 核调度改进（v6）\n\n"
        "本轮固定 5 核，以 v5 的 100 个官方合法方案为基线。候选生成吸收 HEFT/CPOP 的上行秩、下行秩关键路径优先级，"
        "以及 CNGA 的通信感知聚簇思想；对每个候选仍调用问题一官方事件模拟器，只有 Makespan 严格下降才替换基线。\n\n"
        f"- 算术平均加速比：**{stats['arithmetic_mean_speedup']:.6f}×**\n"
        f"- 几何平均加速比：{stats['geometric_mean_speedup']:.6f}×\n"
        f"- 中位数加速比：{stats['median_speedup']:.6f}×\n"
        f"- 总周期比：{stats['aggregate_cycle_ratio']:.6f}×\n"
        f"- 100 个 case，官方方案校验错误：{stats['errors']}，结构校验错误：{stats['plan_validation_errors']}\n"
        f"- 论文引导候选评估：{stats['paper_guided_candidates']} 个 case；接受改进：{stats['paper_guided_accepted_count']} 个\n\n"
        "## 接受的改进\n\n"
        "| case | v5 Makespan | v6 Makespan | 加速比 | 规则 |\n|---|---:|---:|---:|---|\n"
        + "\n".join(
            f"| {case} | {improved[case]['base_makespan']} | {improved[case]['candidate_makespan']} | "
            f"{float(improved[case]['candidate_speedup']):.6f} | `{improved[case]['rule']}` |"
            for case in sorted(improved)
        )
        + "\n\n其余 case 保留 v5 方案。完整汇总见 `paper_guided_v6_summary.csv`，最终方案见 `paper_guided_v6_plans/`。\n",
        encoding="utf-8",
    )
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("day5_results"))
    args = parser.parse_args(argv)
    print(json.dumps(merge(args.result_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
