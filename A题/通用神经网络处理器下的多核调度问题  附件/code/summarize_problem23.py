"""Summarize the corrected five-core problem 2/3 batch."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


def describe(values):
    ordered = sorted(values)
    n = len(ordered)
    return {
        "arithmetic_mean": statistics.fmean(values),
        "geometric_mean": math.exp(statistics.fmean(math.log(value) for value in values)),
        "median": statistics.median(values),
        "p10": ordered[max(0, math.ceil(0.10 * n) - 1)],
        "p90": ordered[max(0, math.ceil(0.90 * n) - 1)],
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "day5_results"
    q23_path = results / "problem23_5core_corrected" / "problem23_5core_summary.csv"
    q1_path = results / "paper_guided_v6_summary.csv"
    q1_stats_path = results / "paper_guided_v6_stats.json"
    q23_rows = {
        row["case"]: row
        for row in csv.DictReader(q23_path.open(encoding="utf-8-sig", newline=""))
    }
    q1_rows = {
        row["case"]: row
        for row in csv.DictReader(q1_path.open(encoding="utf-8-sig", newline=""))
    }
    names = sorted(q23_rows)
    if len(names) != 100 or set(names) != set(q1_rows):
        raise RuntimeError("expected exactly the same 100 cases in q1 and q2/3 results")

    rows = []
    for name in names:
        q = q23_rows[name]
        b = q1_rows[name]
        baseline = int(float(b["baseline_makespan"]))
        q1 = int(float(b["problem1_makespan"]))
        q2 = int(float(q["q2_makespan"]))
        q3 = int(float(q["q3_makespan"]))
        scheduled = int(float(q["q2_scheduled_copy_bytes"]))
        physical = int(float(q["q3_physical_ddr_bytes"]))
        rows.append({
            "case": name,
            "baseline_makespan": baseline,
            "q1_makespan": q1,
            "q2_makespan": q2,
            "q3_makespan": q3,
            "q1_speedup": baseline / q1,
            "q2_speedup": baseline / q2,
            "q3_speedup": baseline / q3,
            "q3_over_q2": q2 / q3,
            "q2_scheduled_copy_bytes": scheduled,
            "q3_physical_ddr_bytes": physical,
            "cache_saved_bytes": scheduled - physical,
            "cache_hits": int(float(q["cache_hits"])),
            "cache_accesses": int(float(q["cache_accesses"])),
            "cache_hit_rate": float(q["cache_hit_rate"]),
        })

    baseline_total = sum(row["baseline_makespan"] for row in rows)
    q2_total = sum(row["q2_makespan"] for row in rows)
    q3_total = sum(row["q3_makespan"] for row in rows)
    q1_stats = json.loads(q1_stats_path.read_text(encoding="utf-8"))
    report = {
        "scope": {
            "num_cases": 100,
            "num_cores": 5,
            "plan_dir": "day5_results/paper_guided_v6_plans",
            "q23_summary": str(q23_path.relative_to(root)),
        },
        "validation": {
            "case_count": len(rows),
            "error_count": sum(bool(q23_rows[name].get("error")) for name in names),
            "unique_case_count": len(set(names)),
            "b7": {
                "makespan": 6,
                "scheduled_copy_bytes": 32,
                "added_copy_bytes": 0,
                "q3_cache_hits": 0,
            },
            "fanout_regression": "one source COPY_OUT reused by two target COPY_IN operations",
        },
        "problem1": q1_stats,
        "problem2": {
            "makespan": describe([row["q2_makespan"] for row in rows]),
            "speedup": describe([row["q2_speedup"] for row in rows]),
            "faster_than_singlecore": sum(row["q2_makespan"] < row["baseline_makespan"] for row in rows),
            "equal_to_singlecore": sum(row["q2_makespan"] == row["baseline_makespan"] for row in rows),
            "slower_than_singlecore": sum(row["q2_makespan"] > row["baseline_makespan"] for row in rows),
            "total_baseline_cycles": baseline_total,
            "total_makespan_cycles": q2_total,
            "aggregate_speedup": baseline_total / q2_total,
            "total_scheduled_copy_bytes": sum(row["q2_scheduled_copy_bytes"] for row in rows),
        },
        "problem3": {
            "makespan": describe([row["q3_makespan"] for row in rows]),
            "speedup": describe([row["q3_speedup"] for row in rows]),
            "faster_than_singlecore": sum(row["q3_makespan"] < row["baseline_makespan"] for row in rows),
            "equal_to_singlecore": sum(row["q3_makespan"] == row["baseline_makespan"] for row in rows),
            "slower_than_singlecore": sum(row["q3_makespan"] > row["baseline_makespan"] for row in rows),
            "total_baseline_cycles": baseline_total,
            "total_makespan_cycles": q3_total,
            "aggregate_speedup": baseline_total / q3_total,
            "total_physical_ddr_bytes": sum(row["q3_physical_ddr_bytes"] for row in rows),
            "total_cache_saved_bytes": sum(row["cache_saved_bytes"] for row in rows),
            "cache_hit_cases": sum(row["cache_hits"] > 0 for row in rows),
            "cache_zero_hit_cases": sum(row["cache_hits"] == 0 for row in rows),
            "cache_accesses": sum(row["cache_accesses"] for row in rows),
            "cache_hits": sum(row["cache_hits"] for row in rows),
            "arithmetic_mean_case_hit_rate": statistics.fmean(row["cache_hit_rate"] for row in rows),
        },
        "q3_vs_q2": {
            "makespan_ratio_q2_over_q3": describe([row["q3_over_q2"] for row in rows]),
            "q3_faster_cases": sum(row["q3_makespan"] < row["q2_makespan"] for row in rows),
            "equal_cases": sum(row["q3_makespan"] == row["q2_makespan"] for row in rows),
            "q3_slower_cases": sum(row["q3_makespan"] > row["q2_makespan"] for row in rows),
            "total_q2_cycles": q2_total,
            "total_q3_cycles": q3_total,
            "aggregate_ratio_q2_over_q3": q2_total / q3_total,
        },
        "case_rows": rows,
    }
    json_path = results / "problem23_5core_corrected_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def num(value):
        return "{:.6f}".format(value)

    p1 = report["problem1"]
    p2 = report["problem2"]
    p3 = report["problem3"]
    q32 = report["q3_vs_q2"]
    lines = [
        "# 问题二、问题三五核修正版全量报告",
        "",
        "评估对象为同一套 paper_guided_v6 五核方案，问题一沿用已冻结 v6；问题二、三使用修正后的场景 B 评估器。共 100 个 case，错误数为 {}。".format(report["validation"]["error_count"]),
        "",
        "## 平均加速比",
        "",
        "| 问题 | 算术平均 | 几何平均 | 中位数 | 总周期比 |",
        "|---|---:|---:|---:|---:|",
        "| 问题一 | {} | {} | {} | {} |".format(num(p1["arithmetic_mean_speedup"]), num(p1["geometric_mean_speedup"]), num(p1["median_speedup"]), num(p1["aggregate_cycle_ratio"])),
        "| 问题二 | {} | {} | {} | {} |".format(num(p2["speedup"]["arithmetic_mean"]), num(p2["speedup"]["geometric_mean"]), num(p2["speedup"]["median"]), num(p2["aggregate_speedup"])),
        "| 问题三 | {} | {} | {} | {} |".format(num(p3["speedup"]["arithmetic_mean"]), num(p3["speedup"]["geometric_mean"]), num(p3["speedup"]["median"]), num(p3["aggregate_speedup"])),
        "",
        "问题一的算术平均加速比为 3.357806×；问题二、三按同一单核基线重新计算。总周期比采用 100 个 case 的单核周期总和除以对应多核 Makespan 总和，和算术平均不是同一个统计量。",
        "",
        "## 问题二与问题三",
        "",
        "- 问题二比单核更快的 case：{}/100；问题三比单核更快的 case：{}/100。".format(p2["faster_than_singlecore"], p3["faster_than_singlecore"]),
        "- 问题三相对问题二更快：{} 个，持平：{} 个，变慢：{} 个；总周期比 Q2/Q3 = {}。".format(q32["q3_faster_cases"], q32["equal_cases"], q32["q3_slower_cases"], num(q32["aggregate_ratio_q2_over_q3"])),
        "- 问题二逻辑/DDR 搬运总量：{:,} bytes；问题三物理 DDR 总量：{:,} bytes；由 Cache 读带宽承接的搬运差额：{:,} bytes。".format(p2["total_scheduled_copy_bytes"], p3["total_physical_ddr_bytes"], p3["total_cache_saved_bytes"]),
        "- 问题三累计 Cache 访问 {} 次，命中 {} 次；有命中的 case 为 {}/100，逐 case 命中率算术平均为 {}。".format(p3["cache_accesses"], p3["cache_hits"], p3["cache_hit_cases"], num(p3["arithmetic_mean_case_hit_rate"])),
        "",
        "## 模型修正与回归",
        "",
        "- 问题二/三对同一逻辑 Tensor 按 (tensor, source_core, target_core) 建立跨核传输；一个源核心向多个远端核心发送时只写一次 DDR，远端核心各自生成 COPY_IN。",
        "- 问题三只有原始图输入 COPY_IN 和由原始输入 backing 产生的 spill reload 可查询只读 FIFO L2；跨核中间 Tensor、COPY_OUT 和计算结果 spill 不进入 Cache。命中使用独立 Cache 带宽池，不计入物理 DDR。",
        "- B.7 回归：三个问题均为 Makespan 6、COPY 32 bytes、新增搬运 0；问题三命中 0。另有 fan-out 回归确认一个源 COPY_OUT 被两个目标 COPY_IN 复用。",
        "",
        "完整逐 case 数据见同目录 problem23_5core_corrected_report.json，原始汇总见 problem23_5core_corrected/problem23_5core_summary.csv。",
    ]
    md_path = results / "problem23_5core_corrected_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)
    print(json.dumps({
        "problem1": p1,
        "problem2": p2,
        "problem3": p3,
        "q3_vs_q2": q32,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
