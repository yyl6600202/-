#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题二（场景 B）提交方案的独立复核脚本。

设计原则
--------
本脚本**完全不导入、不依赖提交方的任何求解代码**。它只做三件事：

  1. 用官方 ``stub_multicore_cut_and_schedule.validate_multicore_plan``
     校验每份方案的结构合法性；
  2. 用官方 ``multicore_cut_evaluate_problem_2.evaluate_scene_b``
     重新计算单核基线 M1 与五核 Makespan M5；
  3. 把复算得到的逐例加速比、以及聚合统计量，
     与 ``claimed_results.csv`` / ``claimed_summary.json`` 中申报的数字逐项比对。

因此如果本脚本报告 PASS，说明"用官方评估器重新模拟这些方案文件，
得到的数字与申报数字一致"——这一结论不依赖提交方的程序是否正确。

用法
----
    python verify.py --attachments "C:/path/to/题目附件"

可选参数：
    --cases   case_001 case_016 ...   只复核指定的若干用例（快速抽检）
    --workers 11                      并行进程数（默认取 CPU 数 - 1）
    --plans   ./plans_n5              方案目录
    --tolerance 0.0                   加速比允许的绝对误差（默认 0，即要求严格一致）
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# 官方模块加载（路径由命令行给出，脚本不假设任何提交方目录结构）
# --------------------------------------------------------------------------
_OFFICIAL = {}


def load_official(attachments: Path):
    """从题目附件中导入官方评估器；返回所需函数/配置读取器。"""
    code_dir = Path(attachments) / "code"
    data_dir = Path(attachments) / "data"
    if not code_dir.is_dir():
        raise SystemExit(f"[错误] 找不到官方代码目录: {code_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"[错误] 找不到数据目录: {data_dir}")
    sys.path.insert(0, str(code_dir))

    from evaluation_validation import read_evaluation_config           # noqa: E402
    from multicore_cut_evaluate_problem_2 import (                     # noqa: E402
        evaluate_scene_b, read_scene_b_config)
    from stub_multicore_cut_and_schedule import validate_multicore_plan  # noqa: E402

    config_path = data_dir / "config.txt"
    if not config_path.exists():
        raise SystemExit(f"[错误] 找不到配置文件: {config_path}")
    cfg = read_evaluation_config(config_path)
    scene_b = read_scene_b_config(config_path)
    return {
        "data_dir": data_dir,
        "bandwidth": cfg["bandwidth"],
        "capacity": cfg["capacity"],
        "delay": scene_b["cross_core_copy_delay_cycles"],
        "evaluate_scene_b": evaluate_scene_b,
        "validate_multicore_plan": validate_multicore_plan,
    }


def single_core_plan(graph):
    """题面口径的单核基线：全部非 COPY 算子放进同一个子图、只用一个核。"""
    ids = sorted(int(op["id"]) for op in graph["ops"]
                 if op.get("op") not in {"COPY_IN", "COPY_OUT"})
    return {"node_to_subgraph": {str(op): 0 for op in ids},
            "core_schedules": [[0] if ids else []]}


# --------------------------------------------------------------------------
# 单例复核
# --------------------------------------------------------------------------
def verify_one(case, plans_dir, official):
    """复核一个用例。

    返回 dict：case, ok(结构合法), M1, M5, speedup, error
    """
    graph_path = official["data_dir"] / f"{case}.json"
    plan_path = Path(plans_dir) / f"{case}.json"
    record = {"case": case, "ok": False, "M1": None, "M5": None,
              "speedup": None, "numsg": None, "cores": None, "error": ""}
    if not graph_path.exists():
        record["error"] = f"缺少计算图 {graph_path}"
        return record
    if not plan_path.exists():
        record["error"] = f"缺少方案 {plan_path}"
        return record

    with open(graph_path, encoding="utf-8") as fh:
        graph = json.load(fh)
    with open(plan_path, encoding="utf-8") as fh:
        plan = json.load(fh)

    # --- 1) 官方结构校验（覆盖性 / 子图数 / 子图 DAG / 同核顺序）---------
    try:
        official["validate_multicore_plan"](graph, plan)
    except Exception as exc:                                   # noqa: BLE001
        record["error"] = f"官方结构校验失败: {exc}"
        return record

    bw, cap, delay = official["bandwidth"], official["capacity"], official["delay"]
    ev = official["evaluate_scene_b"]

    try:
        base = ev(graph, single_core_plan(graph), bw, cap, delay)
        mine = ev(graph, plan, bw, cap, delay)
    except Exception as exc:                                   # noqa: BLE001
        record["error"] = f"官方评估失败: {exc}"
        return record

    record["ok"] = True
    record["M1"] = int(base["makespan"])
    record["M5"] = int(mine["makespan"])
    record["speedup"] = record["M1"] / record["M5"] if record["M5"] else float("inf")
    record["numsg"] = len(set(plan["node_to_subgraph"].values()))
    record["cores"] = sum(1 for row in plan["core_schedules"] if row)
    return record


def _worker(args):
    case, plans_dir, attachments = args
    official = load_official(Path(attachments))
    return verify_one(case, plans_dir, official)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def read_claimed(csv_path):
    claimed = {}
    with open(csv_path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            claimed[row["case"]] = row
    return claimed


def main():
    parser = argparse.ArgumentParser(description="问题二提交方案独立复核")
    parser.add_argument("--attachments", required=True,
                        help="题目附件目录（其下应有 code/ 与 data/）")
    parser.add_argument("--plans", default=str(HERE / "plans_n5"))
    parser.add_argument("--claimed", default=str(HERE / "claimed_results.csv"))
    parser.add_argument("--summary", default=str(HERE / "claimed_summary.json"))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="加速比允许的绝对误差，默认 0（严格一致）")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    attachments = Path(args.attachments)
    official_head = load_official(attachments)
    print(f"官方评估器已加载：{Path(attachments) / 'code'}")
    print(f"  带宽 = {official_head['bandwidth']} B/cycle")
    print(f"  容量 = L1 {official_head['capacity']['L1']} / UB {official_head['capacity']['UB']}")
    print(f"  场景 B 跨核延迟 = {official_head['delay']} cycles")
    print()

    claimed = read_claimed(args.claimed)
    if args.cases:
        cases = list(args.cases)
    else:
        cases = sorted(claimed)
    missing = [c for c in cases if c not in claimed]
    if missing:
        raise SystemExit(f"[错误] claimed_results.csv 中缺少这些用例: {missing[:5]}")

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    plans_dir = args.plans
    payload = [(c, plans_dir, str(attachments)) for c in cases]

    print(f"开始复核 {len(cases)} 个用例，并行度 {workers} ……")
    if workers > 1 and len(cases) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(cases))) as pool:
            results = pool.map(_worker, payload, chunksize=1)
    else:
        results = [_worker(p) for p in payload]

    # --- 逐例比对 --------------------------------------------------------
    failures = []
    base_mismatch = []
    max_diff = 0.0
    for rec in results:
        case = rec["case"]
        if not rec["ok"]:
            failures.append((case, rec["error"]))
            continue
        row = claimed[case]
        base_claim = int(row["baseline_makespan"])
        ms_claim = int(row["makespan"])
        sp_claim = float(row["speedup"])
        if rec["M1"] != base_claim:
            base_mismatch.append((case, base_claim, rec["M1"]))
        if rec["M5"] != ms_claim:
            failures.append((case, f"Makespan 不一致：申报 {ms_claim}，复算 {rec['M5']}"))
            continue
        diff = abs(rec["speedup"] - sp_claim)
        max_diff = max(max_diff, diff)
        if diff > args.tolerance:
            failures.append((case, f"加速比不一致：申报 {sp_claim!r}，复算 {rec['speedup']!r}"))

    ok_results = [r for r in results if r["ok"]]
    speedups = [r["speedup"] for r in ok_results]

    if speedups:
        n = len(speedups)
        mean = sum(speedups) / n
        geo = math.exp(sum(math.log(s) for s in speedups) / n) if all(s > 0 for s in speedups) else float("nan")
        ordered = sorted(speedups)
        median = (ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2]))
        stats = dict(cases=n, arithmetic_mean=mean, geometric_mean=geo,
                     median=median, minimum=min(speedups), maximum=max(speedups),
                     total_ratio=sum(r["M1"] for r in ok_results) / sum(r["M5"] for r in ok_results))
    else:
        stats = {}

    # --- 汇总比对 --------------------------------------------------------
    # 关键：申报的聚合量必须在**同一子集**上重算，否则抽检若干用例时会误判。
    full_set = set(claimed)
    is_full = set(cases) == full_set
    claimed_sp = {c: float(claimed[c]["speedup"]) for c in cases}
    claimed_ms = {c: int(claimed[c]["makespan"]) for c in cases}
    claimed_base = {c: int(claimed[c]["baseline_makespan"]) for c in cases}
    cs_sp = list(claimed_sp.values())
    cn = len(cs_sp)
    claimed_stats = dict(
        cases=cn,
        arithmetic_mean=sum(cs_sp) / cn,
        geometric_mean=math.exp(sum(math.log(s) for s in cs_sp) / cn),
        median=(sorted(cs_sp)[cn // 2] if cn % 2
                else 0.5 * (sorted(cs_sp)[cn // 2 - 1] + sorted(cs_sp)[cn // 2])),
        minimum=min(cs_sp), maximum=max(cs_sp),
        total_ratio=sum(claimed_base.values()) / sum(claimed_ms.values()),
    )

    print()
    print("=" * 72)
    print("结构校验与官方重算")
    print("=" * 72)
    print(f"  参与复核用例数        : {len(results)}"
          + ("（全量 100 例）" if is_full else "（抽检子集）"))
    print(f"  官方结构校验通过      : {len(ok_results)}")
    print(f"  官方重算与申报不一致  : {len(failures)}")
    print(f"  单核基线 M1 不一致    : {len(base_mismatch)}")
    print(f"  加速比最大绝对偏差    : {max_diff:.3e}")
    print()
    print("聚合统计（本列由官方评估器重算，比对本列与申报列）")
    print("-" * 72)
    print(f"  {'指标':<22}{'官方重算':>14}{'申报(同子集)':>16}")
    pairs = [("算术平均加速比", "arithmetic_mean"),
             ("几何平均加速比", "geometric_mean"),
             ("中位数加速比", "median"),
             ("最小加速比", "minimum"),
             ("最大加速比", "maximum"),
             ("总周期压缩比", "total_ratio")]
    stat_fail = []
    for label, key in pairs:
        if key not in stats:
            continue
        got, want = stats[key], claimed_stats[key]
        close = abs(got - want) <= max(args.tolerance, 1e-9)
        if not close:
            stat_fail.append((label, want, got))
        flag = "" if close else "   <== 不一致"
        print(f"  {label:<22}{got:>14.6f}{want:>16.6f}{flag}")
    if is_full:
        claimed_summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        print()
        print("与 claimed_summary.json 的全量申报值对照")
        print("-" * 72)
        summary_pairs = [
            ("算术平均加速比", "arithmetic_mean", "arithmetic_mean_speedup"),
            ("几何平均加速比", "geometric_mean", "geometric_mean_speedup"),
            ("中位数加速比", "median", "median_speedup"),
            ("最小加速比", "minimum", "min_speedup"),
            ("最大加速比", "maximum", "max_speedup"),
            ("总周期压缩比", "total_ratio", "total_cycle_ratio"),
        ]
        for label, key, ckey in summary_pairs:
            if ckey not in claimed_summary or key not in stats:
                continue
            want = float(claimed_summary[ckey])
            if abs(stats[key] - want) > max(args.tolerance, 1e-9):
                stat_fail.append((label, want, stats[key]))
            print(f"  {label:<22}{stats[key]:>14.6f}{want:>16.6f}")

    if failures:
        print()
        print("不一致明细（前 20 条）")
        print("-" * 72)
        for case, msg in failures[:20]:
            print(f"  {case}: {msg}")
    if base_mismatch:
        print()
        print("单核基线差异明细（说明双方对 M1 的定义不同，需先统一口径）")
        print("-" * 72)
        for case, want, got in base_mismatch[:20]:
            print(f"  {case}: 申报 {want}  复算 {got}")

    passed = (not failures) and (not base_mismatch) and (not stat_fail)
    print()
    print("=" * 72)
    print("结论：" + ("PASS —— 官方评估器重算结果与申报数字完全一致"
                     if passed else "FAIL —— 存在上述不一致，请先核对口径或实现"))
    print("=" * 72)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"passed": passed, "stats": stats, "failures": failures,
             "base_mismatch": base_mismatch, "max_speedup_diff": max_diff,
             "per_case": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n逐例明细已写入 {args.json_out}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
