# -*- coding: utf-8 -*-
"""v7 试点前探针: 结构画像 + 单 case 评估耗时"""
import json, time, sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from graph_features import build_op_dag
from chain_cluster_partition import _execution_order, _roots_and_lineage
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
PILOT = ['case_016', 'case_024', 'case_051', 'case_044',
         'case_071', 'case_069', 'case_005', 'case_086']

config = read_evaluation_config(DATA / 'config.txt')
waits = read_scene_a_config(DATA / 'config.txt')
v6 = {r['case']: r for r in csv.DictReader(
    open(BASE / 'day5_results/paper_guided_v6_summary.csv', encoding='utf-8-sig'))}

for case in PILOT:
    graph = json.loads((DATA / f'{case}.json').read_text(encoding='utf-8'))
    gd = build_op_dag(graph)
    order = _execution_order(gd)
    lineage, roots, bc, bn = _roots_and_lineage(gd, order, 'bytes')
    total = sum(bc.values())
    top = sorted(bc.values(), reverse=True)[:6]
    # 单 pipe 占比
    from collections import Counter
    pc = Counter()
    for o in graph['ops']:
        pc[o['pipe']] += o['cycles']
    busy = max(pc, key=pc.get)
    plan = json.loads((BASE / 'day5_results/paper_guided_v6_plans' / f'{case}.json').read_text())
    nblocks = len(set(plan['node_to_subgraph'].values()))
    t0 = time.time()
    res = evaluate_scene_a(graph, plan, config['bandwidth'], config['capacity'],
                           waits['task_cross_core_wait_cycles'],
                           waits['task_same_core_wait_cycles'])
    dt = time.time() - t0
    print(f"{case}: ops={len(graph['ops'])} noncopy={len(gd['noncopy']['nodes'])} roots={len(roots)} "
          f"top_branch_share={[round(t/total,3) for t in top]} busy={busy}({pc[busy]/sum(pc.values()):.0%}) "
          f"v6_blocks={nblocks} v6_mk={res['makespan']} v6_summary_mk={v6[case]['problem1_makespan']} "
          f"eval={dt:.1f}s baseline={v6[case]['baseline_makespan']}", flush=True)
