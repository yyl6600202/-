# -*- coding: utf-8 -*-
"""cwhole 对照实验: 整链(不切)多维 LPT 分核, groups 构图, 官方评估"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from graph_features import build_op_dag
from pipe_aware_search import (_detect_chains, _pipe_scale, _plan_from_groups,
                               NUM_CORES)
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
from stub_multicore_cut_and_schedule import validate_multicore_plan
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
config = read_evaluation_config(DATA / 'config.txt')
waits = read_scene_a_config(DATA / 'config.txt')

for case in ['case_051', 'case_044']:
    graph = json.loads((DATA / f'{case}.json').read_text(encoding='utf-8'))
    gd = build_op_dag(graph)
    op_by_id = gd['op_by_id']
    chains, leftovers = _detect_chains(gd)
    cw = sorted((sum(op_by_id[n]['cycles'] for n in c) for c in chains), reverse=True)
    print(f'{case}: {len(chains)} chains, leftover={len(leftovers)}, '
          f'chain cycles top8={cw[:8]}')
    # 整链 LPT (多维)
    _, scale = _pipe_scale(gd, NUM_CORES)
    loads = [defaultdict(int) for _ in range(NUM_CORES)]
    group_of, core_of = {}, {}
    items = [(sum(op_by_id[n]['cycles'] for n in c), i, c)
             for i, c in enumerate(chains)]
    items.sort(reverse=True)
    for _, gid, chain in items:
        vec = defaultdict(int)
        for n in chain:
            vec[op_by_id[n]['pipe']] += op_by_id[n]['cycles']
        best, best_key = 0, None
        for core in range(NUM_CORES):
            worst = max((loads[core].get(p, 0) + vec.get(p, 0)) / s
                        for p, s in scale.items())
            key = (worst, sum(loads[core].values()), core)
            if best_key is None or key < best_key:
                best, best_key = core, key
        for p, v in vec.items():
            loads[best][p] += v
        for n in chain:
            group_of[n] = gid
            core_of[n] = best
    for n in leftovers:
        gid = len(chains) + n % 1
        group_of[n] = gid
        core_of[n] = 0
    plan = _plan_from_groups(graph, group_of, core_of, NUM_CORES)
    validate_multicore_plan(graph, plan)
    nblocks = len(set(plan['node_to_subgraph'].values()))
    t0 = time.time()
    res = evaluate_scene_a(graph, plan, config['bandwidth'], config['capacity'],
                           waits['task_cross_core_wait_cycles'],
                           waits['task_same_core_wait_cycles'])
    dt = time.time() - t0
    ends = [max((op['end'] for op in core['ops']), default=0)
            for core in res['per_core_timeline']]
    mv = res['data_movement_bytes']
    print(f'  cwhole: blocks={nblocks} mk={res["makespan"]} '
          f'added_bytes={mv["added_copy_bytes"]} core_ends={ends} eval={dt:.1f}s',
          flush=True)
