# -*- coding: utf-8 -*-
"""v7 结构探针#2: 试点 case 的 DAG 层次/宽度/跨分支边画像(不跑评估)"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from graph_features import build_op_dag
from chain_cluster_partition import _execution_order, _roots_and_lineage

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
PILOT = ['case_016', 'case_024', 'case_051', 'case_044',
         'case_071', 'case_069', 'case_005', 'case_086']

for case in PILOT:
    graph = json.loads((DATA / f'{case}.json').read_text(encoding='utf-8'))
    gd = build_op_dag(graph)
    dag = gd['noncopy']
    preds, succs = dag['predecessors'], dag['successors']
    nodes = dag['nodes']
    op_by_id = gd['op_by_id']
    order = _execution_order(gd)
    # 层 = 边数最长路径
    level = {}
    for n in dag['topological_order']:
        level[n] = 0 if not preds[n] else 1 + max(level[p] for p in preds[n])
    depth = max(level.values()) + 1
    widths = Counter(level.values())
    topw = sorted(widths.values(), reverse=True)[:4]
    sources = [n for n in nodes if not preds[n]]
    sinks = [n for n in nodes if not succs[n]]
    # 跨 lineage 边
    lineage, roots, bc, bn = _roots_and_lineage(gd, order, 'bytes')
    cross = sum(1 for s, t in dag['edges'] if lineage[s] != lineage[t])
    # 巨分支内部: 它在各层的宽度
    giant = max(roots, key=lambda r: bc[r])
    gw = Counter(level[n] for n in bn[giant])
    gtop = sorted(gw.values(), reverse=True)[:4]
    # 源点出度 -> 列数的线索
    src_out = sorted((len(succs[s]) for s in sources), reverse=True)[:6]
    cyc = Counter(o['cycles'] for o in graph['ops'] if o['op'] not in ('COPY_IN','COPY_OUT'))
    print(f"{case}: n={len(nodes)} depth={depth} maxW={max(widths.values())} "
          f"avgW={len(nodes)/depth:.1f} topW={topw} src={len(sources)} sink={len(sinks)} "
          f"src_out={src_out} cross_ln_edges={cross}/{len(dag['edges'])} "
          f"giantW={gtop} op_cycles_top={cyc.most_common(3)}", flush=True)
