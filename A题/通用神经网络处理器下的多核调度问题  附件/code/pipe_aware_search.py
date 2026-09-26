# -*- coding: utf-8 -*-
"""Pipe-aware candidate search for problem 1 (v7).

Why: v5/v6 assignment rules balance only *total* cycles per core.  The
problem-1 simulator runs each core's tasks sequentially while the four pipes
*inside* a task overlap, so the real bottleneck is the per-core **per-pipe**
load.  This module generates candidates that balance every pipe explicitly:

- ``plpt``   sticky per-op multi-dimensional LPT along the execution order.
             An op follows its dominant predecessor's core unless that would
             push the core's normalized pipe load above ``tau`` times the
             best alternative; then it moves (and pays a boundary).  Convex
             runs + smoothing keep the contracted task graph acyclic.
- ``clpt``   lineage-level multi-dimensional LPT (no branch splitting).
- ``csplit`` explicit chain-segment splitting for chain-structured graphs
             (many long independent chains, e.g. case_016/024/051/044).
             Segments are scattered with multi-dim LPT so the single busy
             pipe is spread over all cores in multiple waves.

Every candidate is validated with the official ``validate_multicore_plan``
and scored by the official ``evaluate_scene_a`` simulator.  A lightweight
analytic estimate ranks candidates so only the top-K are simulated.

Batch usage (from the ``code`` directory):
    python3 pipe_aware_search.py --case case_016 --case case_044 --eval-all
    python3 pipe_aware_search.py --workers 12 --topk 3          # full sweep
    python3 pipe_aware_search.py --merge                        # build v7 outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from graph_features import build_op_dag
from chain_cluster_partition import _execution_order, _roots_and_lineage
from paper_guided_search import _smooth_blocks, _plan_from_blocks, _edge_bytes
from stub_multicore_cut_and_schedule import (
    derive_multicore_plan, validate_multicore_plan,
    _build_op_adjacency, _contract_excluded_copy_nodes,
)
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config

BASE_DIR = CODE_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "day5_results"
V6_SUMMARY = RESULT_DIR / "paper_guided_v6_summary.csv"
V6_PLANS = RESULT_DIR / "paper_guided_v6_plans"
V7_PLANS = RESULT_DIR / "pipe_aware_v7_plans"
DELTA_CSV = RESULT_DIR / "pipe_aware_v7_delta.csv"
NUM_CORES = 5
PIPES = ("PIPE_M", "PIPE_V", "PIPE_MTE2", "PIPE_MTE3")


# ---------------------------------------------------------------------------
# candidate construction
# ---------------------------------------------------------------------------

def _pipe_scale(graph_data, num_cores):
    totals = {pipe: 0 for pipe in PIPES}
    for node in graph_data["noncopy"]["nodes"]:
        op = graph_data["op_by_id"][node]
        totals[op["pipe"]] = totals.get(op["pipe"], 0) + op["cycles"]
    scale = {pipe: max(1.0, value / num_cores) for pipe, value in totals.items()}
    return totals, scale


def _best_core(loads, scale, pipe, cycles, num_cores):
    """Core minimizing the worst normalized pipe load after placing the op."""
    best, best_key = 0, None
    for core in range(num_cores):
        worst = 0.0
        for p, s in scale.items():
            load = loads[core].get(p, 0) + (cycles if p == pipe else 0)
            worst = max(worst, load / s)
        key = (worst, sum(loads[core].values()), core)
        if best_key is None or key < best_key:
            best, best_key = core, key
    return best


def _sticky_lpt_labels(graph_data, order, num_cores, tau):
    dag = graph_data["noncopy"]
    op_by_id = graph_data["op_by_id"]
    traffic = _edge_bytes(graph_data)
    _, scale = _pipe_scale(graph_data, num_cores)
    loads = [defaultdict(int) for _ in range(num_cores)]
    labels = {}
    for node in order:
        op = op_by_id[node]
        pipe, cycles = op["pipe"], op["cycles"]
        preds = [p for p in dag["predecessors"][node] if p in labels]
        home = None
        if preds:
            score = defaultdict(lambda: [0, 0])
            for p in preds:
                score[labels[p]][0] += traffic.get((p, node), 0)
                score[labels[p]][1] += 1
            home = max(score.items(),
                       key=lambda kv: (kv[1][0], kv[1][1], -kv[0]))[0]
        target = _best_core(loads, scale, pipe, cycles, num_cores)
        if home is not None and home != target:
            def _worst(core):
                return max(
                    (loads[core].get(p, 0) + (cycles if p == pipe else 0)) / s
                    for p, s in scale.items())
            if _worst(home) <= tau * _worst(target):
                target = home
        labels[node] = target
        loads[target][pipe] += cycles
    return labels


def _lineage_lpt_labels(graph_data, order, num_cores, dominant="bytes"):
    op_by_id = graph_data["op_by_id"]
    lineage, roots, branch_cycles, branch_nodes = _roots_and_lineage(
        graph_data, order, dominant)
    _, scale = _pipe_scale(graph_data, num_cores)
    branch_vec = {}
    for root in roots:
        vec = defaultdict(int)
        for node in branch_nodes[root]:
            vec[op_by_id[node]["pipe"]] += op_by_id[node]["cycles"]
        branch_vec[root] = dict(vec)
    loads = [defaultdict(int) for _ in range(num_cores)]
    root_to_core = {}
    for root in sorted(roots, key=lambda r: (-branch_cycles[r], r)):
        vec = branch_vec[root]
        best, best_key = 0, None
        for core in range(num_cores):
            worst = 0.0
            for p, s in scale.items():
                worst = max(worst, (loads[core].get(p, 0) + vec.get(p, 0)) / s)
            key = (worst, sum(loads[core].values()), core)
            if best_key is None or key < best_key:
                best, best_key = core, key
        root_to_core[root] = best
        for p, v in vec.items():
            loads[best][p] += v
    return {node: root_to_core[lineage[node]] for node in order}


def _runs_to_plan(graph, order, labels, op_by_id, num_cores, max_blocks):
    identity = {node: node for node in order}
    blocks = _smooth_blocks(order, identity, labels, op_by_id, num_cores,
                            max_blocks=max_blocks)
    return _plan_from_blocks(blocks, num_cores)


def _detect_chains(graph_data):
    dag = graph_data["noncopy"]
    preds, succs = dag["predecessors"], dag["successors"]
    visited, chains = set(), []
    for start in dag["topological_order"]:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        cur = start
        while len(succs[cur]) == 1:
            nxt = next(iter(succs[cur]))
            if len(preds[nxt]) != 1 or nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            cur = nxt
        chains.append(chain)
    leftovers = [n for n in dag["nodes"] if n not in visited]
    return chains, leftovers


def _chain_split_candidate(graph_data, order, num_cores, parts):
    """Split long chains into ``parts`` equal-cycle segments; scatter by LPT.

    Returns (plan, num_groups) or None when the graph is not chain-structured.
    """
    op_by_id = graph_data["op_by_id"]
    chains, leftovers = _detect_chains(graph_data)
    covered = sum(len(c) for c in chains)
    total = len(graph_data["noncopy"]["nodes"])
    if len(chains) < 6 or covered < 0.7 * total:
        return None
    _, scale = _pipe_scale(graph_data, num_cores)
    segments = []  # (cycles, [ops])
    for idx, chain in enumerate(chains):
        work = [op_by_id[n]["cycles"] for n in chain]
        whole = sum(work)
        target = max(1.0, whole / parts)
        seg, acc = [], 0
        for node, cyc in zip(chain, work):
            if seg and acc >= target:
                segments.append((acc, seg))
                seg, acc = [], 0
            seg.append(node)
            acc += cyc
        if seg:
            segments.append((acc, seg))
    for node in leftovers:  # merge points etc. become singleton segments
        segments.append((op_by_id[node]["cycles"], [node]))
    loads = [defaultdict(int) for _ in range(num_cores)]
    group_of, core_of = {}, {}
    scatter = sorted(range(len(segments)), key=lambda i: (-segments[i][0], i))
    for gid in scatter:
        vec = defaultdict(int)
        for node in segments[gid][1]:
            vec[op_by_id[node]["pipe"]] += op_by_id[node]["cycles"]
        best, best_key = 0, None
        for core in range(num_cores):
            worst = 0.0
            for p, s in scale.items():
                worst = max(worst, (loads[core].get(p, 0) + vec.get(p, 0)) / s)
            key = (worst, sum(loads[core].values()), core)
            if best_key is None or key < best_key:
                best, best_key = core, key
        for p, v in vec.items():
            loads[best][p] += v
        for node in segments[gid][1]:
            group_of[node] = gid
            core_of[node] = best
    return group_of, core_of


def _plan_from_groups(graph, group_of, core_of, num_cores):
    """Legalize an arbitrary (group, core) assignment into a plan.

    Subgraph = group.  The contracted task graph is topologically sorted;
    per-core orders are projections of that global order, which satisfies the
    same-core dependency constraint by construction.  Cycles (possible when
    groups are not convex) are collapsed by SCC condensation.
    """
    eligible = sorted(group_of)
    _, full_succs = _build_op_adjacency(graph)
    _, contracted = _contract_excluded_copy_nodes(eligible, full_succs)
    groups = sorted({group_of[n] for n in eligible})
    index = {g: i for i, g in enumerate(groups)}
    sub_succ = defaultdict(set)
    for src in eligible:
        for dst in contracted.get(src, ()):
            a, b = index[group_of[src]], index[group_of[dst]]
            if a != b:
                sub_succ[a].add(b)

    def _topo(succ):
        indeg = Counter()
        nodes = set(succ) | {t for ts in succ.values() for t in ts} | set(
            range(len(groups)))
        for ts in succ.values():
            for t in ts:
                indeg[t] += 1
        ready = sorted(n for n in nodes if indeg[n] == 0)
        order = []
        while ready:
            u = ready.pop(0)
            order.append(u)
            for v in sorted(succ.get(u, ())):
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
                    ready.sort()
        return order

    topo = _topo(sub_succ)
    if len(topo) < len(groups):
        # SCC condensation: merge cyclic groups (keeps plan legal).
        sys.setrecursionlimit(100000)
        lowlink = {}
        index_of, on_stack, stack, order_seen = {}, {}, [], []
        counter = [0]
        scc_of = {}

        def strongconnect(start):
            work = [(start, iter(sorted(sub_succ.get(start, ()))))]
            index_of[start] = lowlink[start] = counter[0]
            counter[0] += 1
            stack.append(start)
            on_stack[start] = True
            while work:
                node, it = work[-1]
                advanced = False
                for nxt in it:
                    if nxt not in index_of:
                        index_of[nxt] = lowlink[nxt] = counter[0]
                        counter[0] += 1
                        stack.append(nxt)
                        on_stack[nxt] = True
                        work.append((nxt, iter(sorted(sub_succ.get(nxt, ())))))
                        advanced = True
                        break
                    elif on_stack.get(nxt):
                        lowlink[node] = min(lowlink[node], index_of[nxt])
                if advanced:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
                if lowlink[node] == index_of[node]:
                    scc_id = len(order_seen)
                    while True:
                        member = stack.pop()
                        on_stack[member] = False
                        scc_of[member] = scc_id
                        if member == node:
                            break
                    order_seen.append(scc_id)

        for g in range(len(groups)):
            if g not in index_of:
                strongconnect(g)
        # rebuild with merged groups
        new_group = {}
        core_votes = defaultdict(Counter)
        for node in eligible:
            g = scc_of[index[group_of[node]]]
            new_group[node] = g
            core_votes[g][core_of[node]] += 1
        new_core = {node: core_votes[new_group[node]].most_common(1)[0][0]
                    for node in eligible}
        return _plan_from_groups(graph, new_group, new_core, num_cores)

    position = {g: i for i, g in enumerate(topo)}
    mapping, core_schedules = {}, [[] for _ in range(num_cores)]
    for node in eligible:
        mapping[str(node)] = index[group_of[node]]
    core_of_group = {}
    for node in eligible:
        core_of_group[index[group_of[node]]] = core_of[node]
    for g in topo:
        core_schedules[core_of_group[g]].append(g)
    return {"node_to_subgraph": mapping, "core_schedules": core_schedules}


# ---------------------------------------------------------------------------
# analytic pre-screen
# ---------------------------------------------------------------------------

def _estimate_makespan(graph, plan, num_cores, bandwidth, same_wait, cross_wait):
    view = derive_multicore_plan(graph, plan)
    op_by_id = {op["id"]: op for op in graph["ops"]}
    tensor_by_id = {t["id"]: t for t in graph["tensors"]}
    producers, consumers, _ = (defaultdict(set), defaultdict(set), None)
    op_ids = set(op_by_id)
    for edge in graph["edges"]:
        s, t = edge["source"], edge["target"]
        if s in op_ids and t not in op_ids:
            producers[t].add(s)
        elif s not in op_ids and t in op_ids:
            consumers[s].add(t)
    mapping = view["mapping"]
    task_pipe = {g: defaultdict(int) for g in view["subgraph_ids"]}
    for node, g in mapping.items():
        op = op_by_id[node]
        task_pipe[g][op["pipe"]] += op["cycles"]
    for tid, prods in producers.items():
        size = tensor_by_id[tid]["size"]
        cost = max(1, math.ceil(size / bandwidth))
        prod_tasks = {mapping[p] for p in prods if p in mapping}
        cons_tasks = {mapping[c] for c in consumers.get(tid, ()) if c in mapping}
        if prod_tasks and cons_tasks - prod_tasks:
            for g in prod_tasks:
                task_pipe[g]["PIPE_MTE3"] += cost
            for g in cons_tasks - prod_tasks:
                task_pipe[g]["PIPE_MTE2"] += cost
        elif not prod_tasks and cons_tasks:
            for g in cons_tasks:
                task_pipe[g]["PIPE_MTE2"] += cost
    task_dur = {g: max(task_pipe[g].values()) if task_pipe[g] else 0
                for g in view["subgraph_ids"]}
    core_time = [0] * num_cores
    for core, order in enumerate(view["core_orders"].values()):
        core_time[core] = sum(task_dur[g] for g in order) + same_wait * len(order)
    # critical chain through the task DAG
    preds = view["subgraph_preds"]
    memo = {}

    def _cp(g):
        if g not in memo:
            best = 0
            for p in preds[g]:
                wait = cross_wait if view["core_by_subgraph"][p] != \
                    view["core_by_subgraph"][g] else same_wait
                best = max(best, _cp(p) + wait)
            memo[g] = best + task_dur[g]
        return memo[g]

    chain = max((_cp(g) for g in view["subgraph_ids"]), default=0)
    return max(max(core_time), chain)


# ---------------------------------------------------------------------------
# candidate generation + evaluation
# ---------------------------------------------------------------------------

def _assign_groups_lpt(graph_data, group_of, num_cores):
    """Multi-dimensional (per-pipe) LPT assignment of groups to cores."""
    op_by_id = graph_data["op_by_id"]
    _, scale = _pipe_scale(graph_data, num_cores)
    group_vec = defaultdict(lambda: defaultdict(int))
    for node, g in group_of.items():
        op = op_by_id[node]
        group_vec[g][op["pipe"]] += op["cycles"]
    loads = [defaultdict(int) for _ in range(num_cores)]
    core_of = {}
    order = sorted(group_vec, key=lambda g: (-sum(group_vec[g].values()), g))
    for g in order:
        vec = group_vec[g]
        best, best_key = 0, None
        for core in range(num_cores):
            worst = max((loads[core].get(p, 0) + vec.get(p, 0)) / s
                        for p, s in scale.items())
            key = (worst, sum(loads[core].values()), core)
            if best_key is None or key < best_key:
                best, best_key = core, key
        for p, v in vec.items():
            loads[best][p] += v
        core_of[g] = best
    return {node: core_of[g] for node, g in group_of.items()}


def _group_dag(graph_data, group_of):
    """Group-level DAG over the contracted (copy-aware) op adjacency."""
    dag = graph_data["noncopy"]
    preds = {g: set() for g in set(group_of.values())}
    for s, t in dag["edges"]:
        a, b = group_of[s], group_of[t]
        if a != b:
            preds[b].add(a)
    return preds


def _assign_groups_heft(graph_data, group_of, num_cores,
                        cross_wait=1000, same_wait=100):
    """HEFT list scheduling of groups onto cores.

    Mirrors the simulator's cost model: tasks on one core run sequentially
    (``same_wait`` gap), a task starts only after all predecessor tasks are
    done (``cross_wait`` extra when the predecessor ran on another core).
    Groups are processed by upward rank and placed on the core with the
    earliest estimated finish time, which keeps chains on one core when
    scattering them would only add waits.
    """
    op_by_id = graph_data["op_by_id"]
    preds = _group_dag(graph_data, group_of)
    succs = {g: set() for g in preds}
    for g, ps in preds.items():
        for p in ps:
            succs[p].add(g)
    gvec = {g: defaultdict(int) for g in preds}
    for node, g in group_of.items():
        op = op_by_id[node]
        gvec[g][op["pipe"]] += op["cycles"]
    dur = {g: max(gvec[g].values()) if gvec[g] else 0 for g in preds}
    # iterative upward rank over a reverse topological order (deep DAGs)
    indeg = {g: len(preds[g]) for g in preds}
    ready = sorted(g for g in preds if indeg[g] == 0)
    topo = []
    while ready:
        u = ready.pop()
        topo.append(u)
        for v in sorted(succs[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
                ready.sort()
    rank = {}
    for g in reversed(topo):
        rank[g] = dur[g] + max(
            (cross_wait + rank.get(s, dur[s]) for s in succs[g]), default=0)
    for g in preds:  # cyclic leftovers (SCC-merged later in the plan builder)
        if g not in rank:
            rank[g] = dur[g]
    order = sorted(preds, key=lambda g: (-rank[g], g))
    core_avail = [0] * num_cores
    end_of = {}
    core_of = {}
    for g in order:
        best, best_key = 0, None
        for core in range(num_cores):
            start = core_avail[core]
            for p in preds[g]:
                if p not in core_of:  # cyclic predecessor, merged later
                    continue
                wait = same_wait if core_of[p] == core else cross_wait
                start = max(start, end_of[p] + wait)
            finish = start + max(1, dur[g])
            key = (finish, start, core)
            if best_key is None or key < best_key:
                best, best_key = core, key
        core_of[g] = best
        start = max(core_avail[best], max(
            (end_of[p] + (same_wait if core_of[p] == best else cross_wait)
             for p in preds[g] if p in core_of), default=0))
        end_of[g] = start + max(1, dur[g])
        core_avail[best] = end_of[g] + same_wait
    return {node: core_of[g] for node, g in group_of.items()}


def _kruskal_groups(graph_data, num_cores, beta):
    """Union ops along heaviest-byte edges while per-pipe loads stay under cap.

    Produces medium-size connected groups whose cut edges are the cheap ones
    (small tensors).  ``beta`` caps each group's per-pipe cycles at
    beta * W_pipe / num_cores so groups stay balanceable.
    """
    dag = graph_data["noncopy"]
    op_by_id = graph_data["op_by_id"]
    traffic = _edge_bytes(graph_data)
    totals, _ = _pipe_scale(graph_data, num_cores)
    cap = {p: beta * v / num_cores for p, v in totals.items()}
    parent = {n: n for n in dag["nodes"]}
    vec = {n: defaultdict(int, {op_by_id[n]["pipe"]: op_by_id[n]["cycles"]})
           for n in dag["nodes"]}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    weighted = sorted(
        ((traffic.get((s, t), 0), op_by_id[s]["cycles"] + op_by_id[t]["cycles"], s, t)
         for s, t in dag["edges"]), reverse=True)
    for _, __, s, t in weighted:
        ra, rb = find(s), find(t)
        if ra == rb:
            continue
        merged = defaultdict(int)
        for p in PIPES:
            merged[p] = vec[ra].get(p, 0) + vec[rb].get(p, 0)
        if any(merged[p] > cap[p] for p in PIPES if cap[p] > 0):
            continue
        if sum(merged.values()) > beta * sum(totals.values()) / num_cores:
            continue
        big, small = (ra, rb) if sum(vec[ra].values()) >= sum(vec[rb].values()) else (rb, ra)
        parent[small] = big
        vec[big] = merged
    group_of = {}
    remap = {}
    for n in dag["nodes"]:
        root = find(n)
        if root not in remap:
            remap[root] = len(remap)
        group_of[n] = remap[root]
    return group_of


def _components_groups(graph_data):
    dag = graph_data["noncopy"]
    parent = {n: n for n in dag["nodes"]}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for s, t in dag["edges"]:
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    group_of, remap = {}, {}
    for n in dag["nodes"]:
        root = find(n)
        if root not in remap:
            remap[root] = len(remap)
        group_of[n] = remap[root]
    return group_of


def _chain_groups(graph_data):
    chains, leftovers = _detect_chains(graph_data)
    group_of = {}
    for gid, chain in enumerate(chains):
        for n in chain:
            group_of[n] = gid
    for n in leftovers:
        group_of[n] = len(chains) + n  # singleton junction groups
    return group_of


def _merge_groups_acyclic(graph_data, num_cores, beta, shared_min_size=4096,
                          shared_max_fanout=16):
    """Greedily merge micro-chain groups along heavy-byte links.

    Starts from the micro-chain partition (acyclic group DAG) and merges
    group pairs weighted by direct-edge bytes plus shared-big-tensor affinity,
    skipping any merge that would exceed the per-pipe cap or create a cycle
    (tracked with incrementally maintained descendant bitmasks).  The result
    is a medium-grain, tensor-cohesive, acyclic grouping.
    """
    op_by_id = graph_data["op_by_id"]
    dag = graph_data["noncopy"]
    group_of = _chain_groups(graph_data)
    groups = sorted(set(group_of.values()))
    remap = {g: i for i, g in enumerate(groups)}
    group_of = {n: remap[g] for n, g in group_of.items()}
    k = len(groups)
    traffic = _edge_bytes(graph_data)
    pair_w = defaultdict(int)
    for (s, t), b in traffic.items():
        a, c = group_of[s], group_of[t]
        if a != c:
            pair_w[(a, c)] += b
    # shared big-tensor affinity between consumer groups
    noncopy = set(dag["nodes"])
    for tid, cons in graph_data["consumers"].items():
        tensor = graph_data["tensor_by_id"][tid]
        if tensor["size"] < shared_min_size:
            continue
        cs = sorted({group_of[c] for c in cons if c in noncopy})
        if len(cs) < 2 or len(cs) > shared_max_fanout:
            continue
        share = tensor["size"] // (len(cs) - 1)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, c = cs[i], cs[j]
                pair_w[(a, c)] += share
    succ = [set() for _ in range(k)]
    pred = [set() for _ in range(k)]
    for (s, t) in traffic:  # reachability uses REAL directed edges only
        a, c = group_of[s], group_of[t]
        if a != c:
            succ[a].add(c)
            pred[c].add(a)
    totals, _ = _pipe_scale(graph_data, num_cores)
    cap_total = beta * sum(totals.values()) / num_cores
    gvec = [defaultdict(int) for _ in range(k)]
    for n, g in group_of.items():
        gvec[g][op_by_id[n]["pipe"]] += op_by_id[n]["cycles"]
    alive = [True] * k
    gparent = list(range(k))

    def find(x):
        root = x
        while gparent[root] != root:
            root = gparent[root]
        while gparent[x] != root:
            gparent[x], x = root, gparent[x]
        return root

    # initial descendant bitmasks via one topological pass
    indeg = [0] * k
    for a in range(k):
        for c in succ[a]:
            indeg[c] += 1
    queue = [g for g in range(k) if indeg[g] == 0]
    topo = []
    while queue:
        u = queue.pop()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    mask = [0] * k
    for u in reversed(topo):
        m = 1 << u
        for v in succ[u]:
            m |= mask[v]
        mask[u] = m
    if len(topo) < k:
        return group_of  # unexpected cycle in micro-chain DAG; skip merging

    for (a, c), w in sorted(pair_w.items(), key=lambda kv: (-kv[1], kv[0])):
        ra, rc = find(a), find(c)
        if ra == rc:
            continue
        merged_total = sum(gvec[ra].values()) + sum(gvec[rc].values())
        if merged_total > cap_total:
            continue
        if (mask[ra] >> rc) & 1 or (mask[rc] >> ra) & 1:
            continue  # would create a cycle
        big, small = (ra, rc) if sum(gvec[ra].values()) >= sum(gvec[rc].values()) else (rc, ra)
        alive[small] = False
        gparent[small] = big
        for p in PIPES:
            gvec[big][p] = gvec[big].get(p, 0) + gvec[small].get(p, 0)
        absorb = mask[small]
        # rewire edges around the absorbed group
        out_nb = (succ[small] | succ[big]) - {big, small}
        in_nb = (pred[small] | pred[big]) - {big, small}
        for u in succ[small]:
            if u in (big, small):
                continue
            pred[u].discard(small)
            pred[u].add(big)
        for u in pred[small]:
            if u in (big, small):
                continue
            succ[u].discard(small)
            succ[u].add(big)
        succ[big], pred[big] = out_nb, in_nb
        succ[small].clear()
        pred[small].clear()
        mask[big] |= absorb
        # ancestors of the merged group gain small's descendant bits
        stack = list(in_nb)
        seen = set(stack)
        while stack:
            u = stack.pop()
            if not alive[u]:
                continue
            mask[u] |= absorb & ~(1 << small)
            for v in pred[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    remap2 = {}
    final = {}
    for n in sorted(group_of):
        g = find(group_of[n])
        if g not in remap2:
            remap2[g] = len(remap2)
        final[n] = remap2[g]
    return final


def _stripe_groups(graph_data, order, num_cores, num_stripes, weight="busy"):
    """Cut the execution order into ``num_stripes`` convex intervals.

    Convex intervals of the physical order keep the wavefront aligned: every
    core advances through the same phases of the graph, so cross-core waits
    hide inside each wave.  ``weight='busy'`` equalises the busiest-pipe
    cycles per stripe, ``'total'`` equalises total cycles (v6-style).
    """
    op_by_id = graph_data["op_by_id"]
    totals, _ = _pipe_scale(graph_data, num_cores)
    busy = max(totals, key=totals.get) if any(totals.values()) else None

    def w(node):
        if weight == "busy" and busy:
            return op_by_id[node]["cycles"] if op_by_id[node]["pipe"] == busy else 0
        return op_by_id[node]["cycles"]

    whole = sum(w(n) for n in order) or 1
    target = whole / num_stripes
    group_of, gid, acc = {}, 0, 0
    for node in order:
        group_of[node] = gid
        acc += w(node)
        if acc >= target and gid < num_stripes - 1:
            gid += 1
            acc = 0
    return group_of


def _assign_stripes_rr(graph_data, group_of, num_cores):
    return {node: g % num_cores for node, g in group_of.items()}


def generate_pipe_candidates(graph, num_cores=NUM_CORES):
    graph_data = build_op_dag(graph)
    order = _execution_order(graph_data)
    candidates = []
    lineage, _, _, _ = _roots_and_lineage(graph_data, order, "bytes")
    groupings = [("clpt_g", {n: lineage[n] for n in order})]
    comp = _components_groups(graph_data)
    if len(set(comp.values())) >= 3:
        groupings.append(("ccomp", comp))
    groupings.append(("cwhole", _chain_groups(graph_data)))
    for beta in (0.1, 0.2, 0.4, 0.7):
        groupings.append((f"ck{int(beta * 10):02d}",
                          _kruskal_groups(graph_data, num_cores, beta)))
    for beta in (0.2, 0.4):
        groupings.append((f"mk{int(beta * 10):02d}",
                          _merge_groups_acyclic(graph_data, num_cores, beta)))
    for stripes in (10, 16, 25, 40):
        groupings.append((f"st{stripes}", _stripe_groups(
            graph_data, order, num_cores, stripes, "busy")))
    for rule, grp in groupings:
        assigners = [("", _assign_groups_lpt), ("_h", _assign_groups_heft)]
        if rule.startswith("st"):
            assigners.append(("_rr", _assign_stripes_rr))
        for mode, assign in assigners:
            plan = _plan_from_groups(
                graph, grp, assign(graph_data, grp, num_cores), num_cores)
            candidates.append((f"{rule}{mode}", plan))

    results, seen = [], set()
    for rule, plan in candidates:
        fingerprint = json.dumps(plan, sort_keys=True, separators=(",", ":"))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        valid, error = True, ""
        try:
            validate_multicore_plan(graph, plan)
        except Exception as exc:  # rejected candidates are reported, not fatal
            valid, error = False, f"{type(exc).__name__}: {exc}"
        results.append({"rule": rule, "plan": plan, "valid": valid,
                        "validation_error": error})
    return results


def _eval_worker(payload):
    case = payload["case"]
    graph = json.loads((DATA_DIR / f"{case}.json").read_text(encoding="utf-8"))
    config = read_evaluation_config(DATA_DIR / "config.txt")
    waits = read_scene_a_config(DATA_DIR / "config.txt")
    row = {"case": case, "v6_makespan": payload["incumbent"],
           "baseline": payload["baseline"], "evaluated": 0, "rejected": 0,
           "best_makespan": None, "best_rule": "", "improved": False,
           "error": "", "candidates": []}
    try:
        candidates = generate_pipe_candidates(graph, NUM_CORES)
    except Exception as exc:
        row["error"] = f"generate: {type(exc).__name__}: {exc}"
        return row
    scored = []
    for cand in candidates:
        if not cand["valid"]:
            row["rejected"] += 1
            row["candidates"].append(
                {"rule": cand["rule"], "error": cand["validation_error"]})
            continue
        try:
            est = _estimate_makespan(
                graph, cand["plan"], NUM_CORES, config["bandwidth"],
                waits["task_same_core_wait_cycles"],
                waits["task_cross_core_wait_cycles"])
        except Exception as exc:
            row["rejected"] += 1
            row["candidates"].append(
                {"rule": cand["rule"], "error": f"est: {type(exc).__name__}: {exc}"})
            continue
        scored.append((est, cand))
    scored.sort(key=lambda item: item[0])
    topk = payload["topk"] if payload["topk"] > 0 else len(scored)
    best = None
    for est, cand in scored[:topk]:
        try:
            result = evaluate_scene_a(
                graph, cand["plan"], config["bandwidth"], config["capacity"],
                waits["task_cross_core_wait_cycles"],
                waits["task_same_core_wait_cycles"])
            mk = result["makespan"]
        except Exception as exc:
            row["candidates"].append(
                {"rule": cand["rule"], "est": est,
                 "error": f"eval: {type(exc).__name__}: {exc}"})
            continue
        row["evaluated"] += 1
        row["candidates"].append(
            {"rule": cand["rule"], "est": est, "makespan": mk})
        if best is None or mk < best[1]:
            best = (cand, mk)
    if best is not None:
        row["best_makespan"] = best[1]
        row["best_rule"] = best[0]["rule"]
        if best[1] < payload["incumbent"]:
            row["improved"] = True
            V7_PLANS.mkdir(parents=True, exist_ok=True)
            (V7_PLANS / f"{case}.json").write_text(
                json.dumps(best[0]["plan"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
    return row


def run_sweep(cases, topk, workers):
    with V6_SUMMARY.open(newline="", encoding="utf-8-sig") as stream:
        v6 = {row["case"]: row for row in csv.DictReader(stream)}
    payloads = [{
        "case": case,
        "incumbent": int(float(v6[case]["problem1_makespan"])),
        "baseline": int(float(v6[case]["baseline_makespan"])),
        "topk": topk,
    } for case in cases]
    # big graphs first so they do not become stragglers
    def _size(payload):
        return len(json.loads((DATA_DIR / f"{payload['case']}.json")
                              .read_text(encoding="utf-8"))["ops"])
    for payload in payloads:
        if payload["topk"] <= 0:
            continue
        n = _size(payload)
        payload["topk"] = 99 if n <= 6000 else (5 if n <= 15000 else 4)
    payloads.sort(key=_size, reverse=True)
    results = []
    if workers > 1 and len(payloads) > 1:
        with Pool(workers) as pool:
            for row in pool.imap_unordered(_eval_worker, payloads):
                results.append(row)
                print(json.dumps({k: row[k] for k in (
                    "case", "best_rule", "best_makespan", "improved",
                    "evaluated", "error")}, ensure_ascii=False), flush=True)
    else:
        for payload in payloads:
            row = _eval_worker(payload)
            results.append(row)
            print(json.dumps({k: row[k] for k in (
                "case", "best_rule", "best_makespan", "improved",
                "evaluated", "error")}, ensure_ascii=False), flush=True)
    return results


# ---------------------------------------------------------------------------
# merge: build v7 summary / stats / report / plans
# ---------------------------------------------------------------------------

def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def merge():
    base_rows = _read_csv(V6_SUMMARY)
    delta_rows = {}
    for delta_path in sorted(RESULT_DIR.glob("pipe_aware_v7*delta*.csv")):
        for row in _read_csv(delta_path):
            case = row["case"]
            if row.get("improved", "").lower() != "true":
                delta_rows.setdefault(case, row)
                continue
            prev = delta_rows.get(case)
            if (prev is None
                    or prev.get("improved", "").lower() != "true"
                    or float(row["best_makespan"]) < float(prev["best_makespan"])):
                delta_rows[case] = row
    improved = {case: row for case, row in delta_rows.items()
                if row.get("improved", "").lower() == "true"}
    # reconcile: per-case best makespan across deltas must match the plan file;
    # re-evaluate to be certain the stored plan reproduces the recorded number
    config = read_evaluation_config(DATA_DIR / "config.txt")
    waits = read_scene_a_config(DATA_DIR / "config.txt")
    for case, row in sorted(improved.items()):
        plan_path = V7_PLANS / f"{case}.json"
        if not plan_path.exists():
            continue
        graph = json.loads((DATA_DIR / f"{case}.json").read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        mk = evaluate_scene_a(graph, plan, config["bandwidth"],
                              config["capacity"],
                              waits["task_cross_core_wait_cycles"],
                              waits["task_same_core_wait_cycles"])["makespan"]
        row["best_makespan"] = str(mk)
    V7_PLANS.mkdir(parents=True, exist_ok=True)
    for row in base_rows:
        case = row["case"]
        target = V7_PLANS / f"{case}.json"
        if case not in improved and not target.exists():
            shutil.copy2(V6_PLANS / f"{case}.json", target)

    rows = []
    for row in base_rows:
        case = row["case"]
        row = dict(row)
        update = improved.get(case)
        row["pipe_aware_rule"] = update["best_rule"] if update else ""
        row["pipe_aware_base_makespan"] = update["v6_makespan"] if update else ""
        row["pipe_aware_candidate_makespan"] = update["best_makespan"] if update else ""
        row["pipe_aware_improved"] = "1" if update else "0"
        if update:
            row["winner"] = update["best_rule"]
            row["problem1_makespan"] = update["best_makespan"]
            row["speedup"] = repr(int(float(update["baseline"])) /
                                  int(float(update["best_makespan"])))
        row["incumbent_plan"] = f"day5_results\\pipe_aware_v7_plans\\{case}.json"
        rows.append(row)

    summary_path = RESULT_DIR / "pipe_aware_v7_summary.csv"
    fields = list(rows[0])
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["case"]))

    # official structural validation of every final plan
    validation_errors = []
    for row in rows:
        case = row["case"]
        graph = json.loads((DATA_DIR / f"{case}.json").read_text(encoding="utf-8"))
        plan = json.loads((V7_PLANS / f"{case}.json").read_text(encoding="utf-8"))
        try:
            validate_multicore_plan(graph, plan)
        except Exception as exc:
            validation_errors.append(f"{case}: {type(exc).__name__}: {exc}")

    speeds = [float(row["speedup"]) for row in rows]
    baseline_total = sum(int(float(row["baseline_makespan"])) for row in rows)
    schedule_total = sum(int(float(row["problem1_makespan"])) for row in rows)
    from statistics import median
    stats = {
        "num_cores": NUM_CORES,
        "num_cases": len(rows),
        "errors": 0,
        "plan_validation_errors": len(validation_errors),
        "validation_error_detail": validation_errors,
        "arithmetic_mean_speedup": sum(speeds) / len(speeds),
        "geometric_mean_speedup": math.exp(
            sum(math.log(v) for v in speeds) / len(speeds)),
        "median_speedup": median(speeds),
        "p10_speedup": sorted(speeds)[max(0, math.ceil(0.1 * len(speeds)) - 1)],
        "minimum_speedup": min(speeds),
        "maximum_speedup": max(speeds),
        "faster_cases": sum(int(float(r["problem1_makespan"])) <
                            int(float(r["baseline_makespan"])) for r in rows),
        "equal_cases": sum(int(float(r["problem1_makespan"])) ==
                           int(float(r["baseline_makespan"])) for r in rows),
        "slower_cases": sum(int(float(r["problem1_makespan"])) >
                            int(float(r["baseline_makespan"])) for r in rows),
        "baseline_total_cycles": baseline_total,
        "problem1_total_cycles": schedule_total,
        "aggregate_cycle_ratio": baseline_total / schedule_total,
        "pipe_aware_candidates": len(delta_rows),
        "pipe_aware_accepted_cases": sorted(improved),
        "pipe_aware_accepted_count": len(improved),
    }
    (RESULT_DIR / "pipe_aware_v7_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 问题一：pipe 感知 5 核调度改进（v7）", "",
        "本轮以 v6 的 100 个官方合法方案为基线，新增 pipe 感知候选生成器"
        "（`pipe_aware_search.py`）：粘性 per-op 多维 LPT（plpt）、lineage 级"
        "多维 LPT（clpt）、链切段散布（csplit）。核心观察：模拟器让同一核内的"
        "多条 pipe 并行、同一核的 Task 串行，因此负载均衡必须按 pipe 分别进行，"
        "而不是只均衡总 cycles。每个候选仍由官方事件模拟器评分，只有 Makespan "
        "严格下降才替换基线。", "",
        f"- 算术平均加速比：**{stats['arithmetic_mean_speedup']:.6f}×**",
        f"- 几何平均加速比：{stats['geometric_mean_speedup']:.6f}×",
        f"- 中位数加速比：{stats['median_speedup']:.6f}×",
        f"- p10/min/max：{stats['p10_speedup']:.4f} / "
        f"{stats['minimum_speedup']:.4f} / {stats['maximum_speedup']:.4f}",
        f"- 总周期比：{stats['aggregate_cycle_ratio']:.6f}×",
        f"- 100 个 case，官方方案校验错误：{stats['plan_validation_errors']}",
        f"- pipe 感知候选覆盖：{stats['pipe_aware_candidates']} 个 case；"
        f"接受改进：{stats['pipe_aware_accepted_count']} 个", "",
        "## 接受的改进", "",
        "| case | v6 Makespan | v7 Makespan | 新加速比 | 规则 |",
        "|---|---:|---:|---:|---|",
    ]
    for case in sorted(improved):
        row = improved[case]
        lines.append(
            f"| {case} | {row['v6_makespan']} | {row['best_makespan']} | "
            f"{int(float(row['baseline'])) / int(float(row['best_makespan'])):.6f} "
            f"| `{row['best_rule']}` |")
    lines += ["",
              "其余 case 保留 v6 方案。完整汇总见 `pipe_aware_v7_summary.csv`，"
              "最终方案见 `pipe_aware_v7_plans/`。"]
    (RESULT_DIR / "pipe_aware_v7_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--topk", type=int, default=0,
                        help="officially evaluate only the K best-ranked "
                             "candidates per case (0 = all)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--delta-out", type=Path, default=DELTA_CSV)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args(argv)
    if args.merge:
        print(json.dumps(merge(), ensure_ascii=False, indent=2))
        return
    if not args.cases:
        with V6_SUMMARY.open(newline="", encoding="utf-8-sig") as stream:
            args.cases = sorted(row["case"] for row in csv.DictReader(stream))
    results = run_sweep(sorted(set(args.cases)), args.topk, args.workers)
    fields = ["case", "v6_makespan", "baseline", "best_makespan", "best_rule",
              "improved", "evaluated", "rejected", "error", "candidates"]
    with args.delta_out.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: r["case"]):
            row = dict(row)
            row["candidates"] = json.dumps(row["candidates"], ensure_ascii=False)
            writer.writerow(row)
    print(f"delta written: {args.delta_out}")


if __name__ == "__main__":
    main()
