# -*- coding: utf-8 -*-
"""
问题一(5核) 乐观上界诊断分析
==============================
对每个 case:
  W   = 所有 op cycles 之和 (COPY_IN/COPY_OUT cycles=0, 传输耗时未计入 -> W 偏乐观)
  CP  = Op 级 DAG 最长路径 (只累加 op cycles, 忽略通信等待)
  T_lb_naive = max(W/5, CP)            -- 题目设定口径
  T_lb_pipe  = max(max_p W_p/5, CP)    -- pipe 感知口径:
       每核内部 M/V/MTE2/MTE3 多 pipe 并行, 每种 pipe 全机共 5 个实例,
       因此有效瓶颈是"最忙 pipe 的总周期 / 5", 而不是 W/5。
       (实测存在 makespan < W/5 的 case, 说明 naive 口径会被现状违反)
  S_ub = baseline_makespan / T_lb
输出:
  day5_results/upper_bound_analysis.csv  +  终端汇总 a)~e)
运行: wsl -d Ubuntu-24.04 -- python3 <本文件>   (仅标准库)
"""
import os, json, csv, glob, sys
from collections import deque, defaultdict

N_CORES = 5
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                      # .../附件
DATA_DIR = os.path.join(BASE, 'data')
SUMMARY_CSV = os.path.join(BASE, 'day5_results', 'paper_guided_v6_summary.csv')
PROFILE_CSV = os.path.join(BASE, 'case_profile.csv')
OUT_CSV = os.path.join(BASE, 'day5_results', 'upper_bound_analysis.csv')
PIPES = ['PIPE_M', 'PIPE_V', 'PIPE_MTE2', 'PIPE_MTE3']


def analyze_case(path):
    """返回 dict(n_ops, W, CP, pipe_cycles) ; 有环返回 None 并打印警告"""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    ops = d['ops']
    op_ids = set(o['id'] for o in ops)
    cyc = {o['id']: o.get('cycles', 0) for o in ops}
    pipe_of = {o['id']: o.get('pipe', '?') for o in ops}

    producer = defaultdict(list)   # tensor_id -> [op_id,...]
    consumer_edges = []            # (tensor_id, op_id)
    direct_edges = []              # (op_id, op_id)
    for e in d['edges']:
        s, t = e['source'], e['target']
        s_is_op, t_is_op = s in op_ids, t in op_ids
        if s_is_op and t_is_op:
            direct_edges.append((s, t))
        elif s_is_op:
            producer[s].append  # noop
            producer[t].append(s)
        elif t_is_op:
            consumer_edges.append((s, t))
        # tensor->tensor 边忽略(理论上不该出现)

    preds = defaultdict(set)
    for tid, oid in consumer_edges:
        for p in producer.get(tid, ()):
            if p != oid:
                preds[oid].add(p)
    for s, t in direct_edges:
        if s != t:
            preds[t].add(s)

    n = len(ops)
    # Kahn 拓扑 (注意: 只用 op 节点)
    children = defaultdict(list)
    indeg = {o['id']: 0 for o in ops}
    for v, ps in preds.items():
        indeg[v] = len(ps)
        for p in ps:
            children[p].append(v)
    q = deque([i for i in indeg if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    name = os.path.basename(path)
    if len(topo) < n:
        print(f"[WARN] {name}: 检测到依赖环或悬空边, 拓扑仅覆盖 {len(topo)}/{n} 个 op, 跳过该 case")
        return None

    dist = {}
    cp = 0
    for u in topo:
        best = 0
        for p in preds.get(u, ()):
            if dist[p] > best:
                best = dist[p]
        dist[u] = best + cyc[u]
        if dist[u] > cp:
            cp = dist[u]

    pipe_cycles = defaultdict(int)
    for o in ops:
        pipe_cycles[o.get('pipe', '?')] += o.get('cycles', 0)
    return {'n_ops': n, 'W': sum(cyc.values()), 'CP': cp, 'pipe_cycles': dict(pipe_cycles)}


def main():
    # 当前最好成绩
    cur = {}
    with open(SUMMARY_CSV, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            cur[r['case']] = {
                'baseline': float(r['baseline_makespan']),
                'p1': float(r['problem1_makespan']),
                'speedup': float(r['speedup']),
            }
    # case 类型
    typ = {}
    with open(PROFILE_CSV, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            typ[r['case_id']] = r['graph_class']
    # profile 里的 critical_path_cycles 用于交叉验证
    prof_cp = {}
    with open(PROFILE_CSV, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                prof_cp[r['case_id']] = float(r['critical_path_cycles'])
            except (KeyError, ValueError):
                pass

    rows, skipped = [], []
    cp_diff_max = 0.0
    for path in sorted(glob.glob(os.path.join(DATA_DIR, 'case_???.json'))):
        case = os.path.splitext(os.path.basename(path))[0]
        if case not in cur:
            print(f"[WARN] {case}: summary 中无记录, 跳过")
            continue
        a = analyze_case(path)
        if a is None:
            skipped.append(case)
            continue
        W, CP = a['W'], a['CP']
        if case in prof_cp and prof_cp[case] > 0:
            cp_diff_max = max(cp_diff_max, abs(CP - prof_cp[case]) / prof_cp[case])
        busy_pipe = max(a['pipe_cycles'].items(), key=lambda kv: kv[1]) if a['pipe_cycles'] else ('?', 0)
        t_lb_naive = max(W / N_CORES, CP)
        t_lb_pipe = max(busy_pipe[1] / N_CORES, CP)
        s_ub = cur[case]['baseline'] / t_lb_pipe if t_lb_pipe > 0 else float('inf')
        s_ub_naive = cur[case]['baseline'] / t_lb_naive if t_lb_naive > 0 else float('inf')
        sp = cur[case]['speedup']
        rows.append({
            'case': case, 'type': typ.get(case, '?'), 'n_ops': a['n_ops'],
            'W': W, 'CP': CP, 'CP/W': CP / W if W else 0.0,
            'baseline_makespan': cur[case]['baseline'], 'problem1_makespan': cur[case]['p1'],
            'speedup': sp,
            'S_ub': s_ub, 'S_ub_naive': s_ub_naive,
            'headroom': s_ub - sp, 'ratio': sp / s_ub if s_ub > 0 else 0.0,
            'T_lb_pipe': t_lb_pipe,
            'busiest_pipe': busy_pipe[0],
            'busiest_share': busy_pipe[1] / W if W else 0.0,
            'violates_naive': cur[case]['p1'] < t_lb_naive,
            **{f'share_{p}': a['pipe_cycles'].get(p, 0) / W if W else 0.0 for p in PIPES},
        })

    # ---------- 写 CSV ----------
    cols = ['case', 'type', 'n_ops', 'W', 'CP', 'CP/W', 'baseline_makespan',
            'problem1_makespan', 'speedup', 'S_ub', 'headroom', 'ratio',
            'S_ub_naive', 'T_lb_pipe', 'busiest_pipe', 'busiest_share',
            'violates_naive'] + [f'share_{p}' for p in PIPES]
    with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float('nan')

    def median(xs):
        xs = sorted(xs)
        m = len(xs) // 2
        return (xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2) if xs else float('nan')

    print('=' * 78)
    print(f"交叉验证: 自算 CP 与 case_profile.critical_path_cycles 最大相对偏差 = {cp_diff_max:.2%}")
    print(f"naive 下界(max(W/5,CP))被现状违反(makespan < T_lb_naive)的 case 数: "
          f"{sum(1 for r in rows if r['violates_naive'])}/100")
    if skipped:
        print(f"跳过(依赖环): {skipped}")

    # a) 全体
    print('\n[a] 全部 100 case (pipe 感知口径 S_ub = baseline / max(CP, 最忙pipe/5)):')
    print(f"    S_ub      均值 {mean([r['S_ub'] for r in rows]):.4f}  中位数 {median([r['S_ub'] for r in rows]):.4f}")
    print(f"    S_ub_naive均值 {mean([r['S_ub_naive'] for r in rows]):.4f}  中位数 {median([r['S_ub_naive'] for r in rows]):.4f}  (题目口径, 仅供参考)")
    print(f"    当前      均值 {mean([r['speedup'] for r in rows]):.4f}  中位数 {median([r['speedup'] for r in rows]):.4f}")
    print(f"    headroom  均值 {mean([r['headroom'] for r in rows]):.4f}  当前/上界 均值 {mean([r['ratio'] for r in rows]):.2%}")

    # b) 按类型
    print('\n[b] 按类型分组:')
    print(f"    {'type':<16}{'n':>3}{'当前均速':>10}{'S_ub均值':>10}{'headroom':>10}{'当前/上界':>10}{'CP/W均':>8}")
    for t in sorted({r['type'] for r in rows}):
        g = [r for r in rows if r['type'] == t]
        print(f"    {t:<16}{len(g):>3}{mean([r['speedup'] for r in g]):>10.4f}"
              f"{mean([r['S_ub'] for r in g]):>10.4f}{mean([r['headroom'] for r in g]):>10.4f}"
              f"{mean([r['ratio'] for r in g]):>10.2%}{mean([r['CP/W'] for r in g]):>8.4f}")

    # c) 当前 speedup 最低 25
    bot = sorted(rows, key=lambda r: r['speedup'])[:25]
    print('\n[c] 当前 speedup 最低的 25 个 case:')
    print(f"    {'case':<10}{'type':<16}{'sp':>6}{'S_ub':>7}{'head':>6}{'CP/W':>7}  最忙pipe(占比)")
    for r in bot:
        print(f"    {r['case']:<10}{r['type']:<16}{r['speedup']:>6.3f}{r['S_ub']:>7.3f}"
              f"{r['headroom']:>6.3f}{r['CP/W']:>7.4f}  {r['busiest_pipe']}({r['busiest_share']:.0%})")

    # d) headroom 最大 15
    top = sorted(rows, key=lambda r: -r['headroom'])[:15]
    print('\n[d] headroom 最大的 15 个 case (最值得再优化):')
    print(f"    {'case':<10}{'type':<16}{'sp':>6}{'S_ub':>7}{'head':>6}{'CP/W':>7}  最忙pipe(占比)")
    for r in top:
        print(f"    {r['case']:<10}{r['type']:<16}{r['speedup']:>6.3f}{r['S_ub']:>7.3f}"
              f"{r['headroom']:>6.3f}{r['CP/W']:>7.4f}  {r['busiest_pipe']}({r['busiest_share']:.0%})")

    # e) 结论素材: 底部 case 分类
    low = [r for r in rows if r['speedup'] < 1.5]
    near = [r for r in low if r['ratio'] >= 0.8 or r['headroom'] < 0.3]
    room = [r for r in low if r not in near]
    print(f"\n[e] speedup<1.5 的 case 共 {len(low)} 个: "
          f"到顶/近顶(当前>=80%上界 或 headroom<0.3) {len(near)} 个, 明显有救 {len(room)} 个")
    if room:
        print(f"    有救 case: {[r['case'] for r in sorted(room, key=lambda x: -x['headroom'])]}")
        print(f"    有救组特征: 类型分布 { {t: sum(1 for r in room if r['type']==t) for t in sorted({r['type'] for r in room})} }, "
              f"CP/W均值 {mean([r['CP/W'] for r in room]):.4f}, 最忙pipe占比均值 {mean([r['busiest_share'] for r in room]):.2%}")
    print(f"\nCSV 已写出: {OUT_CSV}")


if __name__ == '__main__':
    main()
