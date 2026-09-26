# A 题问题一：链簇生成器重写与 5 核实测

## 生成器

本次重写新增了 `code/chain_cluster_partition.py`。生成流程对应参考材料中的“链簇、装箱、连续子图”思路：

1. 用 Step1 的确定性拓扑执行序遍历所有非 COPY 算子。
2. 对每个算子选择数据量最大的已处理前驱作为主链前驱；没有前驱的算子成为根。
3. 同一根节点的后继组成一个 lineage 分支。根分支可以按拓扑出现顺序连续分组，也可以按分支 cycles 加权连续分组；另外保留按 cycles 装箱和按根 ID 循环的候选用于小图探索。
4. 只有当相邻算子的核心标签改变时才切 Task。每个 Task 是全局拓扑序上的连续区间，因此压缩后的 Task 图保持 DAG。
5. 所有方案先经过 `validate_multicore_plan`，再交给官方问题一事件模拟器评估。

批量入口是 `code/chain_problem1_batch.py`。它同时评估单核基准、已有 5 核方案和链簇候选，以官方 Makespan 为第一目标，新增搬运量为平局项；每个 case 最终保存一个官方可接受方案。

## 100-case 结果

结果文件：

- `chain_generator_5core_summary.csv`
- `chain_generator_5core_stats.json`
- `chain_generator_5core_plans/`

| 指标 | 重写生成器 | 原 5 核方案 |
|---|---:|---:|
| 算术平均加速比 | **3.2967×** | 2.9145× |
| 几何平均加速比 | 2.8919× | 2.5299× |
| 中位数加速比 | 3.6399× | 3.0918× |
| 总周期比 | 3.5449× | 3.3007× |
| 最小 / 最大加速比 | 1.0000× / 5.4115× | — |

相对原方案，37 个 case 得到改善，63 个保持原方案；没有 case 变慢。算术平均加速比增加 **0.3822×**，相对提升 **13.11%**。全部 100 个 case 均无评估错误，最终方案均通过官方校验。

按图类别的算术平均加速比为：`high_spill_risk` 3.8935×、`wide_shallow` 3.4840×、`mixed` 2.0001×、`deep_narrow` 1.1941×。提升主要来自宽浅汇聚图和高溢出风险图，例如 `case_002` 的 Makespan 从 259045 降至 56191，`case_062` 从 2780149 降至 578116，`case_063` 从 1045141 降至 219468。

## 复现命令

在附件目录执行：

```powershell
python code/chain_cluster_partition.py data/case_002.json --num-cores 5 --dominants bytes id
python code/chain_problem1_batch.py `
  --num-cores 5 `
  --dominants bytes id `
  --modes root_contiguous root_contiguous_weighted `
  --screen-summary day5_results/problem1_5core_optimized_summary.csv `
  --skip-speedup-at-least 4.5 `
  --output day5_results/chain_generator_5core_summary.csv `
  --plans-dir day5_results/chain_generator_5core_plans
```

批处理中的 12 个已有加速比至少为 4.5× 的 case 直接保留已通过官方评估的 incumbent，以减少超大图的重复模拟；其余 case 均进行了链候选的官方评估。屏幕筛选不会改变结果，只会跳过已经足够好的旧方案。

当前实测平均值还没有达到参考截图中的 3.622×；这套生成器已经将平均值从 2.9145× 提升到 3.2967×，后续继续优化应优先针对 `mixed` 和 `deep_narrow` 类图设计跨层链簇与边界搬运策略。
