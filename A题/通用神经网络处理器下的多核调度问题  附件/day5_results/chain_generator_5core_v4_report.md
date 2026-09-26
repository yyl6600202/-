# A 题问题一：链簇生成器继续优化结果

本轮在上一版链簇生成器上加入了多组轮转分配，并对低于 3.8× 的 53 个 case 做了官方评估；随后对最难的小图扫描了连续分块和 wave 分块数量。每个候选只有在官方 Makespan 严格下降时才替换原方案。

新增规则包括：

- `root_roundrobin`：根链按拓扑顺序切成最多 4 倍核心数的组，再循环分配到核心；
- `root_roundrobin_weighted`：按根分支 cycles 加权切组后循环分配；
- `root_load`：按根分支负载装箱；
- `structured_wave_2`：对 `case_069` 采用 2 个 wave 子图。

## 100-case 5 核结果

结果文件：`chain_generator_5core_v4_summary.csv`、`chain_generator_5core_v4_stats.json` 和 `chain_generator_5core_v4_plans/`。

| 指标 | v4 | 上一版 v1 |
|---|---:|---:|
| 算术平均加速比 | **3.3472×** | 2.9145× |
| 几何平均加速比 | 2.9727× | 2.5299× |
| 中位数加速比 | 3.7161× | 3.0918× |
| 总周期比 | 3.6322× | 3.3007× |
| 最小 / 最大加速比 | 1.0000× / 5.4115× | — |

全部 100 个 case 仍通过官方校验且无评估错误。相较 v1，平均加速比增加 **0.4327×**，相对提升约 **14.84%**。相较上一版 v3，本轮 `case_069` 从 25144 降至 25008 周期，平均值增加 0.00005×。

本轮轮转规则带来的主要改进包括：`case_025` 983471 周期、`case_050` 82168 周期、`case_056` 191133 周期、`case_068` 248861 周期、`case_075` 1073002 周期、`case_085` 2514429 周期、`case_088` 161354 周期；`case_069` 的 wave 分块为 25008 周期。

当前平均值仍低于截图中的 3.622×。剩余主要瓶颈是 `case_047` 等深链图，以及部分 mixed 图的跨层依赖；继续提升需要引入跨层边界搬运试探和按 Pipe 负载调整组边界。

## 复现

在附件根目录执行：

```powershell
python code/chain_roundrobin_batch.py --max-speedup 3.8
python code/chain_group_search.py --case case_050 --case case_056 --case case_068 --case case_075 --case case_085
python code/chain_structured_sweep.py --case case_069
```

上述脚本都使用固定 5 核和官方问题一事件模拟器。最终方案目录中的 100 个 JSON 文件是逐 case 选出的官方可接受方案。
