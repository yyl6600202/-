# A题问题一 LaTeX

本目录基于 `2026 华为杯论文模板/LaTex模板2/GMCM2026.zip` 中的官方 2026 示例整理。`main.tex` 的导言区由官方 `example.tex` 生成，使用官方 `gmcmthesis.cls`、封面图、标题图和宏包配置。

文件说明：

- `main.tex`：独立编译入口。
- `problem1_section.tex`：问题一正文，可复制到完整论文模板中使用。
- `figures/v4_v5_speedup.pdf`：由 v4/v5 汇总结果生成、使用宋体标注的对比图。
- `gmcmthesis.cls`、`gmcm.bst`、`figures/logo.pdf`、`figures/title.pdf`：官方模板依赖文件。
- `accepted_improvements.csv`：混合搜索接受的 6 个改进用例。
- `build_main.py`、`build_figure.py`：从官方示例导言区生成主文件，并重绘结果图。

在本目录执行两次 XeLaTeX：

```powershell
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

`build_main.py` 从随包附带的官方 2026 `example.tex` 导言区重建 `main.tex`。如需重绘加速比图，需将原 A 题附件结果目录保留在本工作区并运行 `python build_figure.py`；正常编译无需此步骤。

正文中的数字来自：

`A题/通用神经网络处理器下的多核调度问题  附件/day5_results/chain_generator_5core_v5_summary.csv`

和

`A题/通用神经网络处理器下的多核调度问题  附件/day5_results/chain_generator_5core_v5_stats.json`。

Windows 下沿用官方模板默认字体配置，正文为宋体，标题使用模板定义的字体，西文使用 Times New Roman；其他操作系统会按官方类文件的系统字体分支选取字体。
