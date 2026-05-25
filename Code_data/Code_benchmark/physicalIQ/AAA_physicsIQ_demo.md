# Physics-IQ Demo 版

## 目标

做一个最小可用的一键入口，只跑用户指定的少量 case，并完成两件事：

1. 生成这些 case 的预测视频  
2. 计算这些 case 的子集 score，并在本地端口展示输入 / 输出 / 分数

## 推荐入口

```bash
python run_physics_iq_demo.py \
  --cases 0002,0010,0042 \
  --methods wan,vace \
  --port 18701
```

访问：

```text
http://127.0.0.1:18701
```

## 输入

- `--cases`：只接受 `take-1` 的 case id，支持逗号分隔
- `--methods`：方法列表，当前支持 `wan,vace`
- `--port`：本地查看端口

## 输出

- `generated_videos/<method>/*.mp4`
- `eval_outputs/results/<method>.csv`
- `eval_outputs/results/<method>.subset_score.json`
- `viewer/index.html`

## 最小实现链路

直接复用现有代码：

- 样本清单：`descriptions/descriptions.csv`
- 生成脚本参考：`wan22_ti2v_physics_iq_eval_multigpu.py`
- 打分函数：`physics-IQ-benchmark-main/code/calculate_iq_score.py`
- 汇总脚本参考：`export_physics_iq_method_summary.py`
- 本地端口服务参考：`phyground/serve_phyground_local.py`

建议 demo wrapper 做 4 步：

1. 从 `descriptions.csv` 里筛出指定 case  
2. 只生成这些 case 的视频  
3. 只对这些 case 产出评测 CSV，再调用 `calculate_iq_score(csv)`  
4. 启一个本地 viewer，把 `case / caption / first frame / gt / pred / subset score` 暴露到端口

## 结果定义

这个版本的分数应明确标成：

```text
subset demo score
```

不要标成官方 Physics-IQ score，因为它不是 198 个 `take-1` case 的全量结果。

## 备注

- 现有 `wan22_ti2v_physics_iq_eval_multigpu.py` 的 `--limit` 只适合生成 smoke test，不适合直接做官方评测。
- demo 版适合联调、可视化和快速回归，不适合对外报分。

## 本地执行

实际执行时建议直接指定空闲 GPU，并把结果放到一个独立 run 目录：

```bash
conda run -n wan python /home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/run_physics_iq_demo.py \
  --cases 0002,0041,0110 \
  --methods wan,vace \
  --wan_device cuda:3 \
  --vace_device cuda:4 \
  --run_name compare_0002_0041_0110 \
  --overwrite
```

只启动本地可视化端口：

```bash
conda run -n wan python /home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/run_physics_iq_demo.py \
  --cases 0002,0041,0110 \
  --methods wan,vace \
  --run_name compare_0002_0041_0110 \
  --skip_generation \
  --skip_evaluation \
  --serve \
  --port 18701
```

访问：

```text
http://127.0.0.1:18701
```

结果目录：

```text
/data/gaoya/AAA_test_video/Benchmark/physics_IQ_demo/runs/compare_0002_0041_0110
```

## 已执行的本地对比页

这次先复用了本地已有结果做一个只读对比页：

- Wan：`/data/gaoya/AAA_test_video/Benchmark/physics_IQ/generated_videos/wan_22_ti2v_5b`
- VACE ctx08：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/output/VACE_1_3B_V2V/context_08f`

实际选用 case：

```text
0005,0020,0029,0032,0038
```

生成对比页：

```bash
python3 /home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/build_physics_iq_compare_viewer.py \
  --cases 0005,0020,0029,0032,0038
```

启动本地端口：

```bash
python3 /home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/serve_physics_iq_compare_viewer.py \
  --port 18711
```

访问：

```text
http://127.0.0.1:18711
```
