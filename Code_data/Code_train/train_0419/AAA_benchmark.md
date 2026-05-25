
## 1.7 Stage0 benchmark 可视化

目录约定：

- output：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/output`
- result：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result`
- tools：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/tools`
- nullcaption output 已并入主 benchmark：
  - `output/VACE_1_3B_V2V_nullcaption/context_08f`
  - `output/VACE_1_3B_V2V_nullcaption/context_fullctx_fullvideo`

重建主可视化页面：

```bash
kport 8040
source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan
python /home/gaoya/Code_Video/Code_data/Code_train/train_0419/build_stage0_metric_linecharts.py
python /home/gaoya/Code_Video/Code_data/Code_train/train_0419/nullcaption_rerun/build_caption_vs_nullcaption_portal.py
python /home/gaoya/Code_Video/Code_data/Code_train/train_0419/nullcaption_rerun/build_fullctx_fullvideo_portal.py
```


启动本地静态端口：

```bash
# 停用端口

cd /data/gaoya/AAA_test_video/Benchmark/stage0_V2V
python -m http.server 8040 --bind 127.0.0.1

/result/model_metric_linecharts_latest/index.html
```

当前主页面：

- `http://127.0.0.1:8040/result/model_metric_linecharts_latest/index.html`
- `http://127.0.0.1:8040/tools/visualization/caption_vs_nullcaption_portal/index.html`
- `http://127.0.0.1:8040/tools/visualization/physicsiq_fullctx_fullvideo_portal/index.html`

增量评测 nullcaption ctx08：

```bash
source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan

python /home/gaoya/Code_Video/Code_data/Code_train/train_0419/backfill_stage0_per_sample_metrics.py \
  --benchmark_root /data/gaoya/AAA_test_video/Benchmark/stage0_V2V \
  --result_root /data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result/per_sample_future_metrics \
  --model_name vace_v2v_ctx08f_nullcaption

python /home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_stage0_vbench_short.py \
  --benchmark_root /data/gaoya/AAA_test_video/Benchmark/stage0_V2V \
  --output_root /data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result/model_metrics_vbench_short_nullcaption \
  --runtime_root /data/gaoya/AAA_test_video/Benchmark/stage0_V2V/tools/runtime \
  --model_name vace_v2v_ctx08f_nullcaption
```

表格里的方法名会统一写成 `*_nullcaption`，例如 `VACE ctx08_nullcaption`，和原始 caption 版本区分开。


