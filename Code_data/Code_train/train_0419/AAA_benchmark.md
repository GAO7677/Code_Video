
## 1.7 Stage0 benchmark 可视化

目录约定：

- output：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/output`
- result：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result`
- tools：`/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/tools`

重建主可视化页面：

```bash
kport 8040
python /home/gaoya/Code_Video/Code_data/Code_train/train_0419/build_stage0_metric_linecharts.py
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
