# PhyGround

本目录提供两个最小工具：

- `build_phyground_index.py`: 解析 `prompts/`、`annotations/`、`videos/`，输出统一索引 `phyground_index.json`
- `serve_phyground_local.py`: 把索引和本地视频目录映射到一个 HTTP 端口

示例：

```bash
python /home/gaoya/Code_Video/Code_data/Code_benchmark/phyground/build_phyground_index.py
python /home/gaoya/Code_Video/Code_data/Code_benchmark/phyground/serve_phyground_local.py --port 18701
```

访问：

```text
http://127.0.0.1:18701/
```
