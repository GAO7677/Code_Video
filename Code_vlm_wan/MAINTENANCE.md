# Code VLM Wan 维护约定

本文件适用于 `/home/gaoya/Code_Video/Code_vlm_wan/`。目标是让项目保持简洁、可复用、可验证，避免继续堆积一次性脚本。

## 目录职责

```text
scripts/
  analysis/                 数据集统计、输入审计
  data/                     数据准备、前缀视频、结果合并
  inference/qwen3vl/        Qwen3-VL 推理和 caption 实验
  inference/qwen38/         Qwen3.8 推理和持久 worker
  visualization/            可视化数据构建和前台服务
tests/                      单元测试
configs/                    小型 case 清单和轻量配置
prompts/                    提示词单一来源
physv_*_viewer/             页面静态资源
```

`Qwen3-VL/` 和 `Qwen3-VL-32B-Thinking-FP8/` 是外部代码/模型目录，不在本项目整理范围内。模型权重、数据集、缓存、推理结果和视频素材统一放在 `/data/gaoya/`，不要写入本项目目录。

## 写新代码前

1. 先用 `rg` 查找已有实现、入口和调用方；能复用就不要新建相似脚本。
2. 明确脚本属于哪个目录；不要把实验脚本直接放到项目根目录。
3. 先定义可验证的输入、输出和成功条件，再改代码。
4. 提示词放在 `prompts/`，不要复制到多个 Python 文件中。
5. 共享逻辑只保留一个实现；其他入口通过导入、命令行参数或兼容链接复用它。

## 简洁性约定

- 一个功能只保留一个 canonical implementation；不要复制出 `*_new`、`*_final`、`*_v2` 等平行版本。
- 优先扩展现有函数和命令行参数，避免为单个 case 增加专用分支。
- 路径使用 `pathlib` 和参数传入；大多数输出使用 `/data/gaoya/agent-data/outputs/`。
- 不在代码中暂存模型、复制数据或硬编码大文件内容。
- 不为尚未出现的需求增加抽象层、配置系统或依赖。
- 不使用 GPU4；GPU 选择通过显式参数或 `CUDA_VISIBLE_DEVICES` 指定。
- 可视化服务默认以前台运行，并在交付时给出完整启动命令。

## 入口和兼容性

根目录中的 `.py` 文件是兼容软链接，真实代码位于 `scripts/`。新代码应编辑分类目录中的 canonical 文件，不要直接改根目录链接。

推荐使用分类后的路径，例如：

```bash
python scripts/inference/qwen38/run_qwen38_json_cases.py ...
python scripts/data/prepare_physv_v2v_cases.py ...
python scripts/visualization/serve_qwen38_multi_viewer.py ...
```

旧的根目录脚本路径暂时保留，以兼容已有启动命令和正在运行的服务。新增脚本不再放到根目录。

## 修改后的最小验证

```bash
cd /home/gaoya/Code_Video/Code_vlm_wan
python -m compileall -q scripts tests
python -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

涉及推理或可视化时，再用一个小 case 做 smoke test；不要为了验证而重新跑完整数据集。

## 变更记录原则

每次新增公共入口、移动目录、改变输出格式或改变提示词时，在提交说明或相关结果目录记录：输入、prompt、模型、GPU、关键参数、输出路径和验证命令。
