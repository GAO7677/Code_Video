# train0705 compare_ablation

这个目录放 `train0705` 的训练级消融实验入口。

原则：

- 不改原始训练文件
- 在本目录单独放一份训练入口
- 各消融都通过本目录脚本启动
- 默认沿用正式训练配置，只改目标消融项


## 1. 文件说明

- `train_stage1b_context_only_no_gt_box_v_newtrain_ablation.py`
  - 本目录专用训练入口
  - 基于原 `train0705/train_stage1b_context_only_no_gt_box_v_newtrain.py`
  - 新增训练级开关：
    - `--disable-cotracker`
    - `--disable-jepa`
    - `--disable-vggt`

- `run_train_stage1b_compare_ablation_base_gpu0235.sh`
  - 公共基座脚本
  - 4 个具体消融脚本都调用它


## 2. 消融实验

### 2.1 No Stage1A init

含义：

- 不加载 `Stage1A` 的 `object_pooler / object_aux_heads` 初始化
- 其它训练配置保持不变

运行命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/run_train_stage1b_no_stage1a_init_gpu0235.sh
```


### 2.2 w/o CoTracker

含义：

- 不使用真实 `CoTracker` 特征
- 训练入口里会用静态 query 复制形成占位 track
- 其它模块保持不变

运行命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/run_train_stage1b_wo_cotracker_gpu0235.sh
```


### 2.3 w/o JEPA

含义：

- 不使用真实 `JEPA` patch tokens
- 训练入口里会用零 token 占位
- 其它模块保持不变

运行命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/run_train_stage1b_wo_jepa_gpu0235.sh
```


### 2.4 w/o VGGT

含义：

- 不使用 `VGGT` 几何 / 深度 / dense token 特征
- 训练入口里直接关闭 VGGT 路径
- 其它模块保持不变

运行命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/run_train_stage1b_wo_vggt_gpu0235.sh
```


## 3. 输出目录

默认输出到：

```text
/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705_compare_ablation/
```

每个实验各自落在一个子目录下：

- `no_stage1a_init`
- `wo_cotracker`
- `wo_jepa`
- `wo_vggt`


## 4. 备注

- 默认使用 `GPU_SET=0,2,3,5`，明确不使用 `gpu4`
- 如需改输出目录或卡号，可在命令前覆盖环境变量，例如：

```bash
GPU_SET=0,2 OUTPUT_DIR=/data/gaoya/agent-data/checkpoints/my_ablation bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/run_train_stage1b_wo_vggt_gpu0235.sh
```
