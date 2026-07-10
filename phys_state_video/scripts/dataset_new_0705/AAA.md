# dataset_new_0705 统一总入口说明

## 当前统一入口

- 页面目录：
  [unified_overview_20260710](/data/gaoya/agent-data/outputs/dataset_new_0705/unified_overview_20260710)
- 页面文件：
  [index.html](/data/gaoya/agent-data/outputs/dataset_new_0705/unified_overview_20260710/index.html)
- 本地访问地址：
  `http://127.0.0.1:18830/`

这个入口把两套结果放在同一页里：

- 刚体 batch：
  [AAA_check_0710](/data/gaoya/agent-data/outputs/dataset_new_0705/AAA_check_0710)
- MPM batch：
  [mpm_preview_batch_20260710](/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_preview_batch_20260710)


## 两套仿真的基本参数

### 刚体方案

- 求解器类型：
  `PyBullet rigid-body + Pyrender`
- 当前批次输出根目录：
  [AAA_check_0710](/data/gaoya/agent-data/outputs/dataset_new_0705/AAA_check_0710)
- 当前 case 数：
  `60`
- family：
  `F1-F10`
- 分辨率：
  `1280x720`
- 视频 fps：
  `30`
- 默认仿真频率：
  `SIM_HZ = 240`
- 默认时间步长：
  `dt = 1 / 240 = 0.0041667 s`
- 默认单段时长：
  `3.0 s`
- 默认步数：
  `3.0 * 240 = 720 steps`
- 额外说明：
  每个 case 的 `pre_roll_s`、`floor_friction`、camera 和 object 参数写在对应 `meta/*.json` 里。

对应实现参考：

- [generate_0706_batch.py](/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/generate_0706_batch.py)
- [render_sim_0705.py](/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/render_sim_0705.py)
- [generate_sim_preview_gallery.py](/home/gaoya/Code_Video/phys_state_video/scripts/generate_sim_preview_gallery.py)


### MPM 方案

- 求解器类型：
  `Genesis MPM Elastic + rigid coupling`
- 当前批次输出根目录：
  [mpm_preview_batch_20260710](/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_preview_batch_20260710)
- 当前 case 数：
  `13`
- family：
  `F1-F13`
- 默认视频 fps：
  `30`
- 默认可视化：
  大多数 case 为 `visual`，`F4` 当前默认 `particle`
- 分辨率：
  当前大多为 `960x544`，以各 case manifest 中 `camera.res` 为准
- 时间步长 / 子步 / horizon / grid density：
  每个 case 独立配置，写在对应 `manifest.json -> sim`
- 额外说明：
  每个 case 的 `dt`、`substeps`、`horizon`、`grid_density`、`mpm_vis_mode`、初末状态等都写在对应 manifest 中。

对应实现参考：

- [render_mpm_preview_case.py](/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/render_mpm_preview_case.py)
- [render_mpm_batch.py](/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/render_mpm_batch.py)
- [export_mpm_family_catalog.py](/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/export_mpm_family_catalog.py)


## 统一总览页生成脚本

- 统一总览页生成器：
  [build_unified_overview_page.py](/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/build_unified_overview_page.py)
- 一键收集并前台启动脚本：
  [run_unified_overview_20260710.sh](/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/run_unified_overview_20260710.sh)


## 一键收集命令

```bash
PYTHONPATH=/home/gaoya/Code_Video/phys_state_video/scripts:/home/gaoya/Code_Video /data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python /home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/build_unified_overview_page.py --rigid-root /data/gaoya/agent-data/outputs/dataset_new_0705/AAA_check_0710 --mpm-root /data/gaoya/agent-data/outputs/dataset_new_0705/mpm_preview_batch_20260710 --output-root /data/gaoya/agent-data/outputs/dataset_new_0705/unified_overview_20260710 --port 18830
```

这条命令会：

- 收集刚体 `AAA_check_0710`
- 收集 MPM `mpm_preview_batch_20260710`
- 生成统一入口：
  [index.html](/data/gaoya/agent-data/outputs/dataset_new_0705/unified_overview_20260710/index.html)


## 一键收集并前台可视化命令

```bash
/home/gaoya/Code_Video/phys_state_video/scripts/dataset_new_0705/run_unified_overview_20260710.sh
```

这条脚本会先刷新统一页面，再以前台方式启动本地服务。


## 前台启动服务命令

```bash
cd /data/gaoya/agent-data/outputs/dataset_new_0705/unified_overview_20260710
/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python -m http.server 18830 --bind 127.0.0.1
```

说明：

- 按当前约定，这个服务默认使用端口 `18830`
- 必须前台启动，不使用 `nohup`、`&` 或后台守护方式


## 当前状态

- 刚体统一页数据：
  `60` 个 case，`F1-F10`
- MPM 统一页数据：
  `13` 个 case，`F1-F13`
- 当前总入口本机验证状态：
  首页 `200 OK`
- 当前 MPM 视频资源本机验证状态：
  `200 OK`


## 相关结果文件

- 刚体总清单：
  [manifest.json](/data/gaoya/agent-data/outputs/dataset_new_0705/AAA_check_0710/manifest.json)
- MPM batch 总清单：
  [batch_manifest.json](/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_preview_batch_20260710/batch_manifest.json)
- MPM family 总表：
  [mpm_family_catalog.md](/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_family_catalog_20260710/mpm_family_catalog.md)
- 统一页数据摘要：
  [overview_data.json](/data/gaoya/agent-data/outputs/dataset_new_0705/unified_overview_20260710/overview_data.json)
