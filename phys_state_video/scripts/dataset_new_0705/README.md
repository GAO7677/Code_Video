# dataset_new_0705

这组脚本用于做 `0613pybullet` 的下一版数据定义层，目标只聚焦两件事：

1. 增加物体与场景多样性。
2. 提升外观真实感与画面观感。

当前目录先不接完整 raw 生成与 episode 导出，而是先把后续生成器必须依赖的“规格层”拆出来，避免继续把逻辑塞进单个大脚本。

## 当前文件职责

- `common_specs.py`
  - 统一定义材质、相机、物体族、场景族和场景蓝图。
- `material_catalog_0705.py`
  - 定义更真实的材质目录、HDRI 目录和地面/背景主题。
  - 目前已接入本机现有纹理：
    - `/data/gaoya/dataset/blender_render_assets/polyhaven_v1/textures`
    - `/data/gaoya/dataset/textures/polyhaven_wood`
- `object_catalog_0705.py`
  - 定义物体族，而不是少量固定模板。
  - 当前覆盖：
    - `ball`
    - `capsule_can`
    - `flat_puck`
    - `upright_cylinder`
    - `crate_box`
    - `tall_box`
    - `cone_frustum`
    - `wedge_ramp`
    - `pillar_occluder`
    - `platform_block`
    - `wheel`
    - `spool`
    - `dumbbell`
- `scene_generators_0705.py`
  - 定义 F1-F10 的参数化场景生成逻辑。
  - 核心变化不是“增加 seed 抖动”，而是每个 family 内部允许切换不同 object family、material family、camera setup 和 surface setup。

## 设计原则

- 不再把“多样性”理解为固定模板附近的小抖动。
- 优先扩大：
  - 物体语义族
  - 材质族
  - 地面/背景/光照主题
  - 相机分布
- 保持与现有渲染脚本兼容：
  - `ObjectInstanceSpec.to_legacy_object_kwargs()` 可以直接喂给旧的 `make_obj(...)` 风格接口。

## 已解决的问题

- 物体种类不再局限于球、盒、圆柱、胶囊、圆盘。
- 材质不再只有单一 procedural 色块，增加了 wood / leather / concrete / painted metal / plastic / rubber 的目录定义。
- 相机不再固定为一个视角，而是有多个 base camera 加 jitter。
  - F1-F10 不再强绑定单一模板组合。

## 下一步实现顺序

1. 新建 `render_sim_0705.py`
   - 复用现有 PyBullet + pyrender 管线。
   - 从 `ScenarioBlueprint` 读取对象、相机、材质和 surface。
2. 新建 `generate_raw_dataset_0705.py`
   - 单 shard raw 生成。
3. 新建 `generate_raw_dataset_parallel_0705.py`
   - 多 GPU 总控。
4. 新建 `extract_events_0705.py`
   - 先抽碰撞、遮挡、支撑丢失、出画等事件。
5. 新建 `prepare_event_episodes_0705.py`
   - 按事件而不是固定窗口切 episode。

## 最小验证方式

后续生成器接上之前，可先在 Python 里验证多样性采样：

```python
from dataset_new_0705.scene_generators_0705 import preview_diversity_report
print(preview_diversity_report())
```

如果这一步看到每个 family 都能产出多个 object family / material / camera 组合，说明“定义层”已经具备继续接 raw 生成器的基础。

## 100 case 生成建议

正式批量生成建议使用：

```bash
PYTHONPATH=/home/gaoya/Code_Video/phys_state_video/scripts:/home/gaoya/Code_Video \
/data/gaoya/miniconda3/envs/wan/bin/python -m dataset_new_0705.generate_0706_batch \
  --output-root /data/gaoya/AAA_test_video/Dataset_physV/0706pybullet \
  --num-cases 100
```

默认输出结构：

```text
/data/gaoya/AAA_test_video/Dataset_physV/0706pybullet/
  manifest.json
  cases/
    F1/
    F2/
    F3/
    F4/
    F5/
    F6/
    F7/
    F8/
    F9/
    F10/
  qa_preview/
  logs/
  reports/
```
