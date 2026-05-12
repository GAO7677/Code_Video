# State Adapter

  - 输入 context RGB + prompt + future 9D oracle state
  - 冻结 Wan 主干，只训练一个轻量 OracleStateAdapter
  - adapter 把 future 9D state 编成按帧的 future-plan token
  - 在每个 Wan DiT block 后，用 zero-init 的 gamma/beta 去调制 future frame 的 latent spatial tokens
  - 训练目标仍然是原视频扩散/flow-matching loss
  - 当前训练数据直接读取 summary root：
    `/home/gaoya/Code_Video/Code_data/data0417/data_summary/stage1adapter_simple_window`

Main files:

- `train_state_adapter.py`
  Train the oracle future-state adapter on top of a preset TV2V checkpoint.

- `oracle_state_adapter.py`
  Adapter module definition.

- `state_adapter_dataset.py`
  Dataset wrapper for oracle-state window samples. It can read the organized summary root directly and defaults to the simple-motion buckets (`no_collision`, `env_only`) under the `train` split.

- `motion_complexity.py`
  Window-level motion-complexity scoring and bucket definitions for static/simple/moderate/complex splits.

- `backfill_motion_complexity.py`
  Offline backfill utility that writes motion-complexity and window-interaction metadata into existing `pair_meta.json` files.

- `window_interactions.py`
  Window-level object-count and future-collision bucket summaries derived from source `event_windows.json`.

Subdirectories:

- `visualizations/`
  Visualization utilities specific to the state-adapter workflow.

Launcher:

- `../run_train_state_adapter_oracle_alltrain.sh`
  Directly launches adapter training on `stage1adapter_simple_window` without any extra raw-to-window preprocessing step.
