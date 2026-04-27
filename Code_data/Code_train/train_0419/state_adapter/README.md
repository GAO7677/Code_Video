# State Adapter

This directory contains the oracle-state adapter side project for Wan2.2 TI2V.

Main files:

- `train_state_adapter.py`
  Train the oracle future-state adapter on top of a preset TV2V checkpoint.

- `oracle_state_adapter.py`
  Adapter module definition.

- `state_adapter_dataset.py`
  Dataset wrapper for oracle-state window samples.

- `build_stage1_subsets.py`
  Build Stage-1A/1B state prediction subsets from Genesis rigid data.

- `build_stage1_oracle_windows.py`
  Convert rigid synthetic data into Wan-friendly oracle-state training windows.

Subdirectories:

- `visualizations/`
  Visualization utilities specific to the state-adapter workflow.
