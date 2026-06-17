1. context_video -> SAM2 产生 prompt/mask/box -> 从 SAM2 采样 query priors -> resize 到 
VGGT 输入分辨率 -> 分别送入 VGGT / CoTracker -> 输出 track



```
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/inspect_vggt_query_points_overlay.py --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml --split train --start-index 0 --num-cases 8 --output-dir /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/track_pipeline_5rows --no-serve


python3 -m http.server 8790 --directory /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/track_pipeline_5rows
```