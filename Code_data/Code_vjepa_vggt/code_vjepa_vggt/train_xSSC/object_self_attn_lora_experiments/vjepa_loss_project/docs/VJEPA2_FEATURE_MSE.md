# V-JEPA2 feature MSE for tiny VAE visualization

This project compares tiny-VAE decoded videos from the xSSC visualization with
V-JEPA2 encoder features.

Default comparison:

```bash
gt_x0.mp4 vs pred_x0.mp4
```

Run on the latest visualization directory:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/compute_vjepa2_feature_mse.py --device cuda:0
```

Run only a few pairs:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/compute_vjepa2_feature_mse.py --device cuda:0 --max-pairs 5
```

Use the 16G V-JEPA2 ViT-g 384 checkpoint instead of the default ViT-L:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/vjepa_loss_project/compute_vjepa2_feature_mse.py \
  --device cuda:0 \
  --model vjepa2-vitg-384 \
  --checkpoint /data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt
```

Outputs default to:

```bash
/data/gaoya/agent-data/outputs/vjepa2_tinyvae_mse/<timestamp>/mse_results.json
```
