#!/usr/bin/env python3
"""OpenVid LoRA inference with the shared matched-Head ablation implementation."""

from code_vjepa_vggt.AAAinfer import wan_openvid_lorav2v

import infer_wan_lora_common22_public_head_ablation as implementation


if __name__ == "__main__":
    implementation.base = wan_openvid_lorav2v
    implementation.main()
