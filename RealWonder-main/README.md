# RealWonder: Real-Time Physical Action-Conditioned Video Generation

<div align="center">

[![Website](https://img.shields.io/badge/Website-RealWonder-blue)](https://liuwei283.github.io/RealWonder/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.05449-red)](https://arxiv.org/abs/2603.05449)
[![twitter](https://img.shields.io/twitter/url?label=TL:DR&url=https%3A%2F%2Ftwitter.com%)](https://x.com/Koven_Yu/status/2029745851095290293?s=20)
</div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/afe46b79-51e5-4b52-8fa7-782ae4ddda86" width="80%" style="max-width: 100%; height: auto;" />
</p>

## About

Current video generation models cannot simulate physical consequences of 3D actions like forces and robotic manipulations, as they lack structural understanding of how actions affect 3D scenes. We present RealWonder, the first real-time system for action-conditioned video generation from a single image. Our key insight is using physics simulation as an intermediate bridge: instead of directly encoding continuous actions, we translate them through physics simulation into visual representations (optical flow and RGB) that video models can process. RealWonder integrates three components: 3D reconstruction from single images, physics simulation, and a distilled video generator requiring only 4 diffusion steps. Our system achieves 13.2 FPS at 480×832 resolution, enabling interactive exploration of forces, robot actions, and camera controls on rigid objects, deformable bodies, fluids, and granular materials.

> **RealWonder: Real-Time Physical Action-Conditioned Video Generation** <br> [Project Page](https://liuwei283.github.io/RealWonder/) | [Paper](https://arxiv.org/abs/2603.05449) <br> [Wei Liu](https://liuwei283.github.io/)\*, [Ziyu Chen](https://ziyc.github.io/)\*, [Zizhang Li](https://kyleleey.github.io/), [Yue Wang](https://yuewang.xyz/), [Hong-Xing (Koven) Yu](https://kovenyu.com/)†, [Jiajun Wu](https://jiajunwu.com/)† <br> Stanford University, University of Southern California <br> \*Equal contribution &nbsp; †Equal advising

<!-- ![Teaser](assets/teaser.jpeg) -->

## Installation

### 1. Create Environment

```bash
conda env create -f default.yml
conda activate realwonder
```

### 2. Install SAM 3D Objects

```bash
cd submodules/sam_3d_objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
./patching/hydra
cd ../..
```

#### Checkpoints

```bash
pip install 'huggingface-hub[cli]<1.0'
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

### 3. Install SAM 2

```bash
cd submodules/sam2
pip install -e .
cd checkpoints && ./download_ckpts.sh && cd ..
cd ../..
```

### 4. Install Genesis

```bash
cd submodules/Genesis
git checkout 3aa206cd84729bc7cc14fb4007aeb95a0bead7aa
pip install -e .
cd ../..
```

### 5. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 6. Download Model Checkpoints

```bash
hf download ziyc/realwonder --include "Realwonder-Distilled-AR-I2V-Flow/*" --local-dir ckpts/
hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP --local-dir wan_models/Wan2.1-Fun-V1.1-1.3B-InP
```

## Usage

### Interactive Demo (Real-Time UI)

Tested on NVIDIA H200 GPU with CUDA 12.1.

#### Installation

```bash
pip install -r demo_web/requirements.txt
```

#### How to run

```bash
cd demo_web
python app.py \
    --demo_data demo_data/lamp \
    --checkpoint_path /path/to/checkpoint.pt
```

### Offline Inference

Run physics simulation:

```bash
python case_simulation.py --config_path demo_data/lamp/config.yaml
```

Run video generation from simulation results:

```bash
python infer_sim.py \
    --checkpoint_path ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt \
    --sim_data_path result/lamp/final_sim \
    --output_path result/lamp/final_sim/final.mp4
```

## Citation

```bibtex
@misc{realwonder2026,
  title={RealWonder: Real-Time Physical Action-Conditioned Video Generation},
  author={Liu, Wei and Chen, Ziyu and Li, Zizhang and Wang, Yue and Yu, Hong-Xing and Wu, Jiajun},
  year={2026},
  eprint={2603.05449},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.05449},
}
```
