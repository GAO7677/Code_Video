#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import genesis as gs


def main() -> None:
    out_dir = Path("/home/gaoya/Code_Video/Code_data/demo_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "sph_box_demo.mp4"

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            lower_bound=(-0.5, -0.5, 0.0),
            upper_bound=(0.5, 0.5, 1.0),
            particle_size=0.01,
        ),
        vis_options=gs.options.VisOptions(
            visualize_sph_boundary=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    scene.add_entity(
        material=gs.materials.SPH.Liquid(
            sampler="pbs",
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.65),
            size=(0.4, 0.4, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.8, 1.0),
            vis_mode="particle",
        ),
    )

    cam = scene.add_camera(
        pos=(1.15, -1.25, 0.95),
        lookat=(0.0, 0.0, 0.42),
        res=(960, 960),
        fov=32,
        GUI=False,
    )

    scene.build()

    cam.start_recording()
    horizon = 240
    for _ in range(horizon):
        scene.step()
        cam.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)
    cam.stop_recording(save_to_filename=video_path, fps=60)

    print(video_path)


if __name__ == "__main__":
    main()
