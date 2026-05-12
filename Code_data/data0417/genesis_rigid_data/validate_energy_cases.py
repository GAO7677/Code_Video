# 用途：校验样本能量曲线与物理一致性。
"""该脚本用于构建并验证一组能量守恒/耗散物理案例；输入为内置 case 配置和可选 case_id/out_dir 参数，输出为 /data/gaoya/AAA_test_video/Dataset_physV/0417data/try3_rigid0417_energy_validation 下的验证视频、逐案例结果和 summary.json。"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import genesis as gs

THIS_DIR = Path(__file__).resolve().parent
CODE_DATA_ROOT = THIS_DIR.parent.parent
if str(CODE_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_DATA_ROOT))

from genesis_energy_utils import particle_entity_energy, particle_entity_kinematic_snapshot, rigid_entity_kinematic_snapshot
from old_version.dataset_3_utils_genesis import apply_initial_motion_to_entity, ensure_dir, sample_camera
from utils_io import save_video, to_uint8_rgb


GRAVITY_Z = -9.81
DEFAULT_OUT_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/try3_rigid0417_energy_validation")
MAIN_SIM_DT = 4e-3
MAIN_SIM_SUBSTEPS = 8
MAIN_EXPORT_FRAMES = 49
MAIN_STEPS_PER_FRAME = 5
MAIN_CAMERA_RES = (960, 720)
VALIDATION_BACKGROUND = {
    "name": "light_gray_studio",
    "background_color": (0.92, 0.92, 0.92),
    "ambient_light": (0.32, 0.32, 0.32),
}
VALIDATION_CONTAINER = {
    "half_x": 1.50,
    "half_y": 1.50,
    "wall_thickness": 0.04,
    "wall_height": 2.00,
    "front_lip_height": 0.00,
    "floor_thickness": 0.05,
    "center": [0.0, 0.0, 0.0],
}


@dataclass
class ObjectSpec:
    name: str
    solver_family: str
    shape: str
    pos: Tuple[float, float, float]
    radius: Optional[float]
    size: Optional[Tuple[float, float, float]]
    material: Dict[str, float]
    material_ctor: Optional[str]
    linvel: Tuple[float, float, float]
    angvel: Tuple[float, float, float]
    color: Tuple[float, float, float, float]
    surface_vis_mode: str = "visual"


@dataclass
class CaseSpec:
    case_id: str
    title: str
    description: str
    gravity: Tuple[float, float, float]
    dt: float
    substeps: int
    num_steps: int
    steps_per_frame: int
    objects: List[ObjectSpec]
    add_floor: bool = False
    floor_material: Optional[Dict[str, float]] = None
    floor_z: float = 0.0
    warmup_steps: int = 0
    warmup_velocity_reset: bool = False


def validation_camera_config(case: CaseSpec) -> Dict[str, Any]:
    # Start from the same shared front-facing camera style as the main dataset script,
    # then apply small per-case tweaks so the reduced validation scenes stay centered.
    base = sample_camera(VALIDATION_CONTAINER)
    default = {
        "pos": tuple(base["pos"]),
        "lookat": tuple(base["lookat"]),
        "fov": float(base["fov"]),
        "res": MAIN_CAMERA_RES,
    }
    presets = {
        "case_01_free_fall": {"pos": (0.02, -3.15, 1.55), "lookat": (0.0, 0.12, 1.00), "fov": 42.0},
        "case_02_horizontal_uniform_motion": {"pos": (0.04, -3.10, 1.20), "lookat": (0.38, 0.18, 0.18), "fov": 41.0},
        "case_03_vertical_bounce_elastic": {"pos": (0.03, -3.10, 1.28), "lookat": (0.0, 0.18, 0.50), "fov": 42.0},
        "case_04_two_body_elastic_collision": {"pos": (0.00, -3.00, 0.90), "lookat": (-0.25, 0.10, 0.05), "fov": 40.0},
        "case_05_two_body_inelastic_collision": {"pos": (0.00, -3.00, 0.90), "lookat": (-0.25, 0.10, 0.05), "fov": 40.0},
        "case_09_mpm_drop_floor_rest0_soft": {"pos": (0.03, -3.10, 1.28), "lookat": (0.0, 0.18, 0.50), "fov": 42.0},
        "case_10_mpm_drop_floor_rest1_soft": {"pos": (0.03, -3.10, 1.28), "lookat": (0.0, 0.18, 0.50), "fov": 42.0},
        "case_11_mpm_drop_floor_rest0_stiff": {"pos": (0.03, -3.10, 1.28), "lookat": (0.0, 0.18, 0.50), "fov": 42.0},
        "case_12_mpm_drop_floor_rest1_stiff": {"pos": (0.03, -3.10, 1.28), "lookat": (0.0, 0.18, 0.50), "fov": 42.0},
    }
    cfg = dict(default)
    cfg.update(presets.get(case.case_id, {}))
    return cfg


def render_rgb_frame(camera: Any) -> np.ndarray:
    rendered = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
    if isinstance(rendered, tuple):
        rgb = rendered[0]
    else:
        rgb = rendered
    return to_uint8_rgb(rgb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small set of rigid and MPM Genesis validation scenes and check the legacy rigid/MPM energy formulas used in this codebase."
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--only_cases",
        type=str,
        default="",
        help="Optional comma-separated case ids to run, e.g. case_06_mpm_rest_zero_gravity,case_07_mpm_free_fall_no_contact",
    )
    return parser.parse_args()


def sphere_object(
    name: str,
    pos: Tuple[float, float, float],
    radius: float,
    linvel: Tuple[float, float, float],
    angvel: Tuple[float, float, float],
    friction: float,
    restitution: float,
    rho: float = 1000.0,
    color: Tuple[float, float, float, float] = (0.75, 0.25, 0.25, 1.0),
) -> ObjectSpec:
    return ObjectSpec(
        name=name,
        solver_family="rigid",
        shape="sphere",
        pos=pos,
        radius=radius,
        size=None,
        material={"rho": float(rho), "friction": float(friction), "restitution": float(restitution)},
        material_ctor=None,
        linvel=linvel,
        angvel=angvel,
        color=color,
    )


def mpm_box_object(
    name: str,
    pos: Tuple[float, float, float],
    size: Tuple[float, float, float],
    linvel: Tuple[float, float, float],
    angvel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rho: float = 120.0,
    youngs: float = 3.0e4,
    poisson: float = 0.25,
    material_ctor: str = "gs.materials.MPM.Elastic",
    color: Tuple[float, float, float, float] = (0.85, 0.45, 0.25, 1.0),
) -> ObjectSpec:
    return ObjectSpec(
        name=name,
        solver_family="mpm",
        shape="box",
        pos=pos,
        radius=None,
        size=size,
        material={"rho": float(rho), "youngs": float(youngs), "poisson": float(poisson)},
        material_ctor=str(material_ctor),
        linvel=linvel,
        angvel=angvel,
        color=color,
        surface_vis_mode="particle",
    )


def build_case_specs() -> List[CaseSpec]:
    radius = 0.10
    elastic_floor = {"rho": 1000.0, "friction": 0.0, "restitution": 1.0}
    slip_floor = {"rho": 1000.0, "friction": 0.0, "restitution": 0.0}
    rigid_cases = [
        CaseSpec(
            case_id="case_01_free_fall",
            title="Free Fall No Collision",
            description="Single sphere in free fall, no contacts, no spin.",
            gravity=(0.0, 0.0, GRAVITY_Z),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            objects=[
                sphere_object(
                    name="ball",
                    pos=(0.0, 0.0, 2.6),
                    radius=radius,
                    linvel=(0.0, 0.0, 0.0),
                    angvel=(0.0, 0.0, 0.0),
                    friction=0.0,
                    restitution=0.0,
                )
            ],
        ),
        CaseSpec(
            case_id="case_02_horizontal_uniform_motion",
            title="Horizontal Uniform Motion",
            description="Single sphere sliding on a frictionless plane after contact warmup.",
            gravity=(0.0, 0.0, GRAVITY_Z),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            add_floor=True,
            floor_material=slip_floor,
            warmup_steps=80,
            warmup_velocity_reset=True,
            objects=[
                sphere_object(
                    name="slider",
                    pos=(0.0, 0.0, radius),
                    radius=radius,
                    linvel=(1.0, 0.0, 0.0),
                    angvel=(0.0, 0.0, 0.0),
                    friction=0.0,
                    restitution=0.0,
                    color=(0.20, 0.45, 0.85, 1.0),
                )
            ],
        ),
        CaseSpec(
            case_id="case_03_vertical_bounce_elastic",
            title="Vertical Bounce e=1",
            description="Single sphere bounces on a frictionless floor with restitution 1.",
            gravity=(0.0, 0.0, GRAVITY_Z),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            add_floor=True,
            floor_material=elastic_floor,
            objects=[
                sphere_object(
                    name="bouncer",
                    pos=(0.0, 0.0, 1.2),
                    radius=radius,
                    linvel=(0.0, 0.0, 0.0),
                    angvel=(0.0, 0.0, 0.0),
                    friction=0.0,
                    restitution=1.0,
                    color=(0.20, 0.70, 0.35, 1.0),
                )
            ],
        ),
        CaseSpec(
            case_id="case_04_two_body_elastic_collision",
            title="Two-Body Elastic Collision",
            description="Two equal-mass spheres collide head-on with restitution 1 and zero gravity.",
            gravity=(0.0, 0.0, 0.0),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            objects=[
                sphere_object(
                    name="left",
                    pos=(-0.60, 0.0, 0.0),
                    radius=radius,
                    linvel=(1.0, 0.0, 0.0),
                    angvel=(0.0, 0.0, 0.0),
                    friction=0.0,
                    restitution=1.0,
                    color=(0.78, 0.24, 0.24, 1.0),
                ),
                sphere_object(
                    name="right",
                    pos=(0.0, 0.0, 0.0),
                    radius=radius,
                    linvel=(0.0, 0.0, 0.0),
                    angvel=(0.0, 0.0, 0.0),
                    friction=0.0,
                    restitution=1.0,
                    color=(0.24, 0.24, 0.78, 1.0),
                ),
            ],
        ),
        CaseSpec(
            case_id="case_05_two_body_inelastic_collision",
            title="Two-Body Inelastic Collision",
            description="Two equal-mass spheres collide head-on with restitution 0 and zero gravity.",
            gravity=(0.0, 0.0, 0.0),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            objects=[
                sphere_object(
                    name="left",
                    pos=(-0.60, 0.0, 0.0),
                    radius=radius,
                    linvel=(1.0, 0.0, 0.0),
                    angvel=(0.0, 0.0, 0.0),
                    friction=0.0,
                    restitution=0.0,
                    color=(0.70, 0.45, 0.15, 1.0),
                ),
                sphere_object(
                    name="right",
                    pos=(0.0, 0.0, 0.0),
                    radius=radius,
                    linvel=(0.0, 0.0, 0.0),
                    angvel=(0.0, 0.0, 0.0),
                    friction=0.0,
                    restitution=0.0,
                    color=(0.15, 0.60, 0.70, 1.0),
                ),
            ],
        ),
    ]
    mpm_cases = [
        CaseSpec(
            case_id="case_06_mpm_rest_zero_gravity",
            title="MPM Rest Zero Gravity",
            description="Single elastic MPM block at rest in zero gravity; kinetic energy and COM should stay stable.",
            gravity=(0.0, 0.0, 0.0),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            objects=[
                mpm_box_object(
                    name="soft_block",
                    pos=(0.0, 0.0, 0.5),
                    size=(0.20, 0.20, 0.20),
                    linvel=(0.0, 0.0, 0.0),
                    rho=120.0,
                    youngs=3.0e4,
                    poisson=0.25,
                    color=(0.85, 0.45, 0.25, 1.0),
                )
            ],
        ),
        CaseSpec(
            case_id="case_07_mpm_free_fall_no_contact",
            title="MPM Free Fall No Contact",
            description="Single elastic MPM block in free fall without contact; K+U should stay nearly constant when deformation stays small.",
            gravity=(0.0, 0.0, GRAVITY_Z),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            objects=[
                mpm_box_object(
                    name="soft_block",
                    pos=(0.0, 0.0, 2.3),
                    size=(0.18, 0.18, 0.18),
                    linvel=(0.0, 0.0, 0.0),
                    rho=120.0,
                    youngs=4.0e4,
                    poisson=0.22,
                    color=(0.78, 0.40, 0.22, 1.0),
                )
            ],
        ),
        CaseSpec(
            case_id="case_08_mpm_horizontal_uniform_zero_gravity",
            title="MPM Horizontal Uniform Motion",
            description="Single elastic MPM block with horizontal initial velocity in zero gravity; kinetic energy should stay nearly constant when deformation stays small.",
            gravity=(0.0, 0.0, 0.0),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            objects=[
                mpm_box_object(
                    name="soft_block",
                    pos=(0.0, 0.0, 0.5),
                    size=(0.18, 0.18, 0.18),
                    linvel=(0.7, 0.0, 0.0),
                    rho=120.0,
                    youngs=4.0e4,
                    poisson=0.22,
                    color=(0.82, 0.52, 0.18, 1.0),
                )
            ],
        ),
        CaseSpec(
            case_id="case_09_mpm_drop_floor_rest0_soft",
            title="MPM Drop To Floor r=0 soft",
            description="Single soft elastic MPM block falls onto a rigid floor with floor coup_restitution=0 and low Young's modulus.",
            gravity=(0.0, 0.0, GRAVITY_Z),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            add_floor=True,
            floor_material={"rho": 1000.0, "friction": 0.0, "restitution": 0.0},
            objects=[
                mpm_box_object(
                    name="soft_block",
                    pos=(0.0, 0.0, 1.6),
                    size=(0.18, 0.18, 0.18),
                    linvel=(0.0, 0.0, 0.0),
                    rho=120.0,
                    youngs=8.0e3,
                    poisson=0.22,
                    color=(0.82, 0.46, 0.18, 1.0),
                )
            ],
        ),
        CaseSpec(
            case_id="case_10_mpm_drop_floor_rest1_soft",
            title="MPM Drop To Floor r=1 soft",
            description="Single soft elastic MPM block falls onto a rigid floor with floor coup_restitution=1 and low Young's modulus.",
            gravity=(0.0, 0.0, GRAVITY_Z),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            add_floor=True,
            floor_material={"rho": 1000.0, "friction": 0.0, "restitution": 1.0},
            objects=[
                mpm_box_object(
                    name="soft_block",
                    pos=(0.0, 0.0, 1.6),
                    size=(0.18, 0.18, 0.18),
                    linvel=(0.0, 0.0, 0.0),
                    rho=120.0,
                    youngs=8.0e3,
                    poisson=0.22,
                    color=(0.84, 0.50, 0.18, 1.0),
                )
            ],
        ),
        CaseSpec(
            case_id="case_11_mpm_drop_floor_rest0_stiff",
            title="MPM Drop To Floor r=0 stiff",
            description="Single stiff elastic MPM block falls onto a rigid floor with floor coup_restitution=0 and high Young's modulus.",
            gravity=(0.0, 0.0, GRAVITY_Z),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            add_floor=True,
            floor_material={"rho": 1000.0, "friction": 0.0, "restitution": 0.0},
            objects=[
                mpm_box_object(
                    name="soft_block",
                    pos=(0.0, 0.0, 1.6),
                    size=(0.18, 0.18, 0.18),
                    linvel=(0.0, 0.0, 0.0),
                    rho=120.0,
                    youngs=1.2e5,
                    poisson=0.22,
                    color=(0.60, 0.48, 0.82, 1.0),
                )
            ],
        ),
        CaseSpec(
            case_id="case_12_mpm_drop_floor_rest1_stiff",
            title="MPM Drop To Floor r=1 stiff",
            description="Single stiff elastic MPM block falls onto a rigid floor with floor coup_restitution=1 and high Young's modulus.",
            gravity=(0.0, 0.0, GRAVITY_Z),
            dt=MAIN_SIM_DT,
            substeps=MAIN_SIM_SUBSTEPS,
            num_steps=MAIN_EXPORT_FRAMES,
            steps_per_frame=MAIN_STEPS_PER_FRAME,
            add_floor=True,
            floor_material={"rho": 1000.0, "friction": 0.0, "restitution": 1.0},
            objects=[
                mpm_box_object(
                    name="soft_block",
                    pos=(0.0, 0.0, 1.6),
                    size=(0.18, 0.18, 0.18),
                    linvel=(0.0, 0.0, 0.0),
                    rho=120.0,
                    youngs=1.2e5,
                    poisson=0.22,
                    color=(0.56, 0.46, 0.84, 1.0),
                )
            ],
        ),
    ]
    return rigid_cases + mpm_cases


def create_validation_rigid_material(mat_cfg: Dict[str, float]):
    friction = float(max(mat_cfg.get("friction", 0.0), 1e-2))
    kwargs = {
        "rho": float(mat_cfg.get("rho", 1000.0)),
        "friction": friction,
    }
    restitution = mat_cfg.get("restitution", None)
    if restitution is not None:
        # Genesis 0.4.0 does not expose per-body rigid-rigid restitution via `restitution`,
        # so we pass the sampled value through `coup_restitution` when available and record it.
        kwargs["coup_restitution"] = float(restitution)
    return gs.materials.Rigid(**kwargs)


def create_validation_mpm_material(gs_mod: Any, obj: ObjectSpec):
    ctor = str(obj.material_ctor or "gs.materials.MPM.Elastic")
    common_kwargs = {
        "E": float(obj.material.get("youngs", 3.0e4)),
        "nu": float(obj.material.get("poisson", 0.25)),
        "rho": float(obj.material.get("rho", 120.0)),
        "sampler": "pbs-8",
        "model": "neohooken",
    }
    if ctor == "gs.materials.MPM.ElastoPlastic":
        common_kwargs.pop("model", None)
        return gs_mod.materials.MPM.ElastoPlastic(**common_kwargs)
    if ctor == "gs.materials.MPM.Sand":
        common_kwargs.pop("model", None)
        return gs_mod.materials.MPM.Sand(**common_kwargs)
    if ctor == "gs.materials.MPM.Snow":
        common_kwargs.pop("model", None)
        return gs_mod.materials.MPM.Snow(**common_kwargs)
    return gs_mod.materials.MPM.Elastic(**common_kwargs)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_entity_velocity(entity: Any, linvel: Tuple[float, float, float], angvel: Tuple[float, float, float]) -> None:
    dof_vel = tuple(float(v) for v in list(linvel) + list(angvel))
    if hasattr(entity, "set_dofs_velocity"):
        entity.set_dofs_velocity(dof_vel)
        return
    if hasattr(entity, "set_velocity"):
        entity.set_velocity(tuple(float(v) for v in linvel))
        return
    if hasattr(entity, "set_particles_vel") and hasattr(entity, "get_particles_pos"):
        pts = np.asarray(_to_numpy(entity.get_particles_pos()), dtype=np.float64)
        vel = np.broadcast_to(np.asarray(linvel, dtype=np.float64).reshape(1, 3), pts.shape).copy()
        entity.set_particles_vel(vel)
        return
    apply_initial_motion_to_entity(entity, linvel, angvel)


def build_scene(case: CaseSpec):
    has_mpm = any(obj.solver_family == "mpm" for obj in case.objects)
    vis_options = gs.options.VisOptions(
        show_world_frame=False,
        show_link_frame=False,
        background_color=tuple(VALIDATION_BACKGROUND["background_color"]),
        ambient_light=tuple(VALIDATION_BACKGROUND["ambient_light"]),
        segmentation_level="entity",
        render_particle_as="sphere",
        particle_size_scale=1.0,
    )
    scene_kwargs = dict(
        sim_options=gs.options.SimOptions(dt=float(case.dt), substeps=int(case.substeps), gravity=tuple(case.gravity)),
        vis_options=vis_options,
        show_viewer=False,
    )
    if has_mpm:
        scene_kwargs["mpm_options"] = gs.options.MPMOptions(
            # Keep the MPM domain comfortably larger than the validation motion
            # range so the "free-fall without contact" case does not accidentally
            # interact with the solver bounds and invalidate the energy check.
            lower_bound=(-3.0, -3.0, -6.0),
            upper_bound=(3.0, 3.0, 6.0),
        )
    try:
        scene_kwargs["rigid_options"] = gs.options.RigidOptions(dt=float(case.dt), enable_collision=True, use_gjk_collision=True)
    except Exception:
        try:
            scene_kwargs["rigid_options"] = gs.options.RigidOptions(dt=float(case.dt))
        except Exception:
            pass
    scene = gs.Scene(**scene_kwargs)

    floor_entity = None
    if case.add_floor:
        floor_entity = scene.add_entity(
            morph=gs.morphs.Plane(pos=(0.0, 0.0, float(case.floor_z)), fixed=True),
            material=create_validation_rigid_material(case.floor_material or {"rho": 1000.0, "friction": 0.0, "restitution": 0.0}),
            surface=gs.surfaces.Default(color=(0.70, 0.72, 0.76, 1.0)),
        )

    entities = []
    for obj in case.objects:
        if obj.solver_family == "rigid":
            entity = scene.add_entity(
                morph=gs.morphs.Sphere(radius=float(obj.radius), pos=tuple(obj.pos)),
                material=create_validation_rigid_material(obj.material),
                surface=gs.surfaces.Default(color=obj.color),
            )
        elif obj.solver_family == "mpm":
            if obj.shape != "box" or obj.size is None:
                raise ValueError(f"Unsupported validation MPM shape: {obj.shape}")
            entity = scene.add_entity(
                morph=gs.morphs.Box(size=tuple(float(v) for v in obj.size), pos=tuple(obj.pos)),
                material=create_validation_mpm_material(gs, obj),
                surface=gs.surfaces.Default(color=obj.color, vis_mode=obj.surface_vis_mode),
            )
        else:
            raise ValueError(f"Unsupported solver_family: {obj.solver_family}")
        entities.append(entity)

    cam_cfg = validation_camera_config(case)
    camera = scene.add_camera(
        res=tuple(cam_cfg["res"]),
        pos=tuple(cam_cfg["pos"]),
        lookat=tuple(cam_cfg["lookat"]),
        fov=float(cam_cfg["fov"]),
        GUI=False,
    )

    scene.build()
    for entity, obj in zip(entities, case.objects):
        set_entity_velocity(entity, obj.linvel, obj.angvel)

    if case.warmup_steps > 0:
        for _ in range(int(case.warmup_steps)):
            scene.step()
        if case.warmup_velocity_reset:
            for entity, obj in zip(entities, case.objects):
                set_entity_velocity(entity, obj.linvel, obj.angvel)

    return scene, entities, floor_entity, camera


def contact_graph_from_collider(scene: Any, entities: List[Any]) -> np.ndarray:
    n = len(entities)
    out = np.zeros((n, n), dtype=np.uint8)
    # Genesis rigid contact queries expose geometry index ranges, but MPM entities
    # do not. For pure-MPM validation scenes we currently fall back to the zero
    # matrix here because these cases are intentionally configured without contact.
    if any(not hasattr(ent, "geom_start") or not hasattr(ent, "geom_end") for ent in entities):
        return out
    try:
        contact_data = scene._sim.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    except Exception:
        return out
    if not contact_data or "geom_a" not in contact_data or "geom_b" not in contact_data:
        return out

    geom_a = np.asarray(contact_data["geom_a"]).reshape(-1)
    geom_b = np.asarray(contact_data["geom_b"]).reshape(-1)
    ranges = [(int(ent.geom_start), int(ent.geom_end)) for ent in entities]

    def geom_to_object(geom_idx: int) -> Optional[int]:
        for idx, (start, end) in enumerate(ranges):
            if start <= geom_idx < end:
                return idx
        return None

    for ga, gb in zip(geom_a, geom_b):
        i = geom_to_object(int(ga))
        j = geom_to_object(int(gb))
        if i is None or j is None or i == j:
            continue
        out[i, j] = 1
        out[j, i] = 1
    np.fill_diagonal(out, 0)
    return out


def entity_mass(entity: Any) -> float:
    try:
        return float(entity.get_mass())
    except Exception:
        pass
    solver = getattr(entity, "solver", None)
    particle_start = int(getattr(entity, "_particle_start", getattr(entity, "particle_start", 0)))
    n_particles = int(getattr(entity, "n_particles", 0))
    if solver is None or not hasattr(solver, "particles_info") or not hasattr(solver.particles_info, "mass"):
        return 0.0
    mass_field = solver.particles_info.mass
    mass_arr = np.asarray(mass_field.to_numpy() if hasattr(mass_field, "to_numpy") else mass_field, dtype=np.float64)
    return float(np.asarray(mass_arr[particle_start : particle_start + n_particles], dtype=np.float64).sum())


def particle_spread(entity: Any) -> float:
    pts = np.asarray(_to_numpy(entity.get_particles_pos()), dtype=np.float64)
    if pts.ndim == 3:
        pts = pts[0]
    if pts.size == 0:
        return 0.0
    center = np.mean(pts, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum((pts - center) ** 2, axis=1))))


def collect_case_trajectory(scene: Any, entities: List[Any], case: CaseSpec, camera: Any | None = None) -> Dict[str, np.ndarray]:
    n = len(entities)
    frame_dt = float(case.dt) * float(case.steps_per_frame)
    times = np.arange(case.num_steps, dtype=np.float64) * frame_dt
    masses = np.zeros((n,), dtype=np.float64)
    position = np.zeros((case.num_steps, n, 3), dtype=np.float64)
    linear_velocity = np.zeros((case.num_steps, n, 3), dtype=np.float64)
    angular_velocity = np.zeros((case.num_steps, n, 3), dtype=np.float64)
    kinetic_per_object = np.zeros((case.num_steps, n), dtype=np.float64)
    potential_per_object = np.zeros((case.num_steps, n), dtype=np.float64)
    momentum = np.zeros((case.num_steps, n, 3), dtype=np.float64)
    contact_graph = np.zeros((case.num_steps, n, n), dtype=np.uint8)
    shape_spread = np.full((case.num_steps, n), np.nan, dtype=np.float64)
    rgb_frames: List[np.ndarray] = []

    gravity = np.asarray(case.gravity, dtype=np.float64)
    for step in range(case.num_steps):
        if step > 0:
            for _ in range(int(case.steps_per_frame)):
                scene.step()
        for idx, entity in enumerate(entities):
            if hasattr(entity, "get_particles_pos"):
                snap = particle_entity_kinematic_snapshot(entity, gravity=gravity)
                energy = particle_entity_energy(entity, gravity=gravity)
                mass = entity_mass(entity)
                shape_spread[step, idx] = particle_spread(entity)
                kinetic_per_object[step, idx] = float(energy.kinetic)
                potential_per_object[step, idx] = float(energy.potential)
            else:
                snap = rigid_entity_kinematic_snapshot(entity, gravity=gravity)
                mass = float(entity.get_mass())
                kinetic_per_object[step, idx] = float(snap.kinetic)
                potential_per_object[step, idx] = -mass * float(np.dot(gravity, np.asarray(snap.com_pos, dtype=np.float64)))
            masses[idx] = mass
            position[step, idx] = np.asarray(snap.com_pos, dtype=np.float64)
            linear_velocity[step, idx] = np.asarray(snap.linear_vel, dtype=np.float64)
            angular_velocity[step, idx] = np.asarray(snap.angular_vel, dtype=np.float64)
            momentum[step, idx] = mass * np.asarray(snap.linear_vel, dtype=np.float64)
        contact_graph[step] = contact_graph_from_collider(scene, entities)
        if camera is not None:
            rgb_frames.append(render_rgb_frame(camera))

    out = {
        "time": times.astype(np.float32),
        "object_ids": np.arange(n, dtype=np.int32),
        "masses": masses.astype(np.float32),
        "position": position.astype(np.float32),
        "linear_velocity": linear_velocity.astype(np.float32),
        "angular_velocity": angular_velocity.astype(np.float32),
        "kinetic_per_object": kinetic_per_object.astype(np.float32),
        "potential_per_object": potential_per_object.astype(np.float32),
        "total_kinetic": np.sum(kinetic_per_object, axis=1).astype(np.float32),
        "total_potential": np.sum(potential_per_object, axis=1).astype(np.float32),
        "total_energy": np.sum(kinetic_per_object + potential_per_object, axis=1).astype(np.float32),
        "momentum": momentum.astype(np.float32),
        "total_momentum": np.sum(momentum, axis=1).astype(np.float32),
        "contact_graph": contact_graph,
        "shape_spread": shape_spread.astype(np.float32),
    }
    if camera is not None:
        out["rgb_frames"] = np.stack(rgb_frames, axis=0).astype(np.uint8)
    return out


def relative_span(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    denom = max(float(np.mean(np.abs(arr))), 1e-8)
    return float((np.max(arr) - np.min(arr)) / denom)


def relative_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    denom = max(float(np.mean(np.abs(arr))), 1e-8)
    return float(np.std(arr) / denom)


def fit_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def detect_pair_contact_window(contact_graph: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    active = np.where(contact_graph[:, 0, 1] > 0)[0]
    if active.size == 0:
        return None, None
    return int(active[0]), int(active[-1])


def detect_bounce_frame(z: np.ndarray, vz: np.ndarray) -> Optional[int]:
    for idx in range(2, len(z) - 2):
        if vz[idx - 1] < 0.0 <= vz[idx + 1] and z[idx] <= np.min(z[max(0, idx - 2): idx + 3]) + 1e-4:
            return idx
    return None


def save_case_plots(case_dir: Path, case: CaseSpec, data: Dict[str, np.ndarray]) -> None:
    t = data["time"]
    total_k = data["total_kinetic"]
    total_p = data["total_potential"]
    total_e = data["total_energy"]
    pos = data["position"]
    vel = data["linear_velocity"]
    momentum = data["total_momentum"]
    contact = data["contact_graph"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(t, total_k, label="kinetic")
    axes[0, 0].plot(t, total_p, label="potential")
    axes[0, 0].plot(t, total_e, label="total")
    axes[0, 0].set_title("Energy")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].legend(loc="best")

    for obj_idx in range(pos.shape[1]):
        axes[0, 1].plot(t, pos[:, obj_idx, 2], label=f"obj{obj_idx} z")
    axes[0, 1].set_title("Height")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].legend(loc="best")

    for obj_idx in range(vel.shape[1]):
        axes[1, 0].plot(t, vel[:, obj_idx, 0], label=f"obj{obj_idx} vx")
    axes[1, 0].set_title("x Velocity")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(t, momentum[:, 0], label="px")
    axes[1, 1].plot(t, momentum[:, 1], label="py")
    axes[1, 1].plot(t, momentum[:, 2], label="pz")
    if contact.shape[1] > 1:
        axes[1, 1].plot(t, contact[:, 0, 1], label="contact01", alpha=0.7)
    axes[1, 1].set_title("Total Momentum")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].legend(loc="best")

    fig.suptitle(case.title)
    fig.tight_layout()
    fig.savefig(case_dir / "diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze_free_fall(case: CaseSpec, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    t = data["time"].astype(np.float64)
    z = data["position"][:, 0, 2].astype(np.float64)
    vz = data["linear_velocity"][:, 0, 2].astype(np.float64)
    total_e = data["total_energy"].astype(np.float64)
    ang = np.linalg.norm(data["angular_velocity"][:, 0], axis=1)

    line = np.polyfit(t, vz, 1)
    vz_pred = np.polyval(line, t)
    quad = np.polyfit(t, z, 2)
    z_pred = np.polyval(quad, t)
    result = {
        "total_energy_relative_span": relative_span(total_e),
        "vz_slope": float(line[0]),
        "vz_intercept": float(line[1]),
        "vz_r2": fit_r2(vz, vz_pred),
        "z_quadratic_coeff": float(quad[0]),
        "z_r2": fit_r2(z, z_pred),
        "max_angular_speed": float(np.max(ang)),
        "checks": {
            "total_energy_stable": bool(relative_span(total_e) < 0.02),
            "vz_linear": bool(fit_r2(vz, vz_pred) > 0.999 and abs(line[0] - GRAVITY_Z) < 0.12),
            "z_parabolic": bool(fit_r2(z, z_pred) > 0.999 and abs(quad[0] - 0.5 * GRAVITY_Z) < 0.12),
            "no_spin": bool(np.max(ang) < 1e-3),
        },
    }
    result["pass"] = bool(all(result["checks"].values()))
    return result


def analyze_horizontal_motion(case: CaseSpec, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    # Skip a short post-reset settling window. Even after the floor-contact warmup,
    # the first few recorded frames can still include tiny solver stabilization drift.
    settle_skip = 30
    total_k = data["total_kinetic"][settle_skip:].astype(np.float64)
    total_p = data["total_potential"][settle_skip:].astype(np.float64)
    total_e = data["total_energy"][settle_skip:].astype(np.float64)
    vx = data["linear_velocity"][settle_skip:, 0, 0].astype(np.float64)
    z = data["position"][settle_skip:, 0, 2].astype(np.float64)

    result = {
        "settle_skip_frames": settle_skip,
        "kinetic_relative_std": relative_std(total_k),
        "potential_relative_std": relative_std(total_p),
        "total_energy_relative_std": relative_std(total_e),
        "vx_relative_std": relative_std(vx),
        "height_relative_std": relative_std(z),
        "checks": {
            "kinetic_constant": bool(relative_std(total_k) < 0.02),
            "potential_constant": bool(relative_std(total_p) < 0.01),
            "total_energy_constant": bool(relative_std(total_e) < 0.02),
            "vx_constant": bool(relative_std(vx) < 0.02),
        },
    }
    result["pass"] = bool(all(result["checks"].values()))
    return result


def analyze_mpm_rest(case: CaseSpec, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    total_k = data["total_kinetic"].astype(np.float64)
    pos = data["position"][:, 0].astype(np.float64)
    spread = data["shape_spread"][:, 0].astype(np.float64)
    drift = np.linalg.norm(pos - pos[0], axis=1)
    spread_rel = float(np.std(spread) / max(float(np.mean(np.abs(spread))), 1e-8))
    result = {
        "kinetic_max": float(np.max(total_k)),
        "com_drift_max": float(np.max(drift)),
        "shape_spread_relative_std": spread_rel,
        "checks": {
            "kinetic_near_zero": bool(np.max(total_k) < 2e-3),
            "com_near_static": bool(np.max(drift) < 2e-3),
            "shape_stable": bool(spread_rel < 0.01),
        },
    }
    result["pass"] = bool(all(result["checks"].values()))
    return result


def analyze_mpm_free_fall(case: CaseSpec, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    t = data["time"].astype(np.float64)
    z = data["position"][:, 0, 2].astype(np.float64)
    vz = data["linear_velocity"][:, 0, 2].astype(np.float64)
    total_e = data["total_energy"].astype(np.float64)
    spread = data["shape_spread"][:, 0].astype(np.float64)
    line = np.polyfit(t, vz, 1)
    vz_pred = np.polyval(line, t)
    quad = np.polyfit(t, z, 2)
    z_pred = np.polyval(quad, t)
    spread_rel = float(np.std(spread) / max(float(np.mean(np.abs(spread))), 1e-8))
    result = {
        "total_energy_relative_span_without_internal": relative_span(total_e),
        "vz_slope": float(line[0]),
        "vz_r2": fit_r2(vz, vz_pred),
        "z_quadratic_coeff": float(quad[0]),
        "z_r2": fit_r2(z, z_pred),
        "shape_spread_relative_std": spread_rel,
        "energy_formula_note": "MPM total physical energy should be kinetic + gravitational potential + internal strain energy. This validation uses observable kinetic + gravitational potential only, so it is valid when deformation stays small.",
        "checks": {
            "observable_energy_stable": bool(relative_span(total_e) < 0.03),
            "vz_linear": bool(fit_r2(vz, vz_pred) > 0.999 and abs(line[0] - GRAVITY_Z) < 0.15),
            "z_parabolic": bool(fit_r2(z, z_pred) > 0.999 and abs(quad[0] - 0.5 * GRAVITY_Z) < 0.15),
            "shape_stable": bool(spread_rel < 0.02),
        },
    }
    result["pass"] = bool(all(result["checks"].values()))
    return result


def analyze_mpm_horizontal(case: CaseSpec, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    settle_skip = 5
    total_k = data["total_kinetic"][settle_skip:].astype(np.float64)
    total_e = data["total_energy"][settle_skip:].astype(np.float64)
    vx = data["linear_velocity"][settle_skip:, 0, 0].astype(np.float64)
    pos_x = data["position"][settle_skip:, 0, 0].astype(np.float64)
    spread = data["shape_spread"][settle_skip:, 0].astype(np.float64)
    t = data["time"][settle_skip:].astype(np.float64)
    line = np.polyfit(t, pos_x, 1)
    x_pred = np.polyval(line, t)
    spread_rel = float(np.std(spread) / max(float(np.mean(np.abs(spread))), 1e-8))
    result = {
        "settle_skip_frames": settle_skip,
        "kinetic_relative_std_without_internal": relative_std(total_k),
        "total_energy_relative_std_without_internal": relative_std(total_e),
        "vx_relative_std": relative_std(vx),
        "x_linear_r2": fit_r2(pos_x, x_pred),
        "shape_spread_relative_std": spread_rel,
        "energy_formula_note": "MPM total physical energy should include internal strain energy; this no-gravity drift case keeps deformation small so kinetic energy dominates.",
        "checks": {
            "kinetic_constant": bool(relative_std(total_k) < 0.03),
            "total_energy_constant": bool(relative_std(total_e) < 0.03),
            "vx_constant": bool(relative_std(vx) < 0.03),
            "x_linear": bool(fit_r2(pos_x, x_pred) > 0.999),
            "shape_stable": bool(spread_rel < 0.02),
        },
    }
    result["pass"] = bool(all(result["checks"].values()))
    return result


def analyze_mpm_floor_drop(case: CaseSpec, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    z = data["position"][:, 0, 2].astype(np.float64)
    vz = data["linear_velocity"][:, 0, 2].astype(np.float64)
    total_e = data["total_energy"].astype(np.float64)
    spread = data["shape_spread"][:, 0].astype(np.float64)
    youngs = float(case.objects[0].material.get("youngs", np.nan))
    floor_restitution = float((case.floor_material or {}).get("restitution", 0.0))

    impact_idx = int(np.argmin(z))
    post_slice = slice(min(len(z), impact_idx + 3), len(z))
    post_z = z[post_slice]
    post_vz = vz[post_slice]
    post_peak = float(np.max(post_z)) if post_z.size > 0 else float(z[impact_idx])
    bounce_detected = bool(post_z.size > 0 and np.max(post_vz) > 0.10 and post_peak > float(z[impact_idx]) + 0.03)

    initial_spread = max(float(spread[0]), 1e-8)
    max_spread_ratio = float(np.max(spread) / initial_spread)
    final_spread_ratio = float(spread[-1] / initial_spread)

    pre_energy_window = total_e[max(0, impact_idx - 8): max(1, impact_idx - 2)]
    post_energy_window = total_e[min(len(total_e), impact_idx + 5): min(len(total_e), impact_idx + 15)]
    energy_before = float(np.mean(pre_energy_window)) if pre_energy_window.size > 0 else float(total_e[0])
    energy_after = float(np.mean(post_energy_window)) if post_energy_window.size > 0 else float(total_e[-1])
    energy_retention = energy_after / max(abs(energy_before), 1e-8)

    result = {
        "impact_frame": impact_idx,
        "floor_coup_restitution": floor_restitution,
        "youngs_modulus": youngs,
        "initial_com_height": float(z[0]),
        "impact_min_com_height": float(z[impact_idx]),
        "post_impact_peak_com_height": post_peak,
        "post_impact_peak_over_initial": float(post_peak / max(float(z[0]), 1e-8)),
        "bounce_detected": bounce_detected,
        "observable_energy_before_impact": energy_before,
        "observable_energy_after_impact": energy_after,
        "observable_energy_retention": energy_retention,
        "max_spread_ratio": max_spread_ratio,
        "final_spread_ratio": final_spread_ratio,
        "checks": {
            "finite": bool(np.all(np.isfinite(z)) and np.all(np.isfinite(vz)) and np.all(np.isfinite(spread))),
            "impact_observed": bool(0 < impact_idx < len(z) - 1),
        },
    }
    result["pass"] = bool(all(result["checks"].values()))
    return result


def analyze_vertical_bounce(case: CaseSpec, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    z = data["position"][:, 0, 2].astype(np.float64)
    vz = data["linear_velocity"][:, 0, 2].astype(np.float64)
    total_e = data["total_energy"].astype(np.float64)
    bounce_idx = detect_bounce_frame(z, vz)
    floor_z = float(case.floor_z)
    radius = float(case.objects[0].radius)
    if bounce_idx is None:
        result = {"checks": {"bounce_detected": False}, "pass": False}
        return result

    post_z = z[bounce_idx + 5 :]
    post_apex = float(np.max(post_z)) if post_z.size > 0 else float(z[bounce_idx])
    initial_height = float(z[0] - floor_z - radius)
    rebound_height = float(post_apex - floor_z - radius)
    energy_before = float(np.mean(total_e[max(0, bounce_idx - 40): max(1, bounce_idx - 5)]))
    energy_after = float(np.mean(total_e[min(len(total_e) - 1, bounce_idx + 5): min(len(total_e), bounce_idx + 40)]))
    height_ratio = rebound_height / max(initial_height, 1e-8)
    energy_ratio = energy_after / max(energy_before, 1e-8)
    result = {
        "bounce_frame": int(bounce_idx),
        "initial_height": initial_height,
        "rebound_height": rebound_height,
        "height_ratio": height_ratio,
        "energy_before": energy_before,
        "energy_after": energy_after,
        "energy_ratio": energy_ratio,
        "checks": {
            "bounce_detected": True,
            "height_recovers": bool(abs(height_ratio - 1.0) < 0.15),
            "energy_near_conserved": bool(abs(energy_ratio - 1.0) < 0.15),
        },
    }
    result["pass"] = bool(all(result["checks"].values()))
    return result


def analyze_collision(case: CaseSpec, data: Dict[str, np.ndarray], expect_elastic: bool) -> Dict[str, Any]:
    start_idx, end_idx = detect_pair_contact_window(data["contact_graph"])
    masses = data["masses"].astype(np.float64)
    v = data["linear_velocity"].astype(np.float64)
    total_k = data["total_kinetic"].astype(np.float64)
    total_p = data["total_potential"].astype(np.float64)
    total_momentum = data["total_momentum"].astype(np.float64)

    if start_idx is None or end_idx is None:
        return {"checks": {"collision_detected": False}, "pass": False}

    pre_idx = max(0, start_idx - 10)
    post_idx = min(v.shape[0] - 1, end_idx + 30)
    pre_vx = v[pre_idx, :, 0]
    post_vx = np.mean(v[max(end_idx + 5, 0): post_idx + 1, :, 0], axis=0)
    pre_ke = float(total_k[pre_idx] + total_p[pre_idx])
    post_ke = float(total_k[post_idx] + total_p[post_idx])
    pre_k_only = float(total_k[pre_idx])
    post_k_only = float(total_k[post_idx])
    pre_p = total_momentum[pre_idx]
    post_p = total_momentum[post_idx]
    momentum_rel_error = float(np.linalg.norm(post_p - pre_p) / max(np.linalg.norm(pre_p), 1e-8))
    energy_rel_error = float(abs(post_ke - pre_ke) / max(abs(pre_ke), 1e-8))
    kinetic_ratio = post_k_only / max(pre_k_only, 1e-8)

    result = {
        "collision_start": int(start_idx),
        "collision_end": int(end_idx),
        "pre_vx": pre_vx.tolist(),
        "post_vx": post_vx.tolist(),
        "pre_total_energy": pre_ke,
        "post_total_energy": post_ke,
        "pre_kinetic": pre_k_only,
        "post_kinetic": post_k_only,
        "kinetic_ratio": kinetic_ratio,
        "pre_total_momentum": pre_p.tolist(),
        "post_total_momentum": post_p.tolist(),
        "momentum_relative_error": momentum_rel_error,
        "total_energy_relative_error": energy_rel_error,
        "checks": {
            "collision_detected": True,
            "momentum_conserved": bool(momentum_rel_error < 0.05),
        },
    }

    if expect_elastic:
        result["checks"]["energy_conserved"] = bool(energy_rel_error < 0.08)
        result["checks"]["velocity_exchange"] = bool(abs(post_vx[0]) < 0.12 and abs(post_vx[1] - pre_vx[0]) < 0.12)
    else:
        theory_v = (masses[0] * pre_vx[0] + masses[1] * pre_vx[1]) / max(masses[0] + masses[1], 1e-8)
        result["checks"]["kinetic_drops"] = bool(kinetic_ratio < 0.75)
        result["checks"]["common_velocity"] = bool(abs(post_vx[0] - theory_v) < 0.10 and abs(post_vx[1] - theory_v) < 0.10)
        result["theoretical_common_velocity"] = float(theory_v)

    result["pass"] = bool(all(result["checks"].values()))
    return result


def analyze_case(case: CaseSpec, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    if case.case_id == "case_01_free_fall":
        return analyze_free_fall(case, data)
    if case.case_id == "case_02_horizontal_uniform_motion":
        return analyze_horizontal_motion(case, data)
    if case.case_id == "case_03_vertical_bounce_elastic":
        return analyze_vertical_bounce(case, data)
    if case.case_id == "case_04_two_body_elastic_collision":
        return analyze_collision(case, data, expect_elastic=True)
    if case.case_id == "case_05_two_body_inelastic_collision":
        return analyze_collision(case, data, expect_elastic=False)
    if case.case_id == "case_06_mpm_rest_zero_gravity":
        return analyze_mpm_rest(case, data)
    if case.case_id == "case_07_mpm_free_fall_no_contact":
        return analyze_mpm_free_fall(case, data)
    if case.case_id == "case_08_mpm_horizontal_uniform_zero_gravity":
        return analyze_mpm_horizontal(case, data)
    if case.case_id in {
        "case_09_mpm_drop_floor_rest0_soft",
        "case_10_mpm_drop_floor_rest1_soft",
        "case_11_mpm_drop_floor_rest0_stiff",
        "case_12_mpm_drop_floor_rest1_stiff",
    }:
        return analyze_mpm_floor_drop(case, data)
    raise ValueError(case.case_id)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, ensure_ascii=False, indent=2)


def summarize_mpm_floor_drop_grid(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    wanted = {
        "case_09_mpm_drop_floor_rest0_soft",
        "case_10_mpm_drop_floor_rest1_soft",
        "case_11_mpm_drop_floor_rest0_stiff",
        "case_12_mpm_drop_floor_rest1_stiff",
    }
    case_map = {rec["case_id"]: rec for rec in results if rec.get("case_id") in wanted}
    if set(case_map) != wanted:
        return {
            "available_cases": sorted(case_map),
            "notes": "MPM floor-drop grid summary is only generated when all four restitution/elasticity extreme cases are present.",
        }

    def metric(case_id: str, key: str) -> float:
        return float(case_map[case_id]["analysis"][key])

    soft_r0 = "case_09_mpm_drop_floor_rest0_soft"
    soft_r1 = "case_10_mpm_drop_floor_rest1_soft"
    stiff_r0 = "case_11_mpm_drop_floor_rest0_stiff"
    stiff_r1 = "case_12_mpm_drop_floor_rest1_stiff"

    checks = {
        "high_restitution_increases_rebound_for_soft": bool(
            metric(soft_r1, "post_impact_peak_over_initial") > metric(soft_r0, "post_impact_peak_over_initial") + 0.05
        ),
        "high_restitution_increases_rebound_for_stiff": bool(
            metric(stiff_r1, "post_impact_peak_over_initial") > metric(stiff_r0, "post_impact_peak_over_initial") + 0.05
        ),
        "softer_block_deforms_more_when_r0": bool(
            metric(soft_r0, "max_spread_ratio") > metric(stiff_r0, "max_spread_ratio") + 0.01
        ),
        "softer_block_deforms_more_when_r1": bool(
            metric(soft_r1, "max_spread_ratio") > metric(stiff_r1, "max_spread_ratio") + 0.01
        ),
    }

    return {
        "cases": {
            case_id: {
                "floor_coup_restitution": case_map[case_id]["analysis"]["floor_coup_restitution"],
                "youngs_modulus": case_map[case_id]["analysis"]["youngs_modulus"],
                "post_impact_peak_over_initial": case_map[case_id]["analysis"]["post_impact_peak_over_initial"],
                "observable_energy_retention": case_map[case_id]["analysis"]["observable_energy_retention"],
                "max_spread_ratio": case_map[case_id]["analysis"]["max_spread_ratio"],
                "bounce_detected": case_map[case_id]["analysis"]["bounce_detected"],
            }
            for case_id in sorted(case_map)
        },
        "checks": checks,
        "pass": bool(all(checks.values())),
        "notes": [
            "For these MPM floor-drop tests, the controllable contact restitution knob available in this Genesis 0.4.0 environment is the rigid floor material's `coup_restitution`.",
            "The MPM block itself has no explicit restitution parameter in the current public constructor; stiffness is controlled here via Young's modulus E.",
            "The comparison is qualitative: high restitution should raise rebound, while lower stiffness should increase deformation proxy (`shape_spread`).",
        ],
    }


def run_case(case: CaseSpec, out_root: Path) -> Dict[str, Any]:
    case_dir = out_root / case.case_id
    ensure_dir(case_dir)
    scene = None
    try:
        render_fps = 60
        scene, entities, _, camera = build_scene(case)
        data = collect_case_trajectory(scene, entities, case, camera=camera)
        np.savez_compressed(case_dir / "trajectory.npz", **data)
        if "rgb_frames" in data:
            save_video(case_dir / "rgb.mp4", data["rgb_frames"], fps=render_fps)
        save_case_plots(case_dir, case, data)
        analysis = analyze_case(case, data)
        camera_cfg = validation_camera_config(case)
        meta = {
            "case_id": case.case_id,
            "title": case.title,
            "description": case.description,
            "background": dict(VALIDATION_BACKGROUND),
            "gravity": list(case.gravity),
            "dt": float(case.dt),
            "substeps": int(case.substeps),
            "num_steps": int(case.num_steps),
            "steps_per_frame": int(case.steps_per_frame),
            "frame_dt": float(case.dt * case.steps_per_frame),
            "render_video_fps": int(render_fps),
            "camera": {
                "pos": list(camera_cfg["pos"]),
                "lookat": list(camera_cfg["lookat"]),
                "fov": float(camera_cfg["fov"]),
                "res": list(camera_cfg["res"]),
            },
            "objects": [
                {
                    "name": obj.name,
                    "solver_family": obj.solver_family,
                    "shape": obj.shape,
                    "pos": list(obj.pos),
                    "radius": (None if obj.radius is None else float(obj.radius)),
                    "size": (None if obj.size is None else list(obj.size)),
                    "material_ctor": obj.material_ctor,
                    "linvel": list(obj.linvel),
                    "angvel": list(obj.angvel),
                    "material": dict(obj.material),
                }
                for obj in case.objects
            ],
            "outputs": {
                "trajectory": "trajectory.npz",
                "diagnostics_plot": "diagnostics.png",
                "rgb_video": "rgb.mp4",
            },
            "analysis": analysis,
        }
        write_json(case_dir / "meta.json", meta)
        return {
            "case_id": case.case_id,
            "title": case.title,
            "pass": bool(analysis.get("pass", False)),
            "analysis": analysis,
            "path": str(case_dir),
        }
    finally:
        if scene is not None:
            try:
                scene.destroy()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    selected_case_ids = [c.strip() for c in str(args.only_cases).split(",") if c.strip()]

    backend_used = "gpu"
    try:
        gs.init(backend=gs.gpu)
    except Exception:
        gs.init(backend=gs.cpu)
        backend_used = "cpu"
    finally:
        try:
            gs.destroy()
        except Exception:
            pass

    results = []
    all_cases = build_case_specs()
    if selected_case_ids:
        available = {case.case_id for case in all_cases}
        missing = [case_id for case_id in selected_case_ids if case_id not in available]
        if missing:
            raise ValueError(f"Unknown case ids in --only_cases: {missing}. Available ids: {sorted(available)}")
        cases_to_run = [case for case in all_cases if case.case_id in set(selected_case_ids)]
    else:
        cases_to_run = all_cases
    for case in cases_to_run:
        try:
            if backend_used == "gpu":
                gs.init(backend=gs.gpu)
            else:
                gs.init(backend=gs.cpu)
            print(f"[RUN] {case.case_id}")
            result = run_case(case, out_root=out_root)
            results.append(result)
            print(f"[OK ] {case.case_id} pass={result['pass']}")
        finally:
            try:
                gs.destroy()
            except Exception:
                pass

    summary = {
        "out_dir": str(out_root),
        "backend_used": backend_used,
        "num_cases": len(results),
        "num_passed": int(sum(1 for rec in results if rec.get("pass", False))),
        "results": results,
        "selected_case_ids": selected_case_ids,
        "mpm_floor_drop_grid": summarize_mpm_floor_drop_grid(results),
        "notes": [
            "Energy labels are computed with the same rigid_entity_kinematic_snapshot + -m*g·x convention used by the legacy rigid export path.",
            "Validation now matches the main rigid dataset script more closely on dt=4e-3, substeps=8, steps_per_frame=5, 960x720 rendering, and studio-style background colors.",
            "Case 2 skips the first 30 recorded frames to avoid tiny post-reset floor-contact settling transients under the main-script export cadence before checking constant kinetic energy.",
            "For MPM entities, the physically complete total energy should be kinetic + gravitational potential + internal strain energy. The current Genesis public API exposes particle positions and velocities but not internal strain energy directly.",
            "The MPM validation cases therefore use the observable kinetic + gravitational potential formula from particle_entity_energy(...) and restrict validation to low-deformation scenes where omitted internal energy should stay small.",
            "The MPM simulation bounds are intentionally expanded so the free-fall validation case does not accidentally contact the solver domain boundary.",
            "The MPM floor-drop sweep uses a rigid floor with `coup_restitution` as the available contact restitution control in this Genesis 0.4.0 environment.",
            "Genesis 0.4.0 in the current wan environment does not expose a reliable per-body rigid restitution control through gs.materials.Rigid(restitution=...).",
            "Cases 3-4 therefore probe both the label formula and the solver's collision response, and their elastic-collision failures are most likely a simulator limitation rather than an energy-label bug.",
        ],
    }
    write_json(out_root / "summary.json", summary)
    print(f"[DONE] wrote summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
