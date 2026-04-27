from __future__ import annotations

import json
from pathlib import Path

import genesis as gs
import numpy as np

from genesis_energy_utils import particle_entity_energy, rigid_entity_energy, scene_energy_records


def _close(a: float, b: float, atol: float = 1e-3, rtol: float = 1e-3) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def _assert_close(name: str, value: float, expected: float, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    if not _close(value, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: got {value}, expected {expected}")


def _init_genesis() -> None:
    try:
        gs.init()
    except Exception as exc:
        if "already initialized" not in str(exc).lower():
            raise


def run_rigid_freefall_case() -> dict:
    _init_genesis()
    dt = 0.001
    g = -9.81
    init_z = 0.6
    v0 = -1.2
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, g)),
        show_viewer=False,
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, init_z), radius=0.08),
        material=gs.materials.Rigid(rho=500.0, friction=0.01),
    )
    scene.build()
    ball.set_dofs_velocity((0.0, 0.0, v0, 0.0, 0.0, 0.0))

    energy_series = []
    e0 = rigid_entity_energy(ball)
    energy_series.append(e0.total)
    for _ in range(60):
        scene.step()
        energy_series.append(rigid_entity_energy(ball).total)
    e1 = rigid_entity_energy(ball)
    total_err = abs(e1.total - e0.total)
    energy_series = np.asarray(energy_series, dtype=np.float64)
    return {
        "case": "rigid_freefall",
        "initial": e0.__dict__,
        "final": e1.__dict__,
        "total_energy_abs_error": total_err,
        "framewise_total_energy_max_abs_deviation": float(np.max(np.abs(energy_series - energy_series[0]))),
        "framewise_total_energy_std": float(np.std(energy_series)),
        "num_energy_samples": int(energy_series.shape[0]),
    }


def run_rigid_horizontal_case() -> dict:
    _init_genesis()
    dt = 0.001
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, 0.0)),
        show_viewer=False,
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.4), radius=0.05),
        material=gs.materials.Rigid(rho=700.0, friction=0.01),
    )
    scene.build()
    ball.set_dofs_velocity((0.7, 0.0, 0.0, 0.0, 0.0, 0.0))

    energy_series = []
    e0 = rigid_entity_energy(ball, gravity=(0.0, 0.0, 0.0))
    energy_series.append(e0.total)
    for _ in range(40):
        scene.step()
        energy_series.append(rigid_entity_energy(ball, gravity=(0.0, 0.0, 0.0)).total)
    e1 = rigid_entity_energy(ball, gravity=(0.0, 0.0, 0.0))
    energy_series = np.asarray(energy_series, dtype=np.float64)
    return {
        "case": "rigid_horizontal",
        "initial": e0.__dict__,
        "final": e1.__dict__,
        "kinetic_energy_abs_error": abs(e1.kinetic - e0.kinetic),
        "total_energy_abs_error": abs(e1.total - e0.total),
        "framewise_total_energy_max_abs_deviation": float(np.max(np.abs(energy_series - energy_series[0]))),
        "framewise_total_energy_std": float(np.std(energy_series)),
        "num_energy_samples": int(energy_series.shape[0]),
    }


def run_mpm_rest_case() -> dict:
    _init_genesis()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001, gravity=(0.0, 0.0, 0.0)),
        mpm_options=gs.options.MPMOptions(lower_bound=(-0.5, -0.5, -0.2), upper_bound=(0.5, 0.5, 0.8)),
        show_viewer=False,
    )
    block = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.2), size=(0.18, 0.18, 0.18)),
        material=gs.materials.MPM.Elastic(E=3e4, nu=0.25, rho=120.0, sampler="pbs-8", model="neohooken"),
        surface=gs.surfaces.Default(vis_mode="particle"),
    )
    scene.build()
    e0 = particle_entity_energy(block, gravity=(0.0, 0.0, 0.0))
    for _ in range(5):
        scene.step()
    e1 = particle_entity_energy(block, gravity=(0.0, 0.0, 0.0))
    return {
        "case": "mpm_rest",
        "initial": e0.__dict__,
        "final": e1.__dict__,
        "kinetic_energy_final": e1.kinetic,
        "total_energy_final": e1.total,
    }


def main() -> None:
    out_path = Path("/tmp/genesis_energy_validation.json")

    rigid_freefall = run_rigid_freefall_case()
    _assert_close("rigid_freefall_total", rigid_freefall["final"]["total"], rigid_freefall["initial"]["total"], atol=2e-2, rtol=2e-2)
    if rigid_freefall["framewise_total_energy_max_abs_deviation"] > 2e-2:
        raise AssertionError(
            f"rigid_freefall framewise total energy drift too large: {rigid_freefall['framewise_total_energy_max_abs_deviation']}"
        )

    rigid_horizontal = run_rigid_horizontal_case()
    _assert_close("rigid_horizontal_total", rigid_horizontal["final"]["total"], rigid_horizontal["initial"]["total"], atol=2e-3, rtol=2e-3)
    if rigid_horizontal["framewise_total_energy_max_abs_deviation"] > 2e-3:
        raise AssertionError(
            f"rigid_horizontal framewise total energy drift too large: {rigid_horizontal['framewise_total_energy_max_abs_deviation']}"
        )

    mpm_rest = run_mpm_rest_case()
    if mpm_rest["kinetic_energy_final"] > 1e-4:
        raise AssertionError(f"mpm_rest kinetic energy too large: {mpm_rest['kinetic_energy_final']}")

    report = {
        "results": [rigid_freefall, rigid_horizontal, mpm_rest],
        "scene_energy_records_example": scene_energy_records(rigid_entities=[], particle_entities=[]),
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
