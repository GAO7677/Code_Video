from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :]
    return arr


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass
class EnergyBreakdown:
    kinetic: float
    potential: float
    total: float


@dataclass
class KinematicSnapshot:
    com_pos: np.ndarray
    linear_vel: np.ndarray
    angular_vel: np.ndarray
    kinetic: float


def rigid_link_energy(link: Any, gravity: Iterable[float] = (0.0, 0.0, -9.81)) -> EnergyBreakdown:
    gravity = np.asarray(tuple(gravity), dtype=np.float64)
    mass = float(link.get_mass())
    pos = np.asarray(_to_numpy(link.get_pos()), dtype=np.float64).reshape(3)
    vel = np.asarray(_to_numpy(link.get_vel()), dtype=np.float64).reshape(3)
    ang = np.asarray(_to_numpy(link.get_ang()), dtype=np.float64).reshape(3)

    kinetic_linear = 0.5 * mass * float(np.dot(vel, vel))

    inertia_local = getattr(link, "inertial_i", None)
    link_quat = np.asarray(_to_numpy(link.get_quat()), dtype=np.float64).reshape(4)
    inertial_quat = getattr(link, "inertial_quat", None)
    if inertia_local is None:
        kinetic_angular = 0.0
    else:
        if inertial_quat is None:
            inertial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            inertial_quat = np.asarray(inertial_quat, dtype=np.float64).reshape(4)
        world_inertial_quat = _quat_mul_wxyz(link_quat, inertial_quat)
        rot = _quat_to_rotmat_wxyz(world_inertial_quat)
        inertia_world = rot @ np.asarray(inertia_local, dtype=np.float64) @ rot.T
        kinetic_angular = 0.5 * float(ang @ inertia_world @ ang)

    potential = -mass * float(np.dot(gravity, pos))
    kinetic = kinetic_linear + kinetic_angular
    return EnergyBreakdown(kinetic=kinetic, potential=potential, total=kinetic + potential)


def rigid_entity_energy(entity: Any, gravity: Iterable[float] = (0.0, 0.0, -9.81)) -> EnergyBreakdown:
    total_k = 0.0
    total_p = 0.0
    for link in getattr(entity, "links", []):
        rec = rigid_link_energy(link, gravity=gravity)
        total_k += rec.kinetic
        total_p += rec.potential
    return EnergyBreakdown(kinetic=total_k, potential=total_p, total=total_k + total_p)


def rigid_entity_kinematic_snapshot(entity: Any, gravity: Iterable[float] = (0.0, 0.0, -9.81)) -> KinematicSnapshot:
    links = list(getattr(entity, "links", []))
    if not links:
        pos = np.asarray(_to_numpy(entity.get_pos()), dtype=np.float64).reshape(3)
        vel = np.asarray(_to_numpy(entity.get_vel()), dtype=np.float64).reshape(3)
        ang = np.asarray(_to_numpy(entity.get_ang()), dtype=np.float64).reshape(3)
        energy = rigid_entity_energy(entity, gravity=gravity).kinetic
        return KinematicSnapshot(com_pos=pos, linear_vel=vel, angular_vel=ang, kinetic=energy)

    link_coms = []
    link_masses = []
    try:
        links_vel = _to_numpy(entity.get_links_vel(ref="link_com"))
        if np.asarray(links_vel).ndim == 3:
            links_vel = np.asarray(links_vel)[0]
    except Exception:
        links_vel = None
    try:
        links_ang = _to_numpy(entity.get_links_ang())
        if np.asarray(links_ang).ndim == 3:
            links_ang = np.asarray(links_ang)[0]
    except Exception:
        links_ang = None

    for idx, link in enumerate(links):
        mass = float(link.get_mass())
        link_pos = np.asarray(_to_numpy(link.get_pos()), dtype=np.float64).reshape(3)
        link_quat = np.asarray(_to_numpy(link.get_quat()), dtype=np.float64).reshape(4)
        inertial_pos = getattr(link, "inertial_pos", None)
        if inertial_pos is None:
            com_pos = link_pos
        else:
            com_pos = link_pos + _quat_to_rotmat_wxyz(link_quat) @ np.asarray(inertial_pos, dtype=np.float64).reshape(3)
        link_coms.append(com_pos)
        link_masses.append(mass)

    masses = np.asarray(link_masses, dtype=np.float64)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        total_mass = 1.0
    coms = np.asarray(link_coms, dtype=np.float64)
    com_pos = np.sum(coms * masses[:, None], axis=0) / total_mass

    if links_vel is None:
        linear_vel = np.sum(np.stack([_to_numpy(link.get_vel()) for link in links], axis=0) * masses[:, None], axis=0) / total_mass
    else:
        linear_vel = np.sum(np.asarray(links_vel, dtype=np.float64) * masses[:, None], axis=0) / total_mass

    if links_ang is None:
        angular_vel = np.sum(np.stack([_to_numpy(link.get_ang()) for link in links], axis=0) * masses[:, None], axis=0) / total_mass
    else:
        angular_vel = np.sum(np.asarray(links_ang, dtype=np.float64) * masses[:, None], axis=0) / total_mass

    energy = rigid_entity_energy(entity, gravity=gravity).kinetic
    return KinematicSnapshot(
        com_pos=np.asarray(com_pos, dtype=np.float64).reshape(3),
        linear_vel=np.asarray(linear_vel, dtype=np.float64).reshape(3),
        angular_vel=np.asarray(angular_vel, dtype=np.float64).reshape(3),
        kinetic=float(energy),
    )


def particle_entity_energy(entity: Any, gravity: Iterable[float] = (0.0, 0.0, -9.81)) -> EnergyBreakdown:
    gravity = np.asarray(tuple(gravity), dtype=np.float64)
    pos = _ensure_2d(_to_numpy(entity.get_particles_pos()))
    vel = _ensure_2d(_to_numpy(entity.get_particles_vel()))
    total_mass = None
    try:
        total_mass_raw = entity.get_mass()
        total_mass = float(np.asarray(_to_numpy(total_mass_raw)).reshape(-1)[0])
    except Exception:
        total_mass = None
    if total_mass is None:
        solver = getattr(entity, "solver", None)
        particle_start = int(getattr(entity, "_particle_start", getattr(entity, "particle_start", 0)))
        n_particles = int(getattr(entity, "n_particles"))
        if solver is None or not hasattr(solver, "particles_info"):
            raise RuntimeError("Cannot recover particle masses from entity.")
        info = solver.particles_info
        if hasattr(info, "mass"):
            mass_field = _to_numpy(info.mass.to_numpy() if hasattr(info.mass, "to_numpy") else info.mass)
            total_mass = float(np.asarray(mass_field[particle_start : particle_start + n_particles], dtype=np.float64).sum())
        else:
            raise RuntimeError("Cannot access solver.particles_info.mass for particle entity.")
    if pos.shape[0] == 0:
        return EnergyBreakdown(kinetic=0.0, potential=0.0, total=0.0)
    particle_mass = total_mass / float(pos.shape[0])
    speed_sq = np.sum(vel * vel, axis=1)
    kinetic = 0.5 * particle_mass * float(np.sum(speed_sq))
    potential = -particle_mass * float(np.sum(pos @ gravity))
    return EnergyBreakdown(kinetic=kinetic, potential=potential, total=kinetic + potential)


def particle_entity_kinematic_snapshot(entity: Any, gravity: Iterable[float] = (0.0, 0.0, -9.81)) -> KinematicSnapshot:
    pos = _ensure_2d(_to_numpy(entity.get_particles_pos()))
    vel = _ensure_2d(_to_numpy(entity.get_particles_vel()))
    if pos.shape[0] == 0:
        return KinematicSnapshot(
            com_pos=np.zeros(3, dtype=np.float64),
            linear_vel=np.zeros(3, dtype=np.float64),
            angular_vel=np.zeros(3, dtype=np.float64),
            kinetic=0.0,
        )
    com_pos = np.mean(pos, axis=0)
    linear_vel = np.mean(vel, axis=0)
    centered_pos = pos - com_pos[None, :]
    centered_vel = vel - linear_vel[None, :]
    inertia_like = centered_pos.T @ centered_pos
    rhs = np.sum(np.cross(centered_pos, centered_vel), axis=0)
    try:
        angular_vel = np.linalg.solve(inertia_like + 1e-8 * np.eye(3, dtype=np.float64), rhs)
    except np.linalg.LinAlgError:
        angular_vel = np.zeros(3, dtype=np.float64)
    energy = particle_entity_energy(entity, gravity=gravity).kinetic
    return KinematicSnapshot(
        com_pos=np.asarray(com_pos, dtype=np.float64).reshape(3),
        linear_vel=np.asarray(linear_vel, dtype=np.float64).reshape(3),
        angular_vel=np.asarray(angular_vel, dtype=np.float64).reshape(3),
        kinetic=float(energy),
    )


def scene_energy_records(
    rigid_entities: Iterable[Any] | None = None,
    particle_entities: Iterable[Any] | None = None,
    gravity: Iterable[float] = (0.0, 0.0, -9.81),
) -> Dict[str, Any]:
    rigid_entities = list(rigid_entities or [])
    particle_entities = list(particle_entities or [])
    rigid_records: List[Dict[str, Any]] = []
    particle_records: List[Dict[str, Any]] = []
    total_k = 0.0
    total_p = 0.0

    for entity in rigid_entities:
        rec = rigid_entity_energy(entity, gravity=gravity)
        rigid_records.append(
            {
                "name": getattr(entity, "name", None),
                "kinetic": rec.kinetic,
                "potential": rec.potential,
                "total": rec.total,
            }
        )
        total_k += rec.kinetic
        total_p += rec.potential

    for entity in particle_entities:
        rec = particle_entity_energy(entity, gravity=gravity)
        particle_records.append(
            {
                "name": getattr(entity, "name", None),
                "kinetic": rec.kinetic,
                "potential": rec.potential,
                "total": rec.total,
            }
        )
        total_k += rec.kinetic
        total_p += rec.potential

    return {
        "rigid_entities": rigid_records,
        "particle_entities": particle_records,
        "kinetic": total_k,
        "potential": total_p,
        "total": total_k + total_p,
        "notes": {
            "total_definition": "kinetic + gravitational potential",
            "excludes": ["MPM/FEM internal strain energy", "damping loss", "contact dissipation"],
        },
    }
