import warp as wp
import warp.torch
import torch
import trimesh
import numpy as np
from torch import Tensor

from pathlib import Path
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import Optional, Union, Sequence, Any
from mpm.warp_utils import from_torch_safe
from mpm.compat_constants import PART_N2I
from mpm.compat_utils import prepare_compat_labels


def sample_vel(cfg, seed: Optional[int] = None):
    if seed is None:
        seed = cfg.seed
    rng = np.random.Generator(np.random.PCG64(seed))

    lin_dir = rng.uniform(-1, 1, size=3)
    if lin_dir[1] > 0:
        lin_dir[1] = -lin_dir[1]
    lin_dir /= np.linalg.norm(lin_dir)

    lin_mag = rng.uniform(*cfg.lin_vel_bound)
    lin_vel = lin_dir * lin_mag

    ang_vel = rng.uniform(*cfg.ang_vel_bound, size=3)

    return lin_vel, ang_vel


def denormalize_points_helper_func(points, size, center) -> torch.Tensor:
    if isinstance(size, np.ndarray):
        size = torch.from_numpy(size).to(points)
    if isinstance(center, np.ndarray):
        center = torch.from_numpy(center).to(points)

    denorm_points = (points.clone() - center) / size

    return denorm_points


def denormalize_points(
    points,
    sections,
    state_init,
) -> torch.Tensor:
    denorm_x = list()
    # we need to take care of the denormalization
    group_x = torch.split(points, sections, dim=0)
    for gd, gx in zip(state_init.groups, group_x):
        denorm_x.append(denormalize_points_helper_func(
            gx, gd.size, gd.center
        ))
    denorm_x = torch.concat(denorm_x, dim=0)

    return denorm_x


@dataclass
class MPMInitData(object):

    clip_bound: float
    span: tuple[int, int]

    num_particles: int
    vol: float

    pos: np.ndarray
    lin_vel: np.ndarray = np.zeros(3)
    ang_vel: np.ndarray = np.zeros(3)
    mat: Optional[np.ndarray] = None
    part: Optional[np.ndarray] = None
    knn_indices: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    ind_vel: Optional[np.ndarray] = None
    ori_bounds: Optional[np.ndarray] = None
    sim_bounds: Optional[np.ndarray] = None
    size: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.center is None:
            self.center = self.pos.mean(0)

    @staticmethod
    def alignment(
        min_bound_1: np.ndarray,
        max_bound_1: np.ndarray,
        min_bound_2: np.ndarray,
        max_bound_2: np.ndarray
    ):
        """Calculate the translation and scale factor to transform bound1 to bound2."""
        # calculate the center of the bounding box
        center_1 = (min_bound_1 + max_bound_1) / 2
        center_2 = (min_bound_2 + max_bound_2) / 2

        # calculate the scale factor along each axis
        scale_factor = (max_bound_2 - min_bound_2) / (max_bound_1 - min_bound_1)

        # calculate the translation
        translation = center_2 - center_1 * scale_factor

        return scale_factor, translation

    @classmethod
    def get(
        cls,
        cfg: DictConfig,
        pos: Optional[np.ndarray] = None,
        vol_all: Optional[float] = None,
        transforms: Optional[DictConfig] = None
    ) -> 'MPMInitData':
        kwargs = cls.get_pcd(
            cfg.shape.name,
            pos,
            vol_all,
            cfg.shape.get('asset_root'),
            cfg.shape.get('sort'),
            cfg.shape.get('ori_bounds'),
            cfg.shape.get('sim_bounds'),
            cfg.get('mat_info_dir'),
            transforms,
        )
        return cls(clip_bound=cfg.clip_bound, span=cfg.span, **kwargs)

    @classmethod
    def get_pcd(
            cls,
            name: str,
            pos: Optional[np.ndarray] = None,
            vol_all: Optional[float] = None,
            asset_root: Optional[str] = None,
            sort: Optional[int] = None,
            ori_bounds: Optional[list] = None,
            sim_bounds: Optional[list] = None,
            mat_info_dir: Optional[str] = None,
            transforms: Optional[DictConfig] = None,
        ) -> dict[str, Any]:

        if ori_bounds is not None:
            ori_bounds = np.array(ori_bounds)
        if sim_bounds is not None:
            sim_bounds = np.array(sim_bounds)

        assert ori_bounds is not None, "ori_bounds must be provided for pcd shape."
        assert sim_bounds is not None, "sim_bounds must be provided for pcd shape."

        if asset_root is None:
            asset_path = Path(__file__).resolve().parent.parent.parent / "experiments" / 'assets'
        else:
            asset_path = Path(asset_root)
        precompute_name = f'{name}'
        precompute_name += '.npz'

        recompute = True

        if (asset_path / precompute_name).is_file():
            file = np.load(asset_path / precompute_name)
            p_x = file['p_x']
            # if the number of particles is the same, 
            # we can reuse the precomputed data
            if p_x.shape[0] == pos.shape[0]:
                vol = file['vol']
                part = file.get('part', None)
                mat = file.get('mat', None)
                knn_indices = file.get('knn_indices', None)
                recompute = False

        if recompute:
            assert pos is not None
            assert vol_all is not None

            p_x = pos.copy()

            if sort is not None:
                indices = np.array(list(sorted(range(p_x.shape[0]), reverse=True, key=lambda x: p_x[:, sort][x])))
                p_x = p_x[indices]

            vol = vol_all / p_x.shape[0]

            if mat_info_dir is None:
                mat_info_dir = Path(asset_root.replace("object_data", "simulation_data"))
            else:
                mat_info_dir = Path(mat_info_dir)
            if (mat_info_dir / 'sampled_points_info.npz').is_file():
                mat_info = prepare_compat_labels(mat_info_dir, p_x)
                part = mat_info['part_labels']
                mat = mat_info['mat_labels']
                knn_indices = mat_info['indices']   # len = final_particles, range = 0...sampled_particles
            else:
                part = None
                mat = None
                knn_indices = None
                print(f'  WARNING: categorical labels for material and part not found.')
            to_save_dict = dict(p_x=p_x, vol=vol)
            if part is not None:
                to_save_dict['part'] = part
            if mat is not None:
                to_save_dict['mat'] = mat
            if knn_indices is not None:
                to_save_dict['knn_indices'] = knn_indices
            np.savez(asset_path / precompute_name, **to_save_dict)

        if transforms is not None:
            for transform in transforms:
                p_x, part, mat, knn_indices = cls._transform_particles(p_x, part, mat, knn_indices, transform)

        # for debugging
        debug_pcd = trimesh.Trimesh(vertices=p_x)
        debug_pcd.export(asset_path / f'{name}_transformed.ply')

        bbmin = ori_bounds[0]
        bbmax = ori_bounds[1]
        sim_bbmin = sim_bounds[0]
        sim_bbmax = sim_bounds[1]
        size, center = cls.alignment(bbmin, bbmax, sim_bbmin, sim_bbmax)

        vol = vol * np.prod(size)
        p_x = p_x * size + center
        p_x = np.ascontiguousarray(p_x.reshape(-1, 3))
        x_min, x_max = p_x[:, 0].min(), p_x[:, 0].max()
        y_min, y_max = p_x[:, 1].min(), p_x[:, 1].max()
        z_min, z_max = p_x[:, 2].min(), p_x[:, 2].max()
        print(f'  [pcd] | num_points: {p_x.shape[0]}')
        print(f"  x: [{x_min}, {x_max}]")
        print(f"  y: [{y_min}, {y_max}]")
        print(f"  z: [{z_min}, {z_max}]")

        assert x_min >= 0.0 and x_max <= 1.0
        assert y_min >= 0.0 and y_max <= 1.0
        assert z_min >= 0.0 and z_max <= 1.0

        return dict(
            num_particles=p_x.shape[0],
            vol=vol, pos=p_x,
            center=center, size=size,
            ori_bounds=ori_bounds,
            sim_bounds=sim_bounds,
            mat=mat, part=part,
            knn_indices=knn_indices,
        )

    @staticmethod
    def _transform_particles(
        p_x: np.ndarray,
        part_labels: Optional[np.ndarray],
        mat_labels: Optional[np.ndarray],
        knn_indices: Optional[np.ndarray],
        transform: DictConfig
    ):
        if transform.get('part_name') is not None:
            assert part_labels is not None
            part = PART_N2I[transform.part_name]
            mask = part_labels == part
        else:
            mask = np.ones(p_x.shape[0], dtype=bool)

        if transform.type == 'translate':
            print(f'[TF-Ps]\tApplying <{transform.type}> with params <{transform.translation}> '
                  f'to <{transform.get("part_name") or "whole object"}> ...')
            translation = np.array(transform.translation)
            p_x[mask] += translation.reshape(1, 3)
        elif transform.type == 'scale':
            print(f'[TF-Ps]\tApplying <{transform.type}> with params <{transform.scale}> '
                  f'to <{transform.get("part_name") or "whole object"}> ...')
            scale = np.array(transform.scale)
            if transform.get('origin') is None:
                origin = np.mean(p_x[mask], axis=0)
            else:
                origin = np.array(transform.origin)
            origin = origin.reshape(1, 3)
            p_x[mask] = (p_x[mask] - origin) * scale + origin
        elif transform.type == 'rotate':
            deg_x = transform.get('degree_x', 0)
            theta_x = np.radians(deg_x)
            deg_y = transform.get('degree_y', 0)
            theta_y = np.radians(deg_y)
            deg_z = transform.get('degree_z', 0)
            theta_z = np.radians(deg_z)
            print(f'[TF-Ps]\tApplying <{transform.type}> with params <[x({deg_x}), y({deg_y}), z({deg_z})]> '
                  f'to <{transform.get("part_name") or "whole object"}> ...')

            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])
            R_y = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ])
            R_z = np.array([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ])

            rotation = R_z @ R_y @ R_x

            if transform.get('origin') is None:
                origin = np.mean(p_x[mask], axis=0)
            else:
                origin = np.array(transform.origin)
            origin = origin.reshape(1, 3)
            p_x[mask] = (p_x[mask] - origin) @ rotation.T + origin
        elif transform.type == 'discard':
            discard_parts = transform.discard_parts
            discard_masks = np.zeros(p_x.shape[0], dtype=bool)
            for discard_part in discard_parts:
                discard_masks |= part_labels == PART_N2I[discard_part]
            p_x = p_x[~discard_masks]
            part_labels = part_labels[~discard_masks] if part_labels is not None else None
            mat_labels = mat_labels[~discard_masks] if mat_labels is not None else None
            knn_indices = knn_indices[~discard_masks] if knn_indices is not None else None
            print(f'[TF-Ps]\tApplying <{transform.type}> with params <{discard_parts}> to discard <{discard_masks.sum()}> particles ...')
        else:
            raise ValueError(f"Unsupported transform type: {transform.type}")

        return p_x, part_labels, mat_labels, knn_indices

    def set_lin_vel(self, value: Union[list, np.ndarray]) -> None:
        self.lin_vel = np.array(value)

    def zero_lin_vel(self) -> None:
        self.set_lin_vel(np.zeros_like(self.lin_vel))

    def set_ang_vel(self, value: Union[list, np.ndarray]) -> None:
        self.ang_vel = np.array(value)

    def zero_ang_vel(self) -> None:
        self.set_ang_vel(np.zeros_like(self.ang_vel))

    def set_ind_vel(self, ind_vel: np.ndarray) -> None:
        self.ind_vel = np.array(ind_vel)


class MPMStateInitializer(object):
    def __init__(self) -> None:
        self.groups: list[MPMInitData] = list()
    
    def add_group(self, group: MPMInitData) -> None:
        self.groups.append(group)
    
    def get_group(self, idx: int) -> MPMInitData:
        return self.groups[idx]
    
    def finalize(self) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, list[int]
    ]:
        """ Finalize the state initializer and return the concatenated data (pos, vol, vel, secs). """

        pos_groups = []
        vol_groups = []
        vel_groups = []
        sections = []

        for group in self.groups:
            pos = group.pos.copy()
            vol = np.zeros_like(pos[:, 0]) + group.vol

            if group.ind_vel is None:
                lin_vel = group.lin_vel.copy()
                ang_vel = group.ang_vel.copy()
                vel = lin_vel + np.cross(ang_vel, pos - group.center)
            else:
                vel = group.ind_vel.copy()

            pos_groups.append(pos)
            vol_groups.append(vol)
            vel_groups.append(vel)
            sections.append(group.num_particles)

        pos_groups = np.concatenate(pos_groups, axis=0)
        vol_groups = np.concatenate(vol_groups, axis=0)
        vel_groups = np.concatenate(vel_groups, axis=0)
        
        assert pos_groups.shape[0] == vol_groups.shape[0] == vel_groups.shape[0], \
            f"pos: {pos_groups.shape[0]}, vol: {vol_groups.shape[0]}, vel: {vel_groups.shape[0]}"
        return pos_groups, vol_groups, vel_groups, sections


@wp.struct
class MPMStateStruct(object):
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3)  # current position
    particle_v: wp.array(dtype=wp.vec3)  # particle velocity
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_F_trial: wp.array(
        dtype=wp.mat33
    )  # apply return mapping on this to obtain elastic def grad
    particle_stress: wp.array(dtype=wp.mat33)  # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33)
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float)  # mass
    particle_density: wp.array(dtype=float)  # density

    particle_selection: wp.array(
        dtype=int
    )  # only particle_selection[p] = 0 will be simulated

    # grid
    grid_m: wp.array(dtype=float, ndim=3)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum/velocity
    grid_v_out: wp.array(
        dtype=wp.vec3, ndim=3
    )  # grid node momentum/velocity, after grid update

    def init(
        self,
        shape: Union[Sequence[int], int],
        device: wp.context.Devicelike = None,
        requires_grad=False,
    ) -> None:
        # shape default is int. number of particles
        self.particle_x = wp.zeros(
            shape, dtype=wp.vec3, device=device, requires_grad=requires_grad
        )
        self.particle_v = wp.zeros(
            shape, dtype=wp.vec3, device=device, requires_grad=requires_grad
        )
        self.particle_F = wp.zeros(
            shape, dtype=wp.mat33, device=device, requires_grad=requires_grad
        )

        self.particle_F_trial = wp.zeros(
            shape, dtype=wp.mat33, device=device, requires_grad=requires_grad
        )

        self.particle_stress = wp.zeros(
            shape, dtype=wp.mat33, device=device, requires_grad=requires_grad
        )
        self.particle_C = wp.zeros(
            shape, dtype=wp.mat33, device=device, requires_grad=requires_grad
        )

        self.particle_vol = wp.zeros(
            shape, dtype=float, device=device, requires_grad=False
        )
        self.particle_mass = wp.zeros(
            shape, dtype=float, device=device, requires_grad=False
        )
        self.particle_density = wp.zeros(
            shape, dtype=float, device=device, requires_grad=False
        )

        self.particle_selection = wp.zeros(
            shape, dtype=int, device=device, requires_grad=False
        )

        # grid: will init later
        self.grid_m = wp.zeros(
            (10, 10, 10), dtype=float, device=device, requires_grad=requires_grad
        )
        self.grid_v_in = wp.zeros(
            (10, 10, 10), dtype=wp.vec3, device=device, requires_grad=requires_grad
        )
        self.grid_v_out = wp.zeros(
            (10, 10, 10), dtype=wp.vec3, device=device, requires_grad=requires_grad
        )

    def init_grid(
        self, grid_res: int, device: wp.context.Devicelike = None, requires_grad=False
    ):
        self.grid_m = wp.zeros(
            (grid_res, grid_res, grid_res),
            dtype=float,
            device=device,
            requires_grad=False,
        )
        self.grid_v_in = wp.zeros(
            (grid_res, grid_res, grid_res),
            dtype=wp.vec3,
            device=device,
            requires_grad=requires_grad,
        )
        self.grid_v_out = wp.zeros(
            (grid_res, grid_res, grid_res),
            dtype=wp.vec3,
            device=device,
            requires_grad=requires_grad,
        )

    def from_torch(
        self,
        tensor_x: Tensor,
        tensor_volume: Tensor,
        tensor_velocity: Optional[Tensor] = None,
        tensor_active: Optional[Tensor] = None,
        n_grid: int = 100,
        grid_lim=1.0,
        device="cuda:0",
        requires_grad=True,
    ):
        num_dim, n_particles = tensor_x.shape[1], tensor_x.shape[0]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        self.init_grid(grid_res=n_grid, device=device, requires_grad=requires_grad)

        if tensor_x is not None:
            self.particle_x = from_torch_safe(
                tensor_x.contiguous().detach().clone(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_volume is not None:
            volume_numpy = tensor_volume.detach().cpu().numpy()
            self.particle_vol = wp.from_numpy(
                volume_numpy, dtype=float, device=device, requires_grad=False
            )

        if tensor_velocity is not None:
            self.particle_v = from_torch_safe(
                tensor_velocity.contiguous().detach().clone(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_active is not None:
            self.particle_selection = from_torch_safe(
                tensor_active.contiguous().detach().clone().type(torch.int),
                dtype=wp.int32,
                requires_grad=False,
            )

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=n_particles,
            inputs=[self.particle_F_trial],
            device=device,
        )
        # initial trial deformation gradient is set to identity

        print("Particles initialized from torch data.")
        print("Total particles: ", n_particles)

    def reset_state(
        self,
        tensor_x: Tensor,
        tensor_velocity: Optional[Tensor] = None,
        tensor_density: Optional[Tensor] = None,
        selection_mask: Optional[Tensor] = None,
        device="cuda:0",
        requires_grad=True,
    ):
        # reset p_c, p_v, p_C, p_F_trial
        num_dim, n_particles = tensor_x.shape[1], tensor_x.shape[0]

        if tensor_x is not None:
            self.particle_x = from_torch_safe(
                tensor_x.contiguous().detach(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_velocity is not None:
            self.particle_v = from_torch_safe(
                tensor_velocity.contiguous().detach().clone(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_density is not None and selection_mask is not None:
            wp_density = from_torch_safe(
                tensor_density.contiguous().detach().clone(),
                dtype=wp.float32,
                requires_grad=False,
            )
            # 1 indicate we need to simulate this particle
            wp_selection_mask = from_torch_safe(
                selection_mask.contiguous().detach().clone().type(torch.int),
                dtype=wp.int32,
                requires_grad=False,
            )

            wp.launch(
                kernel=set_float_vec_to_vec_wmask,
                dim=n_particles,
                inputs=[self.particle_density, wp_density, wp_selection_mask],
                device=device,
            )

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=n_particles,
            inputs=[self.particle_F_trial],
            device=device,
        )
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=n_particles,
            inputs=[self.particle_F],
            device=device,
        )

        wp.launch(
            kernel=set_mat33_to_zero,
            dim=n_particles,
            inputs=[self.particle_C],
            device=device,
        )

        wp.launch(
            kernel=set_mat33_to_zero,
            dim=n_particles,
            inputs=[self.particle_stress],
            device=device,
        )

    def continue_from_torch(
        self,
        tensor_x: Tensor,
        tensor_velocity: Optional[Tensor] = None,
        tensor_F: Optional[Tensor] = None,
        tensor_C: Optional[Tensor] = None,
        device="cuda:0",
        requires_grad=True,
    ):
        if tensor_x is not None:
            self.particle_x = from_torch_safe(
                tensor_x.contiguous().detach(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_velocity is not None:
            self.particle_v = from_torch_safe(
                tensor_velocity.contiguous().detach().clone(),
                dtype=wp.vec3,
                requires_grad=requires_grad,
            )

        if tensor_F is not None:
            self.particle_F_trial = from_torch_safe(
                tensor_F.contiguous().detach().clone(),
                dtype=wp.mat33,
                requires_grad=requires_grad,
            )

        if tensor_C is not None:
            self.particle_C = from_torch_safe(
                tensor_C.contiguous().detach().clone(),
                dtype=wp.mat33,
                requires_grad=requires_grad,
            )

    def set_require_grad(self, requires_grad=True):
        self.particle_x.requires_grad = requires_grad
        self.particle_v.requires_grad = requires_grad
        self.particle_F.requires_grad = requires_grad
        self.particle_F_trial.requires_grad = requires_grad
        self.particle_stress.requires_grad = requires_grad
        self.particle_C.requires_grad = requires_grad

        self.grid_v_out.requires_grad = requires_grad
        self.grid_v_in.requires_grad = requires_grad

    def reset_density(
        self,
        tensor_density: Tensor,
        selection_mask: Optional[Tensor] = None,
        device="cuda:0",
        requires_grad=True,
        update_mass=False,
    ):
        n_particles = tensor_density.shape[0]
        if tensor_density is not None:
            wp_density = from_torch_safe(
                tensor_density.contiguous().detach().clone(),
                dtype=wp.float32,
                requires_grad=False,
            )
        
        if selection_mask is not None:
            # 1 indicate we need to simulate this particle
            wp_selection_mask = from_torch_safe(
                selection_mask.contiguous().detach().clone().type(torch.int),
                dtype=wp.int32,
                requires_grad=False,
            )

            wp.launch(
                kernel=set_float_vec_to_vec_wmask,
                dim=n_particles,
                inputs=[self.particle_density, wp_density, wp_selection_mask],
                device=device,
            )
        else:
            wp.launch(
                kernel=set_float_vec_to_vec,
                dim=n_particles,
                inputs=[self.particle_density, wp_density],
                device=device,
            )

        if update_mass:
            num_particles = self.particle_x.shape[0]
            wp.launch(
                kernel=get_float_array_product,
                dim=num_particles,
                inputs=[
                    self.particle_density,
                    self.particle_vol,
                    self.particle_mass,
                ],
                device=device,
            )

    def partial_clone(self, device="cuda:0", requires_grad=True):
        new_state = MPMStateStruct()
        n_particles = self.particle_x.shape[0]
        new_state.init(n_particles, device=device, requires_grad=requires_grad)

        # clone section:
        # new_state.particle_vol = wp.clone(self.particle_vol, requires_grad=False)
        # new_state.particle_density = wp.clone(self.particle_density, requires_grad=False)
        # new_state.particle_mass = wp.clone(self.particle_mass, requires_grad=False)

        # new_state.particle_selection = wp.clone(self.particle_selection, requires_grad=False)

        # set inactive particles pos to previous ones, but leave active pos to zero
        wp.launch(
            kernel=set_vec3_to_vec3_wmask,
            dim=n_particles,
            inputs=[new_state.particle_x, self.particle_x, self.particle_selection],
            device=device,
        )
        wp.copy(new_state.particle_vol, self.particle_vol)
        wp.copy(new_state.particle_density, self.particle_density)
        wp.copy(new_state.particle_mass, self.particle_mass)
        wp.copy(new_state.particle_selection, self.particle_selection)

        # init grid to zero with grid res.
        new_state.init_grid(
            grid_res=self.grid_v_in.shape[0], device=device, requires_grad=requires_grad
        )

        # init some matrix to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=n_particles,
            inputs=[new_state.particle_F_trial],
            device=device,
        )

        new_state.set_require_grad(requires_grad=requires_grad)
        return new_state


@wp.struct
class MPMModelStruct(object):
    ####### essential #######
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: wp.array(dtype=float)
    lam: wp.array(dtype=float)
    E: wp.array(dtype=float)
    nu: wp.array(dtype=float)
    # bulk: wp.array(dtype=float)           # this can be derived
    elasticity: wp.array(dtype=wp.uint8)    # strain-stress
    plasticity: wp.array(dtype=wp.uint8)    # return mapping

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float)
    friction_angle: wp.array(dtype=float)
    # alpha: float
    gravitational_accelaration: wp.vec3
    hardening: float
    xi: float
    plastic_viscosity: float
    softening: float

    ####### for damping
    rpic_damping: float
    grid_v_damping_scale: float

    def init(
        self,
        shape: int,
        device: wp.context.Devicelike = None,
        requires_grad=False,
    ) -> None:
        self.n_particles = shape

        self.elasticity = wp.zeros(
            shape, dtype=wp.uint8, device=device, requires_grad=requires_grad
        )
        self.plasticity = wp.zeros(
            shape, dtype=wp.uint8, device=device, requires_grad=requires_grad
        )

        self.E = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )  # young's modulus
        self.nu = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )  # poisson's ratio

        self.mu = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )
        self.lam = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )
        # self.bulk = wp.zeros(
        #     shape, dtype=float, device=device, requires_grad=requires_grad
        # )

        self.yield_stress = wp.zeros(
            shape, dtype=float, device=device, requires_grad=requires_grad
        )

    def finalize_mu_lam(self, n_particles, device="cuda:0"):
        wp.launch(
            kernel=compute_mu_lam_from_E_nu_clean,
            dim=n_particles,
            inputs=[self.mu, self.lam, self.E, self.nu],
            device=device,
        )

    def init_other_params(self, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.grid_lim = grid_lim
        self.n_grid = n_grid
        self.grid_dim_x = n_grid
        self.grid_dim_y = n_grid
        self.grid_dim_z = n_grid
        (
            self.dx,
            self.inv_dx,
        ) = self.grid_lim / self.n_grid, float(
            n_grid / grid_lim
        )  # [0-1]?

        # self.plastic_viscosity = wp.zeros(
        #     self.n_particles, dtype=float, device=device, requires_grad=False
        # )
        # self.softening = wp.zeros(
        #     self.n_particles, dtype=float, device=device, requires_grad=False
        # )
        # self.softening.fill_(0.1)
        # self.friction_angle = wp.zeros(
        #     self.n_particles, dtype=float, device=device, requires_grad=False
        # )

        self.plastic_viscosity = 0.0
        self.softening = 0.1
        # self.friction_angle = 25.0
        self.friction_angle = wp.zeros(
            self.n_particles, dtype=float, device=device, requires_grad=False
        )
        self.friction_angle.fill_(25.0)

        self.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)

        self.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic

        self.grid_v_damping_scale = 1.1  # globally applied

    def from_torch(
        self, tensor_E: Tensor, tensor_nu: Tensor, device="cuda:0", requires_grad=False
    ):
        self.E = wp.from_torch(tensor_E.contiguous(), requires_grad=requires_grad)
        self.nu = wp.from_torch(tensor_nu.contiguous(), requires_grad=requires_grad)
        n_particles = tensor_E.shape[0]
        self.finalize_mu_lam(n_particles=n_particles, device=device)

    def set_require_grad(self, requires_grad=True):
        self.E.requires_grad = requires_grad
        self.nu.requires_grad = requires_grad
        self.mu.requires_grad = requires_grad
        self.lam.requires_grad = requires_grad
        self.yield_stress.requires_grad = requires_grad


# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2


@wp.struct
class GridCollider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float
    mask: wp.array(dtype=int, ndim=3)


@wp.struct
class Impulse_modifier:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int

    point: wp.vec3
    size: wp.vec3
    mask: wp.array(dtype=int)


@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float


@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float


@wp.struct
class ParticleVelocityModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3

    start_time: float

    end_time: float

    velocity: wp.vec3

    mask: wp.array(dtype=int)


@wp.kernel
def compute_mu_lam_from_E_nu_clean(
    mu: wp.array(dtype=float),
    lam: wp.array(dtype=float),
    E: wp.array(dtype=float),
    nu: wp.array(dtype=float),
):
    p = wp.tid()
    mu[p] = E[p] / (2.0 * (1.0 + nu[p]))
    lam[p] = E[p] * nu[p] / ((1.0 + nu[p]) * (1.0 - 2.0 * nu[p]))


@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_vec3_to_vec3_wmask(
    source_array: wp.array(dtype=wp.vec3),
    target_array: wp.array(dtype=wp.vec3),
    selection_mask: wp.array(dtype=int),
):
    tid = wp.tid()
    if selection_mask[tid] == 1:
        source_array[tid] = target_array[tid]   


@wp.kernel
def set_vec3_to_vec3(
    source_array: wp.array(dtype=wp.vec3), target_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    source_array[tid] = target_array[tid]


@wp.kernel
def set_float_vec_to_vec_wmask(
    source_array: wp.array(dtype=float),
    target_array: wp.array(dtype=float),
    selection_mask: wp.array(dtype=int),
):
    tid = wp.tid()
    if selection_mask[tid] == 1:
        source_array[tid] = target_array[tid]


@wp.kernel
def set_float_vec_to_vec(
    source_array: wp.array(dtype=float), target_array: wp.array(dtype=float)
):
    tid = wp.tid()
    source_array[tid] = target_array[tid]


@wp.kernel
def set_float_vec_to_vec_with_indices(
    source_array: wp.array(dtype=float),
    target_array: wp.array(dtype=float),
    start_index: int,
    end_index: int,
):
    # NOTE: source_array with shape (a + b) and target_array with shape (b)
    tid = wp.tid()
    if tid >= start_index and tid < end_index:
        source_array[tid] = target_array[tid - start_index]


@wp.kernel
def set_uint_vec_to_vec_with_indices(
    source_array: wp.array(dtype=wp.uint8),
    target_array: wp.array(dtype=wp.uint8),
    start_index: int,
    end_index: int,
):
    # NOTE: source_array with shape (a + b) and target_array with shape (b)
    tid = wp.tid()
    if tid >= start_index and tid < end_index:
        source_array[tid] = target_array[tid - start_index]


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def set_mat33_to_zero(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def set_value_to_float_array_with_indices(
    target_array: wp.array(dtype=float),
    value: float,
    start_index: int,
    end_index: int,
):
    tid = wp.tid()
    if tid >= start_index and tid < end_index:
        target_array[tid] = value


@wp.kernel
def set_value_to_uint_array_with_indices(
    target_array: wp.array(dtype=wp.uint8),
    value: wp.uint8,
    start_index: int,
    end_index: int,
):
    tid = wp.tid()
    if tid >= start_index and tid < end_index:
        target_array[tid] = value


@wp.kernel
def set_warpvalue_to_float_array(
    target_array: wp.array(dtype=float), value: warp.types.float32
):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]


def torch2warp_quat(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 4
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.quat,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_float(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=warp.types.float32,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_vec3(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_mat33(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a
