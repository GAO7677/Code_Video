import json
import torch
import warp as wp
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from typing import List, Optional
from mpm.mpm_solver_diff import MPMWARPDiff
from mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
    MPMInitData,
    MPMStateInitializer,
    sample_vel,
    denormalize_points_helper_func
)
from mpm.compat_constants import PART_N2I
from mpm.mpm_constants import ELASTICITY_DICT, PLASTICITY_DICT
from mpm.compat_utils import prepare_compat_labels, prepare_material_params
from util.dataprep import prepare_simulation_data_given_mesh


def array_to_tensor(*array, device, dtype):
    return [torch.from_numpy(a).to(device=device, dtype=dtype) for a in array]


class PhysObject:
    """ Physical objects with mesh representation """
    def __init__(self, config: DictConfig, device: str) -> None:
        self.config = config
        obj_root: Path = Path(config.path)
        obj_name = config.name
        prompt = f'[OBJ | {obj_name}]\t'
        print(f'\n{prompt}Loading data from {obj_root} ...')

        # 1. Prepare mesh representation (volumetric points + mesh vertices)

        vertices, particles, vol_all, kal_mesh = prepare_simulation_data_given_mesh(
            particles_path=Path(config.particles.particles_path) if config.particles.get('particles_path') is not None else None,
            mesh_path=Path(config.particles.mesh_path),
            mesh_sample_mode=config.particles.mesh_sample_mode,
            mesh_sample_resolution=config.particles.mesh_sample_resolution,
            save_dir=obj_root,
            particles_downsample_factor=config.particles.get('downsample_factor', 1),
        )

        self.kal_mesh = kal_mesh.to(device=device)

        # 2. Load physical particles

        # NOTE: currently assumed a large time span for the particles
        config.particles.span = [0, 10**7]
        config.particles.shape.asset_root = config.path
        config.particles.shape.name = "particles"
        config.particles.mat_info_dir = config.material.get('mat_info_dir')     # NOTE: Updated here
        # NOTE: w/o transformation
        comb_pos = np.concatenate([vertices, particles], axis=0)
        init_data = MPMInitData.get(
            config.particles,
            pos=comb_pos,
            vol_all=vol_all,
            transforms=config.get("transforms")     # Do transformation for the particles
        )

        # 3. Do transformation for the mesh vertices AFTER dumping init_data

        if config.get('transforms') is not None:
            # get vertices
            mesh_verts = self.kal_mesh.vertices.cpu().numpy().copy()
            print(f'{prompt}Transform mesh vertices ...')
            print(mesh_verts[:, 0].min(), mesh_verts[:, 0].max())
            print(mesh_verts[:, 1].min(), mesh_verts[:, 1].max())
            print(mesh_verts[:, 2].min(), mesh_verts[:, 2].max())
            for transform in config.get("transforms"):
                mesh_verts = MPMInitData._transform_particles(mesh_verts, None, None, None, transform)[0]
            self.kal_mesh.vertices = torch.from_numpy(mesh_verts).float().to(device=device)
            print(self.kal_mesh.vertices[:, 0].min(), self.kal_mesh.vertices[:, 0].max())
            print(self.kal_mesh.vertices[:, 1].min(), self.kal_mesh.vertices[:, 1].max())
            print(self.kal_mesh.vertices[:, 2].min(), self.kal_mesh.vertices[:, 2].max())

        # 4. Prepare indices

        mv_indices = [1] * vertices.shape[0] + [0] * (init_data.pos.shape[0] - vertices.shape[0])
        self.mv_indices = torch.tensor(mv_indices, device=device, dtype=torch.bool)
        print(f'{prompt}Num mesh vertices: {self.mv_indices.sum()}')

        # 4. Initialize velocity
        if config.particles.get("vel") is not None:
            lin_vel = np.array(config.particles.vel.lin_vel)
            ang_vel = np.array(config.particles.vel.ang_vel)
            print(f'{prompt}Use initial velocity: {config.particles.vel} ...')
        else:
            lin_vel, ang_vel = sample_vel(seed=42)
            print(f'Randomly sample initial velocity ...')
        init_data.set_lin_vel(lin_vel)
        init_data.set_ang_vel(ang_vel)

        self.init_data = init_data


def prepare_materials(
    objects: List[PhysObject],
    state_initializer: MPMStateInitializer,
    mpm_model: MPMModelStruct,
    mpm_state: MPMStateStruct,
    mpm_solver: MPMWARPDiff,
    device: str,
    max_e_order: Optional[int] = None,
):
    print('Preparing material parameters ...')
    wp_device = wp.get_device(device)

    start_idx = 0
    for obj_idx in range(len(objects)):

        obj_group = state_initializer.get_group(obj_idx)
        obj_point = obj_group.pos
        obj_config = objects[obj_idx].config

        mat_info_dir = obj_config.material.get(
            'mat_info_dir', obj_config.path.replace("object_data", "simulation_data")
        )
        generated = obj_config.material.get("generated", None)
        part_mat_info = obj_config.material.get("part_mat_params", None)
        part_labels = obj_group.part
        mat_labels = obj_group.mat

        if generated is not None and generated:
            info_dict = {
                "npz_path": Path(mat_info_dir) / 'sampled_points_info.npz',
                "knn_indices": obj_group.knn_indices,
                'obj_idx': obj_idx,
            }
            part_labels = torch.from_numpy(part_labels).to(dtype=torch.long)
            mat_params = prepare_material_params(
                part_labels, None, device=device, info_dict=info_dict, type='npz', max_e_order=max_e_order
            )

            elasticity = mat_params.pop('elasticity')
            plasticity = mat_params.pop('plasticity')
            profile = mat_params.pop('profile', None)

            if profile is not None:
                with open(Path(obj_config.path) / 'mat_params_debug.json', 'w') as f:
                    json.dump(profile, f, indent=4)
        elif part_labels is not None and mat_labels is not None:
            part_labels = torch.from_numpy(part_labels).to(dtype=torch.long)
            mat_labels = torch.from_numpy(mat_labels).to(dtype=torch.long)

            if part_mat_info is not None:
                if isinstance(part_mat_info, str):
                    selected_file = Path(mat_info_dir) / f'{part_mat_info}.json'
                    print(f'  Loading material info from [{selected_file.name}] ...')
                    with open(selected_file, 'r') as f:
                        part_mat_info = json.load(f)
                mat_params = prepare_material_params(
                    part_labels, mat_labels, device=device, info_dict=part_mat_info, type='values', max_e_order=max_e_order
                )
            else:
                mat_params = prepare_material_params(
                    part_labels, mat_labels, device=device
                )

            elasticity = mat_params.pop('elasticity')
            plasticity = mat_params.pop('plasticity')
            profile = mat_params.pop('profile')

            particles_ori = denormalize_points_helper_func(
                torch.tensor(obj_point), obj_group.size, obj_group.center
            )

            # debug
            np.savez_compressed(
                Path(obj_config.path) / 'mat_params_debug.npz',
                points=particles_ori.numpy(),
                part_labels=part_labels.numpy(),
                mat_labels=mat_labels.numpy(),
            )

            with open(Path(obj_config.path) / 'mat_params_debug.json', 'w') as f:
                json.dump(profile, f, indent=4)
        else:
            mat_params = obj_config.material
            print(f'  {obj_config.name} -> {mat_params}')
            elasticity = int(ELASTICITY_DICT[obj_config.material.pop('elasticity')])
            plasticity = int(PLASTICITY_DICT[obj_config.material.pop('plasticity')])

        print(f'[OBJ | {obj_config.name}]\tInit material parameters from {start_idx} to {start_idx + obj_point.shape[0]}')
        mpm_solver.set_material_property(
            mpm_model, mpm_state,
            elasticity, plasticity,
            start_idx=start_idx, end_idx=start_idx + obj_point.shape[0],
            device=wp_device,
            **mat_params,
        )

        start_idx += obj_point.shape[0]
    print('Material parameters done.')


def prepare_simulation_environment_given_mesh(
    num_grids: int,
    gravity: List[float],
    objects: List[PhysObject],
    device: str,
    max_e_order: Optional[int] = None,
    requires_grad: bool = False,
):
    torch_device = torch.device(device)
    wp_device = wp.get_device(device)

    state_initializer = MPMStateInitializer()

    obj_kal_meshes = list()
    obj_mv_indices = list()

    for obj in objects:

        obj_kal_meshes.append(obj.kal_mesh)
        obj_mv_indices.append(obj.mv_indices)
        state_initializer.add_group(obj.init_data)

    pos_all, vol_all, vel_all, sections = state_initializer.finalize()
    pos_all, vol_all, vel_all = array_to_tensor(
        pos_all, vol_all, vel_all, device=torch_device, dtype=torch.float32
    )

    mpm_state = MPMStateStruct()
    print(f'==> sum sections: {sum(sections)}')
    mpm_state.init(sum(sections), device=wp_device, requires_grad=requires_grad)

    mpm_state.from_torch(
        tensor_x=pos_all,
        tensor_volume=vol_all,
        tensor_velocity=vel_all,
        n_grid=num_grids,
        grid_lim=1.0,
        device=wp_device,
        requires_grad=requires_grad
    )

    mpm_model = MPMModelStruct()
    mpm_model.init(sum(sections), device=wp_device, requires_grad=requires_grad)
    mpm_model.init_other_params(n_grid=num_grids, grid_lim=1.0, device=wp_device)

    mpm_solver = MPMWARPDiff(sum(sections), n_grid=num_grids, grid_lim=1.0, device=wp_device)

    prepare_materials(
        objects=objects,
        state_initializer=state_initializer,
        mpm_model=mpm_model,
        mpm_state=mpm_state,
        mpm_solver=mpm_solver,
        device=device,
        max_e_order=max_e_order,
    )

    env_params = {
        'g': gravity,
        'grid_v_damping_scale': 1.1,    # no damping if > 1.0
    }

    mpm_solver.set_parameters_dict(mpm_model, mpm_state, env_params, device=wp_device)
    mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=wp_device)
    mpm_solver.prepare_mass(mpm_state, device=wp_device)

    mpm_solver.add_bounding_box(freeslip=False)

    out = {
        'mpm_solver': mpm_solver,
        'mpm_model': mpm_model,
        'mpm_state': mpm_state,
        'sections': sections,
        'state_initializer': state_initializer,
        'obj_kal_meshes': obj_kal_meshes,
        'obj_mv_indices': obj_mv_indices,
    }

    return out


# NOTE: 'points' in original space, will then do normalization in this func
def prepare_boundary_conditions_given_mesh(
    obj_mv_indices: List[np.ndarray],
    state_initializer: MPMStateInitializer,
    bc_configs: DictConfig,
    mpm_state: MPMStateStruct,
    mpm_solver: MPMWARPDiff,
    num_grids: int,
    dt: float,
    device: str,
):
    bc_outputs = dict()

    for bc_config in bc_configs:
        if bc_config.type == 'force':
            object_idx = bc_config.get('object_idx', 0)
            group = state_initializer.get_group(object_idx)
            start_frame = bc_config.get('start_frame', 0)
            num_frames = bc_config.num_frames

            force = torch.tensor(bc_config.force, device=device)
            center = torch.tensor(bc_config.center, device=device)
            radius = bc_config.get('radius', 0.1)
            
            pos_ori = denormalize_points_helper_func(torch.tensor(group.pos, device=device), group.size, group.center)
            dist = torch.norm(pos_ori - center.unsqueeze(0), dim=-1)

            apply_force_mask = dist < radius
            apply_force_mask = apply_force_mask.type(torch.int)

            print(
                f'[BC | Force]\t'
                f'Applying force <{bc_config.force}> from <f{start_frame}> to <f{start_frame + num_frames}>, '
                f'#points: <{sum(apply_force_mask)}> ...'
            )

            start_time = start_frame * dt
            end_time = (start_frame + num_frames) * dt

            mpm_solver.add_impulse_on_particles_with_mask(
                mpm_state,
                force,
                dt,
                apply_force_mask,
                start_time=start_time,
                end_time=end_time,
                device=device,
            )

            verts_particles = state_initializer.get_group(object_idx).pos
            verts_particles = torch.from_numpy(verts_particles).float().to(device=device)
            verts = verts_particles[obj_mv_indices[object_idx]]
            dist = torch.norm(verts - center.unsqueeze(0), dim=-1)
            closest_idx = torch.argmin(dist)

            bc_outputs['render_force'] = bc_config.get('render_force', False)
            render_force_info = {
                'object_idx': object_idx,
                'closest_kernel_idx': closest_idx.item(),
                'force': force,
                'start_frame': start_frame,
                'num_frames': num_frames
            }
            if 'render_force_infos' not in bc_outputs:
                bc_outputs['render_force_infos'] = list()
            bc_outputs['render_force_infos'].append(render_force_info)

        elif bc_config.type == 'wind':
            object_idx = bc_config.get('object_idx', 0)
            group = state_initializer.get_group(object_idx)
            start_frame = bc_config.get('start_frame', 0)
            num_frames = bc_config.num_frames

            max_force = bc_config.max_force
            point = np.array(bc_config.point).astype(np.float32)
            size = np.array(bc_config.size).astype(np.float32)

            # normalized
            point = point * group.size + group.center
            size = size * group.size
            point = torch.from_numpy(point).to(device=device)
            size = torch.from_numpy(size).to(device=device)

            frequency = 1. / num_frames
            omega = np.pi * frequency

            print(f'[BC | Wind]\tApplying wind <{bc_config.max_force}> from <f{start_frame}> to <f{start_frame + num_frames}> ...')

            sub_frames = num_frames // 100
            for step in range(0, num_frames, sub_frames):
                cur_frame = start_frame + step
                wind_force = [max_force[i] * np.sin(omega * step) for i in range(3)]
                wind_force = torch.tensor(wind_force, device=device)

                # print(f'[BC | Wind | Debug]\t <{wind_force}> at frame <{cur_frame}> for <{sub_frames}>...')

                start_time = cur_frame * dt
                end_time = (cur_frame + sub_frames) * dt

                mpm_solver.add_impulse_on_particles(
                    mpm_state,
                    wind_force,
                    dt,
                    point=point,
                    size=size,
                    end_time=end_time,
                    start_time=start_time,
                    device=device,
                )

        elif bc_config.type == 'particle_velocity_transition':
            object_idx = bc_config.get('object_idx', 0)
            group = state_initializer.get_group(object_idx)
            start_frame = bc_config.get('start_frame', 0)
            num_frames = bc_config.num_frames

            velocity = torch.tensor(bc_config.velocity, device=device)
            point = np.array(bc_config.point).astype(np.float32)
            size = np.array(bc_config.size).astype(np.float32)

            # normalized
            point = point * group.size + group.center
            size = size * group.size
            point = torch.from_numpy(point).to(device=device)
            size = torch.from_numpy(size).to(device=device)

            print(f'[BC | PaVelTr]\tApplying velocity <{velocity}> from <f{start_frame}> to <f{start_frame + num_frames}> ...')

            start_time = start_frame * dt
            end_time = (start_frame + num_frames) * dt

            mpm_solver.enforce_particle_velocity_translation(
                mpm_state,
                point,
                size,
                velocity,
                start_time=start_time,
                end_time=end_time,
                device=device,
            )

        elif bc_config.type == 'floor':
            object_idx = bc_config.get('object_idx', 0)
            group = state_initializer.get_group(object_idx)
            start_frame = bc_config.get('start_frame', 0)
            num_frames = bc_config.num_frames

            normal = np.array(bc_config.normal).astype(np.float32)
            point = bc_config.get('point')
            surface = bc_config.surface
            friction = bc_config.get('friction', 0.0)

            if point is None:
                axis = np.where(normal==1)[0].item()
                pos = group.pos
                point_idx = np.argmin(pos[:, axis])
                # the distance (w.r.t. the normal) from the point to the floor
                height = bc_config.get('height', 1e-3)
                point = pos[point_idx] - normal * height
                point[axis] = max(point[axis], 0.)
            else:
                # normalized
                point = np.array(point).astype(np.float32)
                point = point * group.size + group.center

            print(f'[BC | Floor]\tApplying floor at <{point}> with normal <{normal}> from <f{start_frame}> to <f{start_frame + num_frames}> ...')

            bc_outputs['floor_info'] = {
                "point": point,
                "normal": normal,
            }

            normal = torch.from_numpy(normal).to(device)
            point = torch.from_numpy(point).to(device)

            start_time = start_frame * dt
            end_time = (start_frame + num_frames) * dt

            mpm_solver.add_surface_collider(
                point, normal, surface, friction, start_time=start_time, end_time=end_time
            )

        elif bc_config.type == 'freeze':
            object_idx = bc_config.get('object_idx', 0)
            group = state_initializer.get_group(object_idx)
            start_frame = bc_config.get('start_frame', 0)
            num_frames = bc_config.get('num_frames', 1e9)

            part_name = bc_config.part_name
            part_id = PART_N2I[part_name]

            pos = group.pos
            part_labels = group.part
            selected_ids = np.where(part_labels == part_id)[0]

            grid_pts_cnt = torch.zeros(
                (num_grids, num_grids, num_grids), device=device, dtype=torch.int32
            )

            dx = 1.0 / num_grids
            inv_dx = 1.0 / dx

            freeze_pos = (pos[selected_ids] * inv_dx).astype(np.int64)

            for x, y, z in freeze_pos:
                grid_pts_cnt[x, y, z] += 1

            mask = grid_pts_cnt >=1
            mask = mask.type(torch.int32)

            print(f'[BC | Freeze]\tFreezing {mask.sum().item()} grids for part <{part_name}> from <f{start_frame}> to <f{start_frame + num_frames}> ...')

            start_time = start_frame * dt
            end_time = (start_frame + num_frames) * dt

            mpm_solver.enforce_grid_velocity_by_mask(mask, end_time=end_time, start_time=start_time)

        else:
            print(f'[BC]\t\tWarning! Unsupported boundary condition type: {bc_config.type}')

    return bc_outputs