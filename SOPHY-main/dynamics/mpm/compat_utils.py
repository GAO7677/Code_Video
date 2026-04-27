import math
import torch
import trimesh
import numpy as np
from pathlib import Path
from scipy.stats import mode
from scipy.spatial import cKDTree
from typing import Optional, Literal, Dict, Any
from mpm.mpm_constants import ELASTICITY_DICT, PLASTICITY_DICT
from mpm.compat_constants import (
    MAT_PARAMS_DICT, PART_I2N, USED_PART_I2N, MAX_E, COMB_ID
)

def prepare_compat_labels(info_dir: Path, final_particles: np.ndarray) -> torch.Tensor:

    if not (info_dir / 'sampled_points.ply').exists():
        return None

    sampled_points = trimesh.load(info_dir / 'sampled_points.ply', process=False).vertices
    sampled_infos = np.load(info_dir / 'sampled_points_info.npz')
    sampled_part_labels = sampled_infos['point_part_labels']
    sampled_mat_labels = sampled_infos['point_mat_labels']

    ptree = cKDTree(sampled_points)
    _, indices = ptree.query(final_particles, k=5, workers=2)

    final_part_labels = sampled_part_labels[indices]
    final_part_labels = mode(final_part_labels, axis=1)[0].squeeze()
    final_part_labels = final_part_labels.astype(np.int64)

    final_mat_labels = sampled_mat_labels[indices]
    final_mat_labels = mode(final_mat_labels, axis=1)[0].squeeze()
    final_mat_labels = final_mat_labels.astype(np.int64)

    out = {
        'part_labels': final_part_labels,
        'mat_labels': final_mat_labels,
        'indices': indices,
    }

    return out

def prepare_material_params_given_ranges(
    part_labels: torch.Tensor,
    mat_labels: torch.Tensor,
    device: str,
    info_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    if info_dict is None:
        info_dict = MAT_PARAMS_DICT

    unique_part_labels = torch.sort((torch.unique(part_labels))).values
    elasticity = torch.zeros_like(part_labels, dtype=torch.uint8)
    plasticity = torch.zeros_like(part_labels, dtype=torch.uint8)
    rho = torch.zeros_like(part_labels, dtype=torch.float32) + 1000.    # Density
    E = torch.zeros_like(part_labels, dtype=torch.float32)              # Young's modulus
    nu = torch.zeros_like(part_labels, dtype=torch.float32)             # Poisson's ratio
    sigma = torch.zeros_like(part_labels, dtype=torch.float32)          # Yield stress
    phi = torch.zeros_like(part_labels, dtype=torch.float32)            # Friction angle

    config = dict()

    for part_label in unique_part_labels:
        part_name = PART_I2N[part_label.item()]

        selected_ids = torch.where(part_labels == part_label)[0]
        selected_mat_labels = mat_labels[selected_ids[0]]
        mat_info = info_dict[selected_mat_labels.item()]

        elasticity[selected_ids] = ELASTICITY_DICT[mat_info['elasticity']]
        plasticity[selected_ids] = PLASTICITY_DICT[mat_info['plasticity']]

        E_unit = 10 ** (len(str(int(mat_info['E'][0]))) - 2)
        E_min_scale = math.ceil(mat_info['E'][0] / E_unit)
        E_max_scale = math.floor(mat_info['E'][1] / E_unit)
        E_possible = [E_unit * i for i in range(E_min_scale, E_max_scale + 1)]
        E_selected = np.random.choice(E_possible, 1).item()
        E[selected_ids] = E_selected

        nu_min_scale = math.ceil(mat_info['nu'][0] * 100)
        nu_max_scale = math.floor(mat_info['nu'][1] * 100)
        nu_possible = [i * 0.01 for i in range(nu_min_scale, nu_max_scale + 1)]
        nu_selected = np.random.choice(nu_possible, 1).item()
        nu[selected_ids] = nu_selected

        if 'sigma_y' in mat_info:
            sigma_unit = 10 ** (len(str(int(mat_info['sigma_y'][0]))) - 2)
            sigma_min_scale = math.ceil(mat_info['sigma_y'][0] / sigma_unit)
            sigma_max_scale = math.floor(mat_info['sigma_y'][1] / sigma_unit)
            sigma_possible = [sigma_unit * i for i in range(sigma_min_scale, sigma_max_scale + 1)]
            sigma_selected = np.random.choice(sigma_possible, 1).item()
            sigma[selected_ids] = sigma_selected
        else:
            sigma[selected_ids] = 0.

        if 'phi' in mat_info:
            phi_min_scale = math.ceil(mat_info['phi'][0])
            phi_max_scale = math.floor(mat_info['phi'][1])
            phi_possible = [i for i in range(phi_min_scale, phi_max_scale + 1)]
            phi_selected = np.random.choice(phi_possible, 1).item()
            phi[selected_ids] = phi_selected
        else:
            phi[selected_ids] = 25.

        part_config = {
            'material': mat_info['name'],
            'elasticity': mat_info['elasticity'],
            'plasticity': mat_info['plasticity'],
            'rho': 1000.,
            'E': E_selected,
            'nu': nu_selected,
        }

        if 'sigma_y' in mat_info:
            part_config['sigma_y'] = sigma_selected
        if 'phi' in mat_info:
            part_config['phi'] = phi_selected

        config.update({part_name: part_config})

    out = {
        'elasticity': elasticity.to(device),
        'plasticity': plasticity.to(device),
        'rho': rho.to(device),
        'E': E.to(device),
        'nu': nu.to(device),
        'yield_stress': sigma.to(device),
        'friction_angle': phi.to(device),
        'profile': config
    }

    return out

def prepare_material_params_given_values(
    part_labels: torch.Tensor,
    mat_labels: torch.Tensor,
    device: str,
    info_dict: Dict[str, Any],
    max_e_order: int,
) -> Dict[str, Any]:
    unique_part_labels = torch.sort((torch.unique(part_labels))).values
    elasticity = torch.zeros_like(part_labels, dtype=torch.uint8)
    plasticity = torch.zeros_like(part_labels, dtype=torch.uint8)
    rho = torch.zeros_like(part_labels, dtype=torch.float32) + 1000.    # Density
    E = torch.zeros_like(part_labels, dtype=torch.float32)              # Young's modulus
    nu = torch.zeros_like(part_labels, dtype=torch.float32)             # Poisson's ratio
    sigma = torch.zeros_like(part_labels, dtype=torch.float32)          # Yield stress
    phi = torch.zeros_like(part_labels, dtype=torch.float32)            # Friction angle

    config = dict()

    for part_label in unique_part_labels:
        part_name = PART_I2N[part_label.item()]
        mat_info = info_dict[part_name]

        selected_ids = torch.where(part_labels == part_label)[0]

        elasticity[selected_ids] = ELASTICITY_DICT[mat_info['elasticity']]
        plasticity[selected_ids] = PLASTICITY_DICT[mat_info['plasticity']]

        E_ori = mat_info['E']
        # clamp E to N * 10 ** 8
        E_unit = len(str(int(E_ori))) - 1
        overflow = E_unit - max_e_order
        if overflow > 0:
            E_selected = E_ori / (10 ** overflow)
            print(f'  Clamping "E" from {E_ori:.2e} to {E_selected:.2e} for {part_name}')
        else:
            E_selected = E_ori
        E[selected_ids] = E_selected

        nu_selected = mat_info['nu']
        nu[selected_ids] = nu_selected

        # rho_selected = mat_info['rho']
        # rho[selected_ids] = rho_selected

        if 'sigma_y' in mat_info:
            sigma_ori = mat_info['sigma_y']
            if overflow > 0:
                sigma_selected = sigma_ori / (10 ** overflow)
                print(f'  Clamping "sigma" from {sigma_ori:.2e} to {sigma_selected:.2e} for {part_name}')
            else:
                sigma_selected = sigma_ori
            sigma[selected_ids] = sigma_selected
        else:
            sigma[selected_ids] = 0.

        if 'phi' in mat_info:
            phi_selected = mat_info['phi']
            phi[selected_ids] = phi_selected
        else:
            phi[selected_ids] = 25.

        part_config = {
            'elasticity': mat_info['elasticity'],
            'plasticity': mat_info['plasticity'],
            'rho': 1000.,
            'E': E_selected,
            'nu': nu_selected,
        }

        if 'mmid' in mat_info:
            part_config['mmid'] = mat_info['mmid']
        if 'mat_name' in mat_info:
            part_config['mat_name'] = mat_info['mat_name']
        if 'mat_id' in mat_info:
            part_config['mat_id'] = mat_info['mat_id']
        if 'mat_sub_type' in mat_info:
            part_config['mat_sub_type'] = mat_info['mat_sub_type']
        if 'sigma_y' in mat_info:
            part_config['sigma_y'] = sigma_selected
        if 'phi' in mat_info:
            part_config['phi'] = phi_selected

        config.update({part_name: part_config})

    out = {
        'elasticity': elasticity.to(device),
        'plasticity': plasticity.to(device),
        'rho': rho.to(device),
        'E': E.to(device),
        'nu': nu.to(device),
        'yield_stress': sigma.to(device),
        'friction_angle': phi.to(device),
        'profile': config
    }

    return out


def prepare_material_params_given_npz(
    part_labels: torch.Tensor,
    mat_labels: torch.Tensor,
    device: str,
    info_dict: Dict[str, Any],
    max_e_order: int,
) -> Dict[str, Any]:
    """ Prepare material parameters given npz file [Per Part] """

    assert 'npz_path' in info_dict, 'npz_path must be provided in info_dict'

    npz_path = info_dict['npz_path']
    npz_data = np.load(npz_path)

    unique_part_labels = torch.sort((torch.unique(part_labels))).values

    # for part_label in unique_part_labels:
    #     part_name = USED_PART_I2N[part_label.item()]
    #     print(f'  {part_name}: #for_sim {part_labels.eq(part_label).sum().item()}')
    #     info_str = ""
    elasticity = torch.zeros_like(part_labels, dtype=torch.uint8)
    plasticity = torch.zeros_like(part_labels, dtype=torch.uint8)
    E = torch.zeros_like(part_labels, dtype=torch.float32)              # Young's modulus
    nu = torch.zeros_like(part_labels, dtype=torch.float32)             # Poisson's ratio
    sigma = torch.zeros_like(part_labels, dtype=torch.float32)          # Yield stress
    phi = torch.zeros_like(part_labels, dtype=torch.float32)            # Friction angle

    config = dict()

    # Material Models
    mmid = npz_data['mmid']
    # Material Parameters
    E_log = npz_data['raw_E']   # in log10
    E_ori = 10 ** E_log
    nu_ori = npz_data['raw_nu'] / 2.
    sigma_log = npz_data['raw_sigma']   # in log10
    sigma_ori = 10 ** sigma_log
    phi_rad = npz_data['raw_phi']
    phi_ori = np.rad2deg(phi_rad)

    # Pred Parts
    pred_part_labels = npz_data['point_part_labels']    # same shape with mmid
    pred_part_labels = torch.from_numpy(pred_part_labels)
    print(f'  Pred particles: {pred_part_labels.shape[0]}')

    for part_label in unique_part_labels:
        part_name = USED_PART_I2N[part_label.item()]
        print(f'  {part_name} ({part_label.item()}): #for_sim {part_labels.eq(part_label).sum().item()}, #for_pred {pred_part_labels.eq(part_label).sum().item()}')
        info_str = ""

        # We calculate the average property from the predicted particles,
        # and then assign the average property to the simulation particles.
        # Note that the number of predicted particles may be different from the number of simulation particles.

        selected_ids = torch.where(part_labels == part_label)[0]
        pred_selected_ids = torch.where(pred_part_labels == part_label)[0]
        selected_mmid = mmid[pred_selected_ids]

        selected_mmid_mode = mode(selected_mmid, axis=0)[0].squeeze()
        e_i, p_i = COMB_ID[selected_mmid_mode.item()]
        elasticity[selected_ids] = ELASTICITY_DICT[e_i]
        plasticity[selected_ids] = PLASTICITY_DICT[p_i]
        info_str += f'    {e_i}, {p_i}\n'

        assert E_ori.shape[0] == pred_part_labels.shape[0], f'{E_ori.shape} != {pred_part_labels.shape}'
        selected_E = E_ori[pred_selected_ids]
        selected_E_mean = selected_E.mean().item()
        E_unit = len(str(int(selected_E_mean))) - 1
        overflow = E_unit - max_e_order
        if overflow > 0:
            E_selected = selected_E_mean / (10 ** overflow)
            print(f'  Clamping "E" from {selected_E_mean:.2e} to {E_selected:.2e} for {part_name}')
        else:
            E_selected = selected_E_mean
        info_str += f'    E: {E_selected:.2e}'
        E[selected_ids] = E_selected

        assert nu_ori.shape[0] == pred_part_labels.shape[0], f'{nu_ori.shape} != {pred_part_labels.shape}'
        selected_nu = nu_ori[pred_selected_ids]
        selected_nu_mean = selected_nu.mean().item()
        info_str += f', nu: {selected_nu_mean:.2f}'
        nu[selected_ids] = selected_nu_mean

        if selected_mmid_mode in [1, 2]:
            # need yield stress
            assert sigma_ori.shape[0] == pred_part_labels.shape[0], f'{sigma_ori.shape} != {pred_part_labels.shape}'
            selected_sigma = sigma_ori[pred_selected_ids]
            selected_sigma_mean = selected_sigma.mean().item()
            if overflow > 0:
                sigma_selected = selected_sigma_mean / (10 ** overflow)
                print(f'  Clamping "sigma" from {selected_sigma_mean:.2e} to {sigma_selected:.2e} for {part_name}')
            else:
                sigma_selected = selected_sigma_mean
            info_str += f', sigma: {sigma_selected:.2e}'
            sigma[selected_ids] = sigma_selected
        else:
            sigma[selected_ids] = 0.

        if selected_mmid_mode in [3]:
            # need friction angle
            selected_phi = phi_ori[selected_ids]
            selected_phi_mean = selected_phi.mean().item()
            phi_selected = selected_phi_mean
            info_str += f', phi: {phi_selected:.2f}'
            phi[selected_ids] = selected_phi_mean
        else:
            phi[selected_ids] = 30.

        print(info_str)

        part_config = {
            'elasticity': e_i,
            'plasticity': p_i,
            'rho': 1000.,
            'E': E_selected,
            'nu': selected_nu_mean,
        }

        if selected_mmid_mode in [1, 2]:
            part_config['sigma_y'] = sigma_selected
        if selected_mmid_mode in [3]:
            part_config['phi'] = phi_selected

        config.update({part_name: part_config})

    rho = torch.ones_like(E, dtype=torch.float32) * 1000.

    print(f'E: {E.shape}, {E.min():.3e}, {E.max():.3e}')
    print(f'nu: {nu.shape}, {nu.min()}, {nu.max()}')
    print(f'sigma: {sigma.shape}, {sigma.min():.3e}, {sigma.max():.3e}')
    print(f'elas: {elasticity.shape}, {torch.unique(elasticity)}')
    print(f'plas: {plasticity.shape}, {torch.unique(plasticity)}')

    out = {
        'elasticity': elasticity.to(device),
        'plasticity': plasticity.to(device),
        'rho': rho.to(device),
        'E': E.to(device),
        'nu': nu.to(device),
        'yield_stress': sigma.to(device),
        'friction_angle': phi.to(device),
        'profile': config
    }

    return out

def prepare_material_params(
    part_labels: torch.Tensor,
    mat_labels: torch.Tensor,
    device: str,
    info_dict: Optional[Dict[str, Any]] = None,
    type: Literal['ranges', 'values', 'npz'] = 'ranges',
    max_e_order: Optional[int] = None,
) -> Dict[str, Any]:
    if type == 'ranges':
        return prepare_material_params_given_ranges(part_labels, mat_labels, device, info_dict)
    elif type == 'values':
        assert info_dict is not None
        if max_e_order is None:
            max_e_order = MAX_E
        print(f'  Using max_e_order: {max_e_order}')
        return prepare_material_params_given_values(part_labels, mat_labels, device, info_dict, max_e_order=max_e_order)
    elif type == 'npz':
        assert info_dict is not None
        if max_e_order is None:
            max_e_order = MAX_E
        print(f'  Using max_e_order: {max_e_order}')
        return prepare_material_params_given_npz(part_labels, mat_labels, device, info_dict, max_e_order=max_e_order)
    else:
        raise ValueError(f'Invalid type: {type}, expected one of ["ranges", "values", "npz"]')
