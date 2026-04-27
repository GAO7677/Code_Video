import os
import json

import torch
from torch.utils import data
from safetensors import safe_open
from typing import Optional, Tuple, Literal

import numpy as np

MAT_FILE_NAME = 'mat_params_new_v3.4.json'

category_n2i = {
    'bag': 0,
    'bed': 1,
    'chair': 2,
    'crib': 3,
    'hat': 4,
    'headband': 5,
    'love_seat': 6,
    'pillow': 7,
    'planter': 8,
    'sofa': 9,
    'teddy_bear': 10,
    'vase': 11,
}

category_i2n = {v: k for k, v in category_n2i.items()}

dirname = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(dirname, 'metadata')

class D3CoMPaT200(data.Dataset):
    def __init__(
        self,
        root: str,
        split: Literal['train', 'valid', 'test']='train',
        categories: Optional[Tuple[str]]=None,
        transform=None,
        sampling: bool=True,
        num_samples: int=4096,
        return_surface: bool=True,
        surface_sampling: bool=True,
        pc_size: int=2048,
        scaling: float=1.0,
        use_empty_for_mat: bool=False,
        use_empty_for_part: bool=False,
        cond_signal: Optional[Literal['text', 'image', 'category']] = None,
        cond_signal_version: Optional[str] = 'v1',
        replica: int=1
    ):

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split
        self.use_empty_for_mat = use_empty_for_mat
        self.use_empty_for_part = use_empty_for_part
        self.cond_signal = cond_signal
        self.cond_signal_version = cond_signal_version

        self.category_n2i = category_n2i
        self.category_i2n = category_i2n

        with open(os.path.join(meta_path, 'mat_categories.json'), "r") as f:
            mat_info = json.load(f)
        if use_empty_for_mat:
            self.mat_i2n = {i + 1: v for i, v in enumerate(mat_info)}
            self.mat_i2n[0] = 'empty'   # padding one for empty
        else:
            self.mat_i2n = {i: v for i, v in enumerate(mat_info)}

        with open(os.path.join(meta_path, 'used_parts_fine.json'), "r") as f:
            part_info = json.load(f)
        if use_empty_for_part:
            self.part_i2n = {i + 1: v for i, v in enumerate(part_info)}
            self.part_i2n[0] = 'empty'  # padding one for empty
            self.part_n2i = {v: i + 1 for i, v in enumerate(part_info)}
            self.part_n2i['empty'] = 0
        else:
            self.part_i2n = {i: v for i, v in enumerate(part_info)}
            self.part_n2i = {v: i for i, v in enumerate(part_info)}

        self.root = root
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.data_path = os.path.join(self.root, 'simulation_data', split)

        if categories is None:
            categories = os.listdir(self.data_path)
        categories.sort()

        print(f'Loading 3DCoMPaT200 dataset from {self.data_path} ...')
        print(f'Using Material Info Name: {MAT_FILE_NAME} ...')
        print(f'Using Condition Signal: {cond_signal} with Version {cond_signal_version} ...')
        print(f'Categories: {categories} | Num Samples: {num_samples} (Sample? {self.sampling}) | PC Size: {pc_size}\n'
              f'Scaling: {scaling} | Empty Mat: {use_empty_for_mat} | Empty Part: {use_empty_for_part} | Replica: {replica}')

        self.models = list()
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.data_path, c)
            assert os.path.isdir(subpath)
            for idx in os.listdir(subpath):
                model_path = os.path.join(subpath, idx, 'vecset_v3.1.npz')
                self.models.append(
                    {'category': c, 'model': model_path, 'idx': idx}
                )

        self.scaling = scaling
        self.replica = replica

    @staticmethod
    def __balance_sampling(labels, num_samples):
        occ_labels = np.where(labels == 1)[0]
        emp_labels = np.where(labels == 0)[0]
        half_samples = num_samples // 2

        occ_idx = np.random.choice(occ_labels, half_samples, replace=False)
        emp_idx = np.random.choice(emp_labels, half_samples, replace=False)
        idx = np.concatenate([occ_idx, emp_idx])
        return idx

    @staticmethod
    def __get_cond_signal(model_path, modality, version='v1') -> torch.Tensor:
        cond_signal_dir = os.path.dirname(model_path).replace('simulation_data', 'data')
        cond_signal_path = os.path.join(cond_signal_dir, 'cond_signal', f'{version}_{modality}.safetensors')
        with safe_open(cond_signal_path, framework='pt') as f:
            cond_signal = f.get_tensor(modality)
        # we need to randomly select one sample from the tensor
        chosen_idx = np.random.choice(cond_signal.shape[0], 1, replace=False)
        return cond_signal[chosen_idx.item()]       # NOTE: use item() to ensure a correct shape

    def __get_mat_params(self, model, parts):
        sim_dir = os.path.dirname(model)
        mat_json = MAT_FILE_NAME
        with open(os.path.join(sim_dir, mat_json), "r") as f:
            mat_params = json.load(f)

        E = np.zeros(len(parts), dtype=np.float32)
        nu = np.zeros(len(parts), dtype=np.float32)
        sigma = np.zeros(len(parts), dtype=np.float32)
        phi = np.zeros(len(parts), dtype=np.float32)
        rho = np.zeros(len(parts), dtype=np.float32)
        mmid = np.zeros(len(parts), dtype=np.int64)

        unique_part_labels = np.sort(np.unique(parts))

        for part_label in unique_part_labels:
            if (self.use_empty_for_part and part_label == 0) or \
               (not self.use_empty_for_part and part_label == -1):
                continue
            part_name = self.part_i2n[part_label]
            try:
                mat_info = mat_params[part_name]
            except KeyError:
                with open('error.txt', 'a') as f:
                    f.write(f'{model} | {part_name}\n')
                continue

            selected_ids = np.where(parts == part_label)[0]

            E_ori = mat_info['E']
            E[selected_ids] = E_ori

            nu_ori = mat_info['nu']
            nu[selected_ids] = nu_ori

            if "sigma_y" in mat_info:
                sigma_ori = mat_info['sigma_y']
                sigma[selected_ids] = sigma_ori
            else:
                sigma[selected_ids] = 0.0   # default value

            if "phi" in mat_info:
                phi_ori = mat_info['phi']
                phi[selected_ids] = phi_ori
            else:
                phi[selected_ids] = 25.0    # default value

            rho_ori = mat_info.get('rho', 1000.0)  # default density
            rho[selected_ids] = rho_ori

            mmid_ori = mat_info['mmid']
            mmid[selected_ids] = int(mmid_ori[1:])

        out = {
            'E': E.clip(0., 1e12),
            'nu': nu.clip(0., 0.5),
            'sigma': sigma.clip(0., 1e12),
            'phi': phi.clip(0., 90.),
            'rho': rho.clip(0., 1e5),
            'mmid': mmid
        }

        return out

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        category = self.models[idx]['category']
        model = self.models[idx]['model']
        idx = self.models[idx]['idx']

        data = np.load(model)

        vol_points = data['vols']                           # float32
        vol_label = data['sign_vols']                       # bool
        vol_mats = data['mat_vols']                         # uint8
        vol_parts = data['part_vols'].astype(np.int64)      # int64
        vol_rgbs = data['rgb_vols']                         # uint8

        near_points = data['nears']                         # float32
        near_label = data['sign_nears']                     # bool
        near_mats = data['mat_nears']                       # uint8
        near_parts = data['part_nears'].astype(np.int64)    # int64
        near_rgbs = data['rgb_nears']                       # uint8

        if self.return_surface:
            surface = data['surfs']                             # float32
            surfs_rgbs = data['rgb_surfs']                      # uint8
            surfs_parts = data['part_surfs'].astype(np.int64)   # int64
            if self.surface_sampling:
                ind = np.random.choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
                surfs_rgbs = surfs_rgbs[ind]
                surfs_parts = surfs_parts[ind]
            surface = torch.from_numpy(surface)
            surfs_rgbs = torch.from_numpy(surfs_rgbs)
            surfs_rgbs = surfs_rgbs / 255.0
            surfs_rgbs = (surfs_rgbs - 0.5) / 0.5
            surfs_parts = torch.from_numpy(surfs_parts)

        if self.sampling:
            ind = self.__balance_sampling(vol_label, self.num_samples)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]
            vol_mats = vol_mats[ind]
            vol_parts = vol_parts[ind]
            vol_rgbs = vol_rgbs[ind]

            ind = self.__balance_sampling(near_label, self.num_samples)
            near_points = near_points[ind]
            near_label = near_label[ind]
            near_mats = near_mats[ind]
            near_parts = near_parts[ind]
            near_rgbs = near_rgbs[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label)
        vol_mats = torch.from_numpy(vol_mats)
        vol_parts = torch.from_numpy(vol_parts)
        vol_rgbs = torch.from_numpy(vol_rgbs)

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label)
            near_mats = torch.from_numpy(near_mats)
            near_parts = torch.from_numpy(near_parts)
            near_rgbs = torch.from_numpy(near_rgbs)

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
            mats = torch.cat([vol_mats, near_mats], dim=0)
            parts = torch.cat([vol_parts, near_parts], dim=0)
            rgbs = torch.cat([vol_rgbs, near_rgbs], dim=0)
        else:
            points = vol_points
            labels = vol_label
            mats = vol_mats
            parts = vol_parts
            rgbs = vol_rgbs

        labels = labels.float()
        mats = mats.long()
        parts = parts.long()
        rgbs = rgbs / 255.0
        rgbs = (rgbs - 0.5) / 0.5

        points *= self.scaling
        surface *= self.scaling

        if self.transform:
            surface, points = self.transform(surface, points)

        # NOTE: when preprocessing the data, we add one to the index for emtpy
        if not self.use_empty_for_mat:
            mats = mats - 1
        # NOTE: when preprocessing the data, we add one to the index for emtpy
        if not self.use_empty_for_part:
            parts = parts - 1
            if self.return_surface:
                surfs_parts = surfs_parts - 1

        mat_params = self.__get_mat_params(model, parts)
        mat_E = torch.from_numpy(mat_params['E'])
        mat_nu = torch.from_numpy(mat_params['nu'])
        mat_sigma = torch.from_numpy(mat_params['sigma'])
        mat_phi = torch.from_numpy(mat_params['phi'])
        mat_rho = torch.from_numpy(mat_params['rho'])
        mat_mmid = torch.from_numpy(mat_params['mmid'])

        surf_mat_params = self.__get_mat_params(model, surfs_parts)
        surf_mat_E = torch.from_numpy(surf_mat_params['E'])
        surf_mat_nu = torch.from_numpy(surf_mat_params['nu'])
        surf_mat_sigma = torch.from_numpy(surf_mat_params['sigma'])
        surf_mat_phi = torch.from_numpy(surf_mat_params['phi'])
        surf_mat_rho = torch.from_numpy(surf_mat_params['rho'])
        surf_mat_mmid = torch.from_numpy(surf_mat_params['mmid'])

        out_list = [
            points, labels, mats, mat_E, mat_nu, mat_sigma, mat_phi, mat_rho, mat_mmid, parts, rgbs
        ]

        if self.cond_signal is not None and self.cond_signal in ['text', 'image']:
            cond_signals = self.__get_cond_signal(model, self.cond_signal, version=self.cond_signal_version)
            out_list.append(cond_signals.float())
        else:
            out_list.append(torch.tensor([0.])) # dummy tensor

        if self.return_surface:
            out_list.extend([
                surface, surfs_rgbs, surfs_parts, surf_mat_E, surf_mat_nu, surf_mat_sigma, surf_mat_phi, surf_mat_rho, surf_mat_mmid
            ])

        out_list.append(category_n2i[category])
        out_list.append(idx)

        return out_list

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica
