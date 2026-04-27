import copy
import torch
import trimesh
import warp as wp
import subprocess
import numpy as np
import kaolin as kal
from pathlib import Path
import nvdiffrast.torch as dr
import torch.nn.functional as F
from typing import Optional, Literal, List, Tuple
# -------------------------------------- gaoya -------------------------------------- #
import sys
sys.path.append("/home/gaoya/Code_Video/SOPHY-main/")



# -------------------------------------- gaoya -------------------------------------- #
from dynamics.util.sph import volume_sampling
from dynamics.util.io import safe_symlink
from dynamics.util.extern.libmesh import check_mesh_contains

from dynamics.util.constants import MANIFOLD_PLUS


def uniform_sampling(mesh: trimesh.Trimesh, resolution: int) -> np.ndarray:
    lower_bound = mesh.vertices.min(0)
    upper_bound = mesh.vertices.max(0)
    dims = np.linspace(lower_bound, upper_bound, resolution).T
    grid = np.stack(np.meshgrid(*dims, indexing='ij'), axis=-1).reshape(-1, 3)
    p_x = grid[check_mesh_contains(mesh, grid)]

    return p_x


def volumetric_sampling(mesh: trimesh.Trimesh, resolution: int, asset_path: Path) -> np.ndarray:
    import pyvista

    bounds = mesh.bounds.copy()
    mesh.vertices = (mesh.vertices - bounds.mean(0)) / (bounds[1] - bounds[0]).max() + 0.5
    cache_obj_path = asset_path / f'temp.obj'
    cache_vtk_path = asset_path / f'temp.vtk'
    mesh.export(cache_obj_path)

    radius = 1.0 / resolution * 0.5
    volume_sampling(cache_obj_path, cache_vtk_path, radius, res=(resolution, resolution, resolution))
    pcd: pyvista.PolyData = pyvista.get_reader(str(cache_vtk_path)).read()
    p_x = np.array(pcd.points).copy()

    # undo normalization
    p_x = (p_x - 0.5) * (bounds[1] - bounds[0]).max() + bounds.mean(0)

    cache_obj_path.unlink(missing_ok=True)
    cache_vtk_path.unlink(missing_ok=True)

    return p_x


def surface_sampling(mesh: trimesh.Trimesh, resolution: int) -> np.ndarray:
    # resolution in this case is the number of points to sample
    points = trimesh.sample.sample_surface_even(mesh, resolution // 2)[0]

    noise = np.random.normal(0, 0.001, points.shape)
    points_n1 = points.copy() + noise

    return np.concatenate([points, points_n1], axis=0)


def get_warp_device(device: torch.device) -> wp.context.Device:
    if device.type == 'cuda':
        return wp.get_device(f'cuda:{device.index}')
    else:
        return wp.get_device('cpu')


def prepare_simulation_data_given_mesh(
    save_dir: Path,
    particles_path: Optional[Path] = None,
    mesh_path: Optional[Path] = None,
    mesh_sample_mode: Literal["uniform", "volumetric", "surface"] = "volumetric",
    mesh_sample_resolution: int = 30,
    particles_downsample_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float, kal.rep.SurfaceMesh]:
    print(f'  Start preparing data for simulation ...')
    save_dir.mkdir(parents=True, exist_ok=True)

    default_settings = {
        'with_materials': True, 
        'with_normals': False, 
        "triangulate": True,
        'heterogeneous_mesh_handler': kal.io.utils.mesh_handler_naive_triangulate
    }

    assert mesh_path.suffix == '.obj', "Only '.obj' mesh files are supported currently."
    # print(f'  Extracting particles from mesh file [{mesh_path.name}] ...')
    safe_symlink(mesh_path, save_dir / mesh_path.name)
    safe_symlink(mesh_path.with_suffix('.mtl'), save_dir / f"{mesh_path.stem}.mtl")
    safe_symlink(mesh_path.with_suffix('.png'), save_dir / f"{mesh_path.stem}.png")
    mesh: trimesh.Trimesh = trimesh.load_mesh(mesh_path, force='mesh', process=False)
    if not mesh.is_watertight and particles_path is None:
        print(f'  Warning [{mesh_path}]: not watertight')
        new_mesh_path = mesh_path.with_suffix('.mp.obj')
        if new_mesh_path.exists():
            print(f'  Warning: reading the fixed mesh from [{new_mesh_path}]')
        else:
            print(f'  Warning: starting manifold plus to fix the mesh')
            subprocess.run([
                MANIFOLD_PLUS,
                '--input', mesh_path.as_posix(),
                '--output', new_mesh_path.as_posix(),
                '--depth', '8'
            ])
        mesh: trimesh.Trimesh = trimesh.load_mesh(new_mesh_path, force='mesh', process=False)
        assert mesh.is_watertight, f"  Failed to fix the mesh: {new_mesh_path}"
        if mesh_sample_mode == "uniform":
            particles = uniform_sampling(mesh, mesh_sample_resolution)
        elif mesh_sample_mode == "volumetric":
            particles = volumetric_sampling(mesh, mesh_sample_resolution, save_dir)
        elif mesh_sample_mode == "surface":
            particles = surface_sampling(mesh, mesh_sample_resolution)
        else:
            raise ValueError(f"  Unsupported mesh sample mode: {mesh_sample_mode}")

    vol_all = mesh.volume

    if particles_path is not None:
        print(f'  Extracting particles from pcd file [{particles_path}] ...')
        pcd = trimesh.load(particles_path, process=False)
        particles = pcd.vertices

    particles = particles.astype(np.float32)

    # downsample particles
    rand_idx = np.random.permutation(particles.shape[0])
    particles = particles[rand_idx][::particles_downsample_factor]

    kal_mesh = kal.io.obj.import_mesh(
        mesh_path, **default_settings, raw_materials=True
    )

    # mesh vertices
    mesh_vertices = kal_mesh.vertices.numpy().astype(np.float32)

    print(f'  Vertices: {mesh_vertices.shape[0]}, Particles: {particles.shape[0]}')

    return mesh_vertices, particles, vol_all, kal_mesh


def mesh_rasterization(
    meshes: List[kal.rep.SurfaceMesh],
    camera: kal.render.camera.Camera,
    background: torch.Tensor,
    nvctx,
    ssaa_scale: int = 2,
    return_alpha: bool = False,
    background_mesh: Optional[kal.rep.SurfaceMesh | List[kal.rep.SurfaceMesh]] = None,
):  
    meshes_new = list()
    if len(meshes) < 0:
        raise ValueError("No meshes to rasterize")
    elif len(meshes) == 1:
        meshes_new.append(copy.deepcopy(meshes[0]))
        # meshes_new.append(meshes[0])
    else:
        meshes_new.extend(meshes)

    if background_mesh is not None:
        if isinstance(background_mesh, kal.rep.SurfaceMesh):
            background_mesh = [background_mesh]
        meshes_new.extend(background_mesh)

    mesh = kal.rep.SurfaceMesh.flatten(meshes_new)
    mesh.to_batched()

    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    
    # Create a fake W (See nvdiffrast documentation)
    proj = camera.projection_matrix()[None]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(
        vertices_camera
    )[..., None]

    vertices_clip = (proj @ homogeneous_vecs).squeeze(-1)

    H, W = camera.height * ssaa_scale, camera.width * ssaa_scale

    rast = dr.rasterize(
        nvctx, vertices_clip, mesh.faces.int(),
        (H, W), grad_db=False
    )
    rast0 = torch.flip(rast[0], dims=(1,))
    hard_mask = rast0[:, :, :, -1:] != 0
    face_idx = (rast0[..., -1].long() - 1).contiguous()

    uv_map = dr.interpolate(
        mesh.uvs, rast0, mesh.face_uvs_idx[0, ...].int()
    )[0] % 1.

    # Preprocess OBJ materials into a format we will use for rendering 
    materials = [
        m['map_Kd'].unsqueeze(0).cuda().float() / 255. if 'map_Kd' in m else
        m['Kd'].reshape(1, 1, 1, 3).cuda() for m in mesh.materials[0]
    ]

    img = torch.zeros((1, H, W, 3), device='cuda', dtype=torch.float32)
    # Obj meshes can be composed of multiple materials
    # so at rendering we need to interpolate from corresponding materials
    im_material_idx = mesh.material_assignments[0, ...][face_idx]  # Assume single assignments map
    im_material_idx[face_idx == -1] = -1
    for i, material in enumerate(materials):
        mask = im_material_idx == i
        mask_idx = torch.nonzero(mask, as_tuple=False)
        _texcoords = uv_map[mask]
        if _texcoords.shape[0] > 0:
            pixel_val = dr.texture(
                materials[i].contiguous(),
                _texcoords.reshape(1, 1, -1, 2).contiguous(),
                filter_mode='linear'
            )
            img[mask] = pixel_val[0, 0]

    # # Combine the image and hard mask to get an RGBA image
    # img = torch.cat([img, hard_mask.float()], dim=-1)

    img_rgba = torch.cat([img, hard_mask.float()], dim=-1)

    # Combine the image with background color
    img = img * hard_mask.float() + background * (1 - hard_mask.float())
    img = img.permute(0, 3, 1, 2).contiguous()
    img = F.interpolate(img, size=(camera.height, camera.width), mode='bilinear', align_corners=False)
    img = img.permute(0, 2, 3, 1).contiguous()

    return torch.clamp(img, 0., 1.)[0], torch.clamp(img_rgba, 0., 1.)[0] if return_alpha else None


def mesh_rasterization_v2(
    meshes: List[kal.rep.SurfaceMesh],
    camera: kal.render.camera.Camera,
    background: torch.Tensor,
    nvctx,
    ssaa_scale: int = 2,
    return_alpha: bool = False,
    background_mesh: Optional[kal.rep.SurfaceMesh | List[kal.rep.SurfaceMesh]] = None,
):
    meshes_new = list()
    if len(meshes) < 0:
        raise ValueError("No meshes to rasterize")
    elif len(meshes) == 1:
        meshes_new.append(copy.deepcopy(meshes[0]))
        # meshes_new.append(meshes[0])
    else:
        meshes_new.extend(meshes)

    if background_mesh is not None:
        if isinstance(background_mesh, kal.rep.SurfaceMesh):
            background_mesh = [background_mesh]
        meshes_new.extend(background_mesh)

    mesh = kal.rep.SurfaceMesh.flatten(meshes_new)
    mesh.to_batched()

    lighting = kal.render.lighting.SgLightingParameters(
        amplitude=6.,
        direction=(0., 2., 0.)
    ).cuda()

    camera.height = camera.height * ssaa_scale
    camera.width = camera.width * ssaa_scale

    render_res = kal.render.easy_render.render_mesh(camera, mesh, lighting=lighting, backend='cuda')
    # render pass
    img = render_res[kal.render.easy_render.RenderPass.render].clamp(0, 1)

    img = img.permute(0, 3, 1, 2).contiguous()
    img = F.interpolate(img, size=(camera.height, camera.width), mode='bilinear', align_corners=False)
    img = img.permute(0, 2, 3, 1).contiguous()

    return None, torch.clamp(img, 0., 1.)[0]
