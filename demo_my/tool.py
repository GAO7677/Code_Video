
def map_pc_to_particles( obj_idx):
    sim_particles = torch.tensor(objs[obj_idx].init_particles).to(device)
    print(f"number of sim_particles: {sim_particles.shape[0]}")
    K = 256
    num_closest = 5 if 'closest_points_num' not in config else config['closest_points_num']
    point_chunks = torch.split(fg_pcs[obj_idx]['points'], K)
    closest_indices = []

    for chunk in tqdm(point_chunks):
        # Calculate pairwise distances between chunk and all particles
        # Using broadcasting to avoid memory issues
        # Shape: [K, 1, 3] - [1, N, 3] -> [K, N, 3] -> [K, N]
        distances = torch.norm(
            chunk.unsqueeze(1) - sim_particles.unsqueeze(0),
            dim=2
        )
        # Get top num_closest indices of closest particles for this chunk
        chunk_closest = torch.topk(distances, k=num_closest, dim=1, largest=False)[1]
        del distances
        closest_indices.append(chunk_closest)

    closest_indices = torch.cat(closest_indices)
    return closest_indices


def get_material_for_each(self, per_material_type):
    if per_material_type == "rigid":
        obj_material = gs.materials.Rigid(
            rho = 1000.0 if 'rigid_rho' not in config else config['rigid_rho'],
            friction = 5.0 if 'rigid_friction' not in config else config['rigid_friction'],
            coup_friction = 5 if 'rigid_coup_friction' not in config else config['rigid_coup_friction'],
            coup_softness = 0.002 if 'rigid_coup_softness' not in config else config['rigid_coup_softness'],
        )
        obj_vis_mode = "visual"
    elif per_material_type == 'pbd_liquid':
        obj_material = gs.materials.PBD.Liquid(
            rho = 1000.0 if 'pbd_rho' not in config else config['pbd_rho'],
            density_relaxation = 0.2 if 'pbd_density_relaxation' not in config else config['pbd_density_relaxation'],
            viscosity_relaxation = 0.1 if 'pbd_viscosity_relaxation' not in config else config['pbd_viscosity_relaxation'],
        )
        obj_vis_mode = "particle"

    elif per_material_type == "pbd_cloth":
        obj_material = gs.materials.PBD.Cloth(
            rho=4.0 if 'pbd_rho' not in config else config['pbd_rho'],
            static_friction=0.6 if 'pbd_static_friction' not in config else config['pbd_static_friction'],
            kinetic_friction=0.35 if 'pbd_kinetic_friction' not in config else config['pbd_kinetic_friction'],
            stretch_compliance=1e-7 if 'pbd_stretch_compliance' not in config else config['pbd_stretch_compliance'],
            bending_compliance=1e-5 if 'pbd_bending_compliance' not in config else config['pbd_bending_compliance'],
            stretch_relaxation=0.7 if 'pbd_stretch_relaxation' not in config else config['pbd_stretch_relaxation'],
            bending_relaxation=0.1 if 'pbd_bending_relaxation' not in config else config['pbd_bending_relaxation'],
            air_resistance=5e-3 if 'pbd_air_resistance' not in config else config['pbd_air_resistance'],

        )
        obj_vis_mode = "particle"
    elif per_material_type == "pbd_elastic":
        obj_material = gs.materials.PBD.Elastic(
            rho=300.0 if 'pbd_elastic_rho' not in config else config['pbd_elastic_rho'],
            static_friction=0.15 if 'pbd_elastic_static_friction' not in config else config['pbd_elastic_static_friction'],
            kinetic_friction=0.0 if 'pbd_elastic_kinetic_friction' not in config else config['pbd_elastic_kinetic_friction'],
            stretch_compliance=0.0 if 'pbd_elastic_stretch_compliance' not in config else config['pbd_elastic_stretch_compliance'],
            bending_compliance=0.0 if 'pbd_elastic_bending_compliance' not in config else config['pbd_elastic_bending_compliance'],
            volume_compliance=0.0 if 'pbd_elastic_volume_compliance' not in config else config['pbd_elastic_volume_compliance'],
            stretch_relaxation=0.1 if 'pbd_elastic_stretch_relaxation' not in config else config['pbd_elastic_stretch_relaxation'],
            bending_relaxation=0.1 if 'pbd_elastic_bending_relaxation' not in config else config['pbd_elastic_bending_relaxation'],
            volume_relaxation=0.1 if 'pbd_elastic_volume_relaxation' not in config else config['pbd_elastic_volume_relaxation'],
        )
        obj_vis_mode = "particle"
    elif per_material_type == "pbd_particle":
        obj_material = gs.materials.PBD.Particle()
        obj_vis_mode = "particle"
    elif per_material_type == "mpm_sand":
        obj_material = gs.materials.MPM.Sand(
            E = 1e6 if 'MPM_E' not in config else config['MPM_E'],
            nu = 0.2 if 'MPM_nu' not in config else config['MPM_nu'],
            rho = 1000.0 if 'MPM_rho' not in config else config['MPM_rho'],
            friction_angle = 45 if 'MPM_friction_angle' not in config else config['MPM_friction_angle'],
        )
        obj_vis_mode = "particle"
    elif per_material_type == "mpm_elastic":
        obj_material = gs.materials.MPM.Elastic(
            E = 1e6 if 'MPM_E' not in config else config['MPM_E'],
            nu = 0.2 if 'MPM_nu' not in config else config['MPM_nu'],
            rho = 1000.0 if 'MPM_rho' not in config else config['MPM_rho'],
        )
        obj_vis_mode = "particle"
    elif per_material_type == "mpm_liquid":
        obj_material = gs.materials.MPM.Liquid(
            E = 1e6 if 'MPM_E' not in config else config['MPM_E'],
            nu = 0.2 if 'MPM_nu' not in config else config['MPM_nu'],
            rho = 1000.0 if 'MPM_rho' not in config else config['MPM_rho'],
        )
        obj_vis_mode = "particle"
    elif per_material_type == "mpm_snow":
        obj_material = gs.materials.MPM.Snow(
            E = 1e6 if 'MPM_E' not in config else config['MPM_E'],
            nu = 0.2 if 'MPM_nu' not in config else config['MPM_nu'],
            rho = 1000.0 if 'MPM_rho' not in config else config['MPM_rho'],
        )
        obj_vis_mode = "particle"
    elif per_material_type == "mpm_elastic2plastic":
        obj_material = gs.materials.MPM.ElastoPlastic(
            E = 1e6 if 'MPM_E' not in config else config['MPM_E'],
            nu = 0.2 if 'MPM_nu' not in config else config['MPM_nu'],
            rho = 1000.0 if 'MPM_rho' not in config else config['MPM_rho'],
        )
        obj_vis_mode = "particle"
    else:
        raise NotImplementedError(f"The current material {per_material_type} is not supported for now")
    return obj_material, obj_vis_mode


