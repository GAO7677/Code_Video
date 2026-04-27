import trimesh
import numpy as np
from dataset.transform import AxisScaling
from dataset.compat200 import D3CoMPaT200


__all__ = [
    'build_3dcompat200_occupancy_dataset',
]


def build_3dcompat200_occupancy_dataset(split, replica, args):
    if split == 'train':
        transform = AxisScaling((0.75, 1.25), True)
        return D3CoMPaT200(
            args.data_path, split=split, transform=transform, sampling=True, num_samples=1024,
            return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size, scaling=2.0,
            use_empty_for_mat=args.use_empty_for_mat, use_empty_for_part=args.use_empty_for_part,
            cond_signal=getattr(args, 'conditional_signal', None), cond_signal_version=getattr(args, 'conditional_signal_ver', 'v2'), replica=replica
        )
    else:
        return D3CoMPaT200(
            args.data_path, split=split, transform=None, sampling=False, return_surface=True,
            surface_sampling=True, pc_size=args.point_cloud_size, scaling=2.0,
            use_empty_for_mat=args.use_empty_for_mat, use_empty_for_part=args.use_empty_for_part,
            cond_signal=getattr(args, 'conditional_signal', None), cond_signal_version=getattr(args, 'conditional_signal_ver', 'v2'), replica=replica
        )


def plot_pointcloud(points, filename, features=None, features_map=None, save_ply=False):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)

    if features is not None:
        unique_features = np.unique(features)
        for i, f in enumerate(unique_features):
            label = features_map[f] if features_map is not None else f
            points_i = points[features == f]
            colors_i = plt.colormaps.get_cmap('tab20')(i)[:3]
            colors_i = np.tile(colors_i, (points_i.shape[0], 1))
            ax.scatter(points_i[:, 0], points_i[:, 2], points_i[:, 1], c=colors_i, s=4, label=label)

    else:
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=4)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_aspect('auto')
    plt.legend(loc='best', markerscale=2)
    plt.savefig(f'dataset/debug/{filename}.png', bbox_inches='tight', pad_inches=0.2)

    if save_ply:
        if features is not None:
            colors = np.zeros((points.shape[0], 3))
            for i, f in enumerate(unique_features):
                colors[features == f] = plt.colormaps.get_cmap('tab20')(i)[:3]
            pcd = trimesh.PointCloud(points, colors)
        else:
            pcd = trimesh.PointCloud(points)
        pcd.export(f'dataset/debug/{filename}.ply')


def test_3dcompat200(split='train'):
    transform = AxisScaling((0.75, 1.25), True)
    dataset = D3CoMPaT200(
        root='path/to/data',
        split=split,
        transform=transform if split=='train' else None,
        sampling=True,
        num_samples=1024,
        return_surface=True,
        surface_sampling=True,
        pc_size=2048,
        scaling=2.0,
        use_empty_for_mat=False,
        use_empty_for_part=False,
        cond_signal='text',
        cond_signal_version='v2',
        replica=1
    )

    # np.random.seed(12)
    ind = np.random.choice(len(dataset), 4, False)
    # ind = [0, 1, 2, 3]
    for i in ind:
        p, l, m, e, nu, sigma, phi, rho, mmid, pa, r, cond, s, sr, spa, se, snu, ssigma, sphi, srho, smmid, c, idx = dataset[i]
        # print(f'Point: {p.shape}, Label: {l.shape}, Mat: {m.shape}, PBMat: {pbm.shape} Part: {pa.shape}, Surf: {s.shape}, SurfRGB: {sr.shape}, SurfPart: {spa.shape}, Category: {c}, Index: {idx}')
        # print(f'Point: {p.dtype}, Label: {l.dtype}, Mat: {m.dtype}, PBMat: {pbm.dtype} Part: {pa.dtype}, Surf: {s.dtype}, SurfRGB: {sr.dtype}, SurfPart: {spa.dtype}')
        print(f'point: {p.shape} {p.dtype}\t\tlabel: {l.shape} {l.dtype}\t\tmat: {m.shape} {m.dtype}')
        print(f'E: {e.shape} {e.dtype} {e.min()} {e.max()}\tnu: {nu.shape} {nu.dtype} {nu.min():.3f} {nu.max():.3f}\trho: {rho.shape} {rho.dtype} {rho.min()} {rho.max()}')
        print(f'sigma: {sigma.shape} {sigma.dtype} {sigma.min()} {sigma.max()}\tphi: {phi.shape} {phi.dtype} {phi.min()} {phi.max()}')
        print(f'mmid: {mmid.shape} {mmid.dtype} {mmid.min()} {mmid.max()}')
        # print(f'pbmat: {pbm.shape} {pbm.dtype}\t\tpart: {pa.shape} {pa.dtype}\t\trgb: {r.shape} {r.dtype}')
        print(f'part: {pa.shape} {pa.dtype}\t\trgb: {r.shape} {r.dtype}')
        print(f'Surf E: {se.shape} {se.dtype} {se.min()} {se.max()}\tSurf nu: {snu.shape} {snu.dtype} {snu.min():.3f} {snu.max():.3f}\tSurf rho: {srho.shape} {srho.dtype} {srho.min()} {srho.max()}')
        print(f'Surf sigma: {ssigma.shape} {ssigma.dtype} {ssigma.min()} {ssigma.max()}\tSurf phi: {sphi.shape} {sphi.dtype} {sphi.min()} {sphi.max()}')
        print(f'Surf mmid: {smmid.shape} {smmid.dtype} {smmid.min()} {smmid.max()}')
        print(f'cond signal {cond.shape}, {cond.dtype}')
        print(f'surf: {s.shape} {s.dtype}\t\tsrgb: {sr.shape} {sr.dtype}')
        print(f'spart: {spa.shape} {spa.dtype}\t\tcategory: {c}  index: {idx}')

        print(f'occ: {l.sum()}/{l.size(0)}')
        print('=========q=========')
        print(f'{p[:, 0].min()} - {p[:, 0].max()}')
        print(f'{p[:, 1].min()} - {p[:, 1].max()}')
        print(f'{p[:, 2].min()} - {p[:, 2].max()}')
        print('=========s=========')
        print(f'{s[:, 0].min()} - {s[:, 0].max()}')
        print(f'{s[:, 1].min()} - {s[:, 1].max()}')
        print(f'{s[:, 2].min()} - {s[:, 2].max()}')
        print(f'rgb range: {r.min()} - {r.max()}')
        print(f'surf rgb range: {sr.min()} - {sr.max()}')
        print(f'unique mat: {np.unique(m)} (use_empty_for_mat==False)')
        print(f'unique surf part: {np.unique(spa)}')
        print()

        # occ_p = p[l==1]
        # occ_rgb = r[l==1]
        # occ_rgb = occ_rgb * 0.5 + 0.5
        # pcd = trimesh.PointCloud(occ_p, colors=occ_rgb)
        # pcd.export(f'dataset/debug/comp200mp_{split}_{dataset.category_i2n[c]}_{idx}.ply')

if __name__ == '__main__':
    split = 'train'
    # test_3dcompat200(split)
    pass
