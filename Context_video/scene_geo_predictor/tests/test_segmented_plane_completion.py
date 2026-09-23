import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
from segmented_plane_completion import complete_segmented_depth
from test_local_plane_completion import setup


def test_planes_and_observation_invariance():
    for slope in [0., .3, -.5]:
        z, mask, k, truth = setup(slope)
        out, filled, audit = complete_segmented_depth(z, mask, k)
        assert filled.sum() == 400, audit
        assert np.allclose(out, truth)
        assert np.array_equal(out[~mask], z[~mask])


def test_edges_and_gaps():
    for mode in ['step', 'open', 'gap']:
        z, mask, k, _ = setup()
        if mode == 'step':
            z[:, 60:] = 4
        elif mode == 'open':
            z[:, 60:] = 0
        else:
            z[10:25, 10:25] = 0
        z[mask] = 0
        out, filled, audit = complete_segmented_depth(z, mask, k)
        if mode != 'gap':
            assert not filled.any(), audit
        else:
            assert filled.sum() == 400
            assert np.all(out[10:25, 10:25] == 0)


def test_sparse_depth_outliers():
    z, mask, k, truth = setup(.2)
    rng = np.random.default_rng(123)
    outliers = (rng.random(z.shape) < .07) & ~mask
    z[outliers] += .3
    out, filled, audit = complete_segmented_depth(z, mask, k)
    assert filled.sum() == 400, audit
    assert np.allclose(out[mask], truth[mask])
    assert np.array_equal(out[~mask], z[~mask])


def test_later_frame_reveals_reference_occlusion():
    z, mask, k, truth = setup()
    # A larger reference silhouette contains static points recovered in RGB1-7.
    target = mask.copy()
    target[30:60, 45:75] = True
    out, filled, audit = complete_segmented_depth(z, target, k)
    assert filled.sum() == 400, audit
    assert np.allclose(out, truth)
    assert np.array_equal(out[~mask], z[~mask])


def test_unknown_border_is_not_a_conflicting_surface():
    z, mask, k, truth = setup()
    z[34, 50:70] = 0  # narrow unknown strip, outside authorized completion area
    out, filled, audit = complete_segmented_depth(z, mask, k)
    assert filled.sum() == 400, audit
    assert np.allclose(out[mask], truth[mask])
    assert np.array_equal(out[~mask], z[~mask])


if __name__ == '__main__':
    test_planes_and_observation_invariance()
    test_edges_and_gaps()
    test_sparse_depth_outliers()
    test_later_frame_reveals_reference_occlusion()
    test_unknown_border_is_not_a_conflicting_surface()
    print('PASS planes, sparse outliers, visible gap, step, open edge, observed invariance')
