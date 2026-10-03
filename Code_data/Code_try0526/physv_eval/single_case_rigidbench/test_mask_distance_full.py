"""Analytical and independent-oracle tests for missing-mask-aware distances."""
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from .mask_distance_full import score_masks, summarize


class MaskDistanceTests(unittest.TestCase):
    def test_identical(self):
        g=np.zeros((3,2,5,7),bool);g[:,:,1:3,2:4]=True
        s=summarize(score_masks(g,g))
        for k in ('l2_full','chamfer_full','l2_legacy','chamfer_legacy','mask_missing_rate'):
            self.assertEqual(s[k],0)
        self.assertEqual(s['iou'],1)

    def test_known_corner_distance_and_two_chamfer_directions(self):
        g=np.zeros((1,1,3,5),bool);p=g.copy();g[0,0,0,0]=1;p[0,0,2,4]=1
        s=summarize(score_masks(g,p));d=np.sqrt(20)/3
        self.assertAlmostEqual(s['l2_full'],d)
        self.assertAlmostEqual(s['chamfer_full'],2*d)
        self.assertEqual(s['iou'],0)

    def test_old_score_can_hide_failure_new_score_cannot(self):
        g=np.zeros((2,1,3,5),bool);g[:,:,1,1]=1;p=g.copy();p[1]=0
        s=summarize(score_masks(g,p));d=np.sqrt(20)/3
        self.assertEqual(s['l2_legacy'],0)
        self.assertEqual(s['chamfer_legacy'],0)
        self.assertAlmostEqual(s['l2_full'],d/2)
        self.assertAlmostEqual(s['chamfer_full'],d)
        self.assertEqual(s['mask_missing_rate'],.5)

    def test_multiactor_missing_actor_not_discarded(self):
        g=np.zeros((2,2,3,5),bool);g[:,:,1,1]=1;p=g.copy();p[:,1]=0
        s=summarize(score_masks(g,p));d=np.sqrt(20)/3
        self.assertEqual(s['l2_legacy'],0)
        self.assertAlmostEqual(s['l2_full'],d/2)
        self.assertEqual(s['missing_actor_frames'],2)
        self.assertEqual(s['gt_present_actor_frames'],4)

    def test_asymmetric_empty_and_both_empty(self):
        g=np.zeros((3,1,3,5),bool);p=g.copy();g[0,0,1,1]=1;p[1,0,1,1]=1
        a=score_masks(g,p);s=summarize(a);d=np.sqrt(20)/3
        np.testing.assert_allclose(a['l2_full'][:,0],[d,d,0])
        np.testing.assert_allclose(a['chamfer_full'][:,0],[2*d,2*d,0])
        self.assertIsNone(s['l2_legacy']);self.assertIsNone(s['chamfer_legacy'])
        self.assertEqual(s['mask_missing_rate'],1)
        self.assertEqual(s['false_positive_actor_frames'],1)
        self.assertEqual(s['both_empty_actor_frames'],1)
        self.assertEqual(s['iou'],1/3)

    def test_no_gt_present_missing_rate_is_undefined(self):
        g=np.zeros((2,1,3,5),bool);s=summarize(score_masks(g,g))
        self.assertIsNone(s['mask_missing_rate'])
        self.assertEqual(s['l2_full'],0)

    def test_brute_force_oracle_and_deletion_monotonicity(self):
        rng=np.random.default_rng(8123)
        for _ in range(40):
            g=rng.random((3,2,7,9))<.12;p=rng.random(g.shape)<.12
            g[:,:,0,0]=True;p[:,:,-1,-1]=True
            a=score_masks(g,p)
            for t in range(3):
                for n in range(2):
                    x=np.argwhere(g[t,n]);y=np.argwhere(p[t,n]);dist=cdist(x,y)
                    self.assertAlmostEqual(a['l2_full'][t,n],np.linalg.norm(x.mean(0)-y.mean(0))/7)
                    self.assertAlmostEqual(a['chamfer_full'][t,n],(dist.min(0).mean()+dist.min(1).mean())/7)
            removed=p.copy();removed[rng.random((3,2))<.5]=False;b=score_masks(g,removed)
            for key in ('l2_full','chamfer_full'):
                self.assertTrue(np.all(b[key]>=a[key]-1e-12))
                self.assertAlmostEqual(summarize(a)[key],summarize(a)[key.replace('_full','_legacy')])

    def test_legacy_keeps_frame_then_actor_weighting(self):
        g=np.zeros((2,2,3,5),bool);g[:,:,0,0]=1;p=g.copy();p[1,1]=0;p[1,0]=0;p[1,0,0,3]=1
        s=summarize(score_masks(g,p))
        # Frame means are 0, 1. Pooling surviving actor-frames would wrongly give 1/3.
        self.assertEqual(s['l2_legacy'],.5)

    def test_future_slice_not_context(self):
        g=np.zeros((10,1,3,5),bool);g[:,:,1,1]=1;p=g.copy();p[8:]=0;a=score_masks(g,p)
        self.assertEqual(summarize(a)['mask_missing_rate'],.2)
        self.assertEqual(summarize(a,8,10)['mask_missing_rate'],1)
        self.assertIsNone(summarize(a,8,10)['l2_legacy'])

    def test_same_centroid_different_shape_is_known_limit(self):
        g=np.zeros((1,1,5,5),bool);p=g.copy();g[0,0,2,1]=g[0,0,2,3]=1;p[0,0,1,2]=p[0,0,3,2]=1
        s=summarize(score_masks(g,p));self.assertEqual(s['l2_full'],0);self.assertEqual(s['iou'],0)
        self.assertGreater(s['chamfer_full'],0)

    def test_invalid_data_is_error_not_empty(self):
        g=np.ones((2,1,3,5),bool)
        for p in [g[:1],np.ones((2,1,3,4),bool),np.full(g.shape,np.nan),np.full(g.shape,.3),np.full(g.shape,2),g[:0],np.zeros((2,0,3,5),bool)]:
            with self.subTest(shape=p.shape):
                with self.assertRaises((ValueError,TypeError)):score_masks(g,p)
        for observed in [np.array([[True],[False]]),np.ones((2,),bool),np.ones((2,1),int)]:
            with self.assertRaises((ValueError,TypeError)):score_masks(g,g,observed=observed)

    def test_invalid_slice_rejected(self):
        g=np.ones((2,1,3,5),bool);a=score_masks(g,g)
        for start,stop in [(-1,2),(0,3),(1,1),(2,1)]:
            with self.assertRaises(ValueError):summarize(a,start,stop)

    def test_actor_and_frame_permutation_preserves_full_mean(self):
        rng=np.random.default_rng(55);g=rng.random((4,3,5,7))<.2;p=rng.random(g.shape)<.2;p[1,1]=0
        a=summarize(score_masks(g,p));b=summarize(score_masks(g[::-1,::-1],p[::-1,::-1]))
        for k in ['l2_full','chamfer_full','mask_missing_rate','iou']:self.assertAlmostEqual(a[k],b[k])

    def test_one_pixel_image(self):
        g=np.ones((1,1,1,1),bool);p=np.zeros_like(g);s=summarize(score_masks(g,p))
        self.assertEqual(s['l2_full'],0);self.assertEqual(s['mask_missing_rate'],1)
        # Degenerate image has zero spatial diameter; missing rate remains necessary.

if __name__=='__main__':unittest.main()
