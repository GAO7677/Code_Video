import unittest
import json
from pathlib import Path
import tempfile
import numpy as np
from context_geometry import camera_from_render,project,observed_prompts,size_from_actor,extract_observed


class CameraContractTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(resolution=[1280,720],
            camera=dict(location=[0,-4,1],target=[0,0,1],effective_yfov_deg=40))
        self.k,self.rt,self.hw = camera_from_render(self.meta)

    def test_camera_target_projects_to_pixel_center(self):
        pixels,depth = project(np.array([0,0,1]),self.k,self.rt)
        np.testing.assert_allclose(pixels,[639.5,359.5])
        self.assertAlmostEqual(float(depth),4)

    def test_rotation_is_proper_opencv(self):
        r = self.rt[:,:3]
        np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r),1)
        pixels,_ = project(np.array([[1,0,1],[0,0,2]]),self.k,self.rt)
        self.assertGreater(pixels[0,0],639.5)
        self.assertLess(pixels[1,1],359.5)

    def test_sphere_size_uses_dynamic_dimensions_only(self):
        np.testing.assert_allclose(size_from_actor(dict(shape="sphere",size_m=dict(radius=.11))),
                                   [.22,.22,.22])

    def test_prompt_contains_projected_center(self):
        positions = np.repeat(np.array([[[0.,0.,1.]]]),8,0)
        boxes,centers = observed_prompts(positions,np.array([[.22,.22,.22]]),self.k,self.rt,self.hw)
        self.assertEqual(boxes.shape,(8,1,4))
        self.assertTrue(np.all(centers>=boxes[...,:2]))
        self.assertTrue(np.all(centers<=boxes[...,2:]))

    def test_reject_behind_camera(self):
        with self.assertRaisesRegex(ValueError,"behind"):
            project(np.array([0,-5,1]),self.k,self.rt)

    def test_reject_camera_forward_mismatch(self):
        self.meta["camera"]["forward"] = [0,0,1]
        with self.assertRaisesRegex(ValueError,"disagree"):
            camera_from_render(self.meta)

    def test_future_and_static_fields_do_not_change_export(self):
        with tempfile.TemporaryDirectory(prefix="observed-context-test-") as folder:
            sample = Path(folder)
            (sample/"raw").mkdir()
            metadata = dict(actors={
                "moving":dict(dynamic=True,shape="sphere",size_m=dict(radius=.1),mass_kg=3),
                "wall":dict(dynamic=False,shape="box",size_m=dict(hx=99,hy=99,hz=99))},
                scenario_spec=dict(secret_static_geometry="not model input"))
            (sample/"metadata.json").write_text(json.dumps(metadata))
            positions = np.zeros((10,2,3))
            positions[:8,0,2] = 1
            positions[:8,0,0] = np.arange(8)*.01
            positions[8:] = 123
            states = dict(positions=positions,frame_times=np.arange(10)/30,
                          object_names=np.array(["moving","wall"]))
            np.savez_compressed(sample/"raw/states_xyzw.npz",**states)
            render = json.loads(json.dumps(self.meta))
            render["camera"]["object_projections_xy_depth"] = {"moving":[.5,.5,4]}
            first,_ = extract_observed(sample,render,np.arange(8)/30)
            positions[8:] = -9999
            positions[:,1] = 5555
            metadata["actors"]["wall"]["size_m"] = {"invalid_static_field":True}
            metadata["scenario_spec"] = {"future_collision":True}
            metadata["actors"]["moving"]["mass_kg"] = 777
            (sample/"metadata.json").write_text(json.dumps(metadata))
            np.savez_compressed(sample/"raw/states_xyzw.npz",**states)
            second,_ = extract_observed(sample,render,np.arange(8)/30)
            self.assertEqual(set(first),{"positions_world","size_m","camera_K","camera_world_to_view",
                                         "mask_boxes_xyxy","center_pixels","frame_times"})
            for key in first:
                np.testing.assert_array_equal(first[key],second[key])


if __name__ == "__main__":
    unittest.main()

