import unittest
import numpy as np
from probe_vggt import validate_predictions, validate_gpu, FORBIDDEN_UUID

GPU0 = "GPU-34579b7b-23fc-35ea-539f-1eac72fb7fa5"
INVENTORY = f"0, {GPU0}\n4, {FORBIDDEN_UUID}\n"


class RawPredictionContractTests(unittest.TestCase):
    def prediction(self):
        return dict(world_points=np.zeros((8,4,5,3),np.float32),
                    world_points_conf=np.ones((8,4,5),np.float32),
                    depth=np.ones((8,4,5,1),np.float32),
                    depth_conf=np.ones((8,4,5),np.float32),
                    extrinsic=np.zeros((8,3,4),np.float32),
                    intrinsic=np.tile(np.eye(3,dtype=np.float32),(8,1,1)),
                    images=np.zeros((8,3,4,5),np.float32))

    def test_accept_exact_eight_frames(self):
        validate_predictions(self.prediction())

    def test_reject_extra_frames(self):
        pred = self.prediction()
        pred["world_points"] = np.zeros((9,4,5,3),np.float32)
        with self.assertRaisesRegex(ValueError,"eight point maps"):
            validate_predictions(pred)

    def test_reject_nonfinite_output(self):
        pred = self.prediction()
        pred["world_points_conf"][0,0,0] = np.nan
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            validate_predictions(pred)

    def test_reject_nonpositive_depth(self):
        pred = self.prediction()
        pred["depth"][0,0,0,0] = 0
        with self.assertRaisesRegex(ValueError,"Nonpositive"):
            validate_predictions(pred)

    def test_confidence_is_not_assumed_probability(self):
        pred = self.prediction()
        pred["world_points_conf"] *= 10
        validate_predictions(pred)

    def test_accept_gpu0(self):
        self.assertEqual(validate_gpu(GPU0,GPU0,INVENTORY),0)

    def test_reject_gpu4_full_uuid(self):
        with self.assertRaises(ValueError):
            validate_gpu(FORBIDDEN_UUID,FORBIDDEN_UUID,INVENTORY)

    def test_reject_uuid_prefix(self):
        with self.assertRaisesRegex(ValueError,"canonical full"):
            validate_gpu("GPU-4a8abb69","GPU-4a8abb69",INVENTORY)

    def test_reject_index(self):
        with self.assertRaises(ValueError):
            validate_gpu("0","0",INVENTORY)

    def test_reject_physical_index4_after_reindexing(self):
        with self.assertRaisesRegex(ValueError,"other than 4"):
            validate_gpu(GPU0,GPU0,f"4, {GPU0}\n")

    def test_reject_multiple_visible_gpus(self):
        with self.assertRaises(ValueError):
            validate_gpu(GPU0,GPU0+","+FORBIDDEN_UUID,INVENTORY)

    def test_reject_unknown_uuid(self):
        with self.assertRaises(ValueError):
            validate_gpu(GPU0,GPU0,"")


if __name__ == "__main__":
    unittest.main()

