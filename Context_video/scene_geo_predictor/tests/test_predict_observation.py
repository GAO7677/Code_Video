import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from future_query_predictor import Predictor
from predict_observation import observation_batch, predict


class ObservationInferenceTest(unittest.TestCase):
    def test_label_free_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            positions = np.zeros((8, 1, 3), dtype=np.float32)
            positions[:, 0, 0] = np.arange(8)/30
            np.savez(root/'geometry.npz', positions_world=positions,
                size_m=np.ones((1, 3), dtype=np.float32)*.2, frame_times=np.arange(8)/30)
            np.savez(root/'scene.npz', scene_xyz=np.ones((4, 3)),
                scene_features=np.zeros((4, 1386)), scene_mask=np.ones(4, dtype=bool))
            model = Predictor(torch.zeros(65), torch.ones(65))
            save_file(model.state_dict(), str(root/'model.safetensors'))
            prediction, times = predict(root/'geometry.npz', root/'scene.npz', root/'model.safetensors')
            self.assertEqual(prediction.shape, (1, 41, 3))
            np.testing.assert_allclose(prediction[0, :, 0], times, atol=1e-6)
            self.assertEqual(len(list(root.iterdir())), 3)
            np.savez(root/'geometry.npz', positions_world=positions,
                size_m=np.ones((1, 3)), frame_times=np.arange(8)/24)
            with self.assertRaises(ValueError):
                observation_batch(root/'geometry.npz', root/'scene.npz')


if __name__ == '__main__':
    unittest.main()
