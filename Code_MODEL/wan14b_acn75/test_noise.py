"""Small CPU contract tests; no model weights or GPU required."""
import unittest

import torch

from noise import context_noise_pair, flow_batch, future_loss, training_scheduler


class NoiseContractTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = training_scheduler("cpu")
        self.clean = torch.randn(2, 16, 5, 4, 6)
        self.noise = torch.randn_like(self.clean)

    def test_original_training_schedule(self):
        torch.testing.assert_close(self.scheduler.timesteps, torch.arange(1000, 0, -1).float())
        self.assertEqual(len(self.scheduler.sigmas), 1001)
        self.assertEqual(float(self.scheduler.sigmas[-1]), 0)

    def test_a_and_c_endpoint_equivalence(self):
        for index in (0, 125, 126, 999):
            a = flow_batch(self.clean, self.scheduler, "A", index, noise=self.noise)
            c = flow_batch(self.clean, self.scheduler, "C", index, noise=self.noise)
            original = self.scheduler.scale_noise(self.clean, a.future_timestep, self.noise)
            torch.testing.assert_close(a.latents, original, rtol=0, atol=0)
            torch.testing.assert_close(c.latents[:, :, :2], self.clean[:, :, :2], rtol=0, atol=0)
            torch.testing.assert_close(a.latents[:, :, 2:], c.latents[:, :, 2:], rtol=0, atol=0)
            self.assertTrue(torch.all(c.timesteps[:, :2] == 0))
            self.assertEqual(a.expert, "high" if index <= 125 else "low")

    def test_n75_uses_paired_sigma_and_same_noise(self):
        for index in (0, 57, 500, 999):
            item = flow_batch(self.clean, self.scheduler, "N75", index, noise=self.noise)
            paired = self.scheduler.sigmas[:1000]
            nearest = int(torch.argmin((paired - item.future_sigma * .75).abs()))
            self.assertEqual(item.context_timestep.item(), self.scheduler.timesteps[nearest].item())
            self.assertEqual(item.context_sigma.item(), paired[nearest].item())
            expected = (1 - item.context_sigma) * self.clean[:, :, :2] + item.context_sigma * self.noise[:, :, :2]
            torch.testing.assert_close(item.latents[:, :, :2], expected, rtol=0, atol=0)
            self.assertGreater(item.context_sigma.item(), 0)

    def test_context_does_not_contribute_to_loss(self):
        target = torch.randn_like(self.clean)
        prediction = target.clone().requires_grad_()
        with torch.no_grad():
            prediction[:, :, :2] += 100
            prediction[:, :, 2:] += 1
        loss = future_loss(prediction, target)
        torch.testing.assert_close(loss, torch.tensor(1.0))
        loss.backward()
        self.assertEqual(torch.count_nonzero(prediction.grad[:, :, :2]).item(), 0)
        self.assertGreater(torch.count_nonzero(prediction.grad[:, :, 2:]).item(), 0)

    def test_bfloat16_matches_original_operation_order(self):
        clean, noise = self.clean.bfloat16(), self.noise.bfloat16()
        item = flow_batch(clean, self.scheduler, "N75", 611, noise=noise)
        original_future = self.scheduler.scale_noise(clean, item.future_timestep, noise)
        expected_context = (1 - item.context_sigma) * clean[:, :, :2] + item.context_sigma * noise[:, :, :2]
        expected = torch.cat((expected_context, original_future[:, :, 2:]), dim=2)
        torch.testing.assert_close(item.latents, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
