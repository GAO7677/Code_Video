import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1] / "wan_"
MODULES_ROOT = ROOT / "modules"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_test_modules():
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None)
    flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: (_ for _ in
                                                                 ()).throw(
                                                                     RuntimeError(
                                                                         "flash_attn stub should not be called in tests"
                                                                     ))
    sys.modules["flash_attn"] = flash_attn

    flash_attn_interface = types.ModuleType("flash_attn_interface")
    flash_attn_interface.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn_interface", loader=None)
    sys.modules["flash_attn_interface"] = flash_attn_interface

    wan_pkg = types.ModuleType("wan_")
    wan_pkg.__path__ = [str(ROOT)]
    sys.modules["wan_"] = wan_pkg

    modules_pkg = types.ModuleType("wan_.modules")
    modules_pkg.__path__ = [str(MODULES_ROOT)]
    sys.modules["wan_.modules"] = modules_pkg

    attention_mod = load_module("wan_.modules.attention",
                                MODULES_ROOT / "attention.py")
    model_mod = load_module("wan_.modules.model", MODULES_ROOT / "model.py")
    state_mod = load_module("wan_.state_condition", ROOT / "state_condition.py")

    def fake_flash_attention(q, k, v, q_lens=None, k_lens=None, **kwargs):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), dropout_p=0.0)
        return out.transpose(1, 2).to(v.dtype)

    model_mod.flash_attention = fake_flash_attention
    return attention_mod, model_mod, state_mod


class StateAdapterTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, cls.model_mod, cls.state_mod = ensure_test_modules()

    def test_canonicalize_predicted_states(self):
        payload = self.state_mod.canonicalize_state_condition({
            "predicted_states": torch.randn(1, 3, 2, 10)
        })
        self.assertIn("state_tokens", payload)
        self.assertEqual(tuple(payload["state_tokens"].shape), (1, 6, 10))

    def test_condition_maps_are_flattened_to_spatiotemporal_tokens(self):
        adapter = self.state_mod.WanObjectStateAdapter(
            model_dim=24,
            memory_token_dim=6,
            map_token_dim=3,
        )
        state_condition = {
            "memory_tokens": torch.randn(1, 2, 6),
            "condition_maps": torch.randn(1, 3, 3, 4, 4),
        }
        encoded = adapter(state_condition)
        # 2 memory tokens + (3 time steps * 4 * 4 spatial cells) map tokens
        self.assertEqual(tuple(encoded.shape), (1, 50, 24))

    def test_wan_model_accepts_state_context(self):
        model = self.model_mod.WanModel(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=8,
            in_dim=4,
            dim=24,
            ffn_dim=48,
            freq_dim=8,
            text_dim=16,
            out_dim=4,
            num_heads=4,
            num_layers=2,
        )
        adapter = self.state_mod.WanObjectStateAdapter(
            model_dim=model.dim,
            state_token_dim=10,
            memory_token_dim=6,
            map_token_dim=3,
        )
        state_condition = {
            "predicted_states": torch.randn(1, 3, 2, 10),
            "memory_tokens": torch.randn(1, 2, 6),
            "condition_maps": torch.randn(1, 3, 3, 4, 4),
        }
        state_context = list(adapter(state_condition).unbind(0))
        x = [torch.randn(4, 2, 4, 4)]
        context = [torch.randn(5, 16)]
        outputs = model(
            x,
            t=torch.tensor([1]),
            context=context,
            seq_len=8,
            state_context=state_context,
            state_scale=1.0,
        )
        self.assertEqual(len(outputs), 1)
        self.assertEqual(tuple(outputs[0].shape), (4, 2, 4, 4))
        self.assertGreater(len(model.get_state_adapter_state_dict()), 0)


if __name__ == "__main__":
    unittest.main()
