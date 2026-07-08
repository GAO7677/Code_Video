from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    "cosmos_reason1_main": (".cosmos_reason1", "main"),
    "pmf_main": (".pmf", "main"),
    "pdi_main": (".pdi", "main"),
    "phyground_main": (".phyground", "main"),
    "physics_iq_main": (".physics_iq", "main"),
    "proxy_main": (".proxy", "main"),
    "score_cosmos_reason1_case": (".cosmos_reason1", "score_case"),
    "score_pmf_case": (".pmf", "score_case"),
    "score_pdi_case": (".pdi", "score_case"),
    "score_phyground_case": (".phyground", "score_case"),
    "score_physics_iq_case": (".physics_iq", "score_case"),
    "score_proxy_case": (".proxy", "score_case"),
    "score_vbench_case": (".vbench", "score_case"),
    "score_vbench2_case": (".vbench2", "score_case"),
    "score_videophy2_case": (".videophy2", "score_case"),
    "score_wmreward_case": (".wmreward", "score_case"),
    "vbench2_main": (".vbench2", "main"),
    "vbench_main": (".vbench", "main"),
    "videophy2_main": (".videophy2", "main"),
    "wmreward_main": (".wmreward", "main"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)

__all__ = [
    "cosmos_reason1_main",
    "pmf_main",
    "pdi_main",
    "phyground_main",
    "physics_iq_main",
    "proxy_main",
    "score_cosmos_reason1_case",
    "score_pmf_case",
    "score_pdi_case",
    "score_phyground_case",
    "score_physics_iq_case",
    "score_proxy_case",
    "score_vbench_case",
    "score_vbench2_case",
    "score_videophy2_case",
    "score_wmreward_case",
    "vbench2_main",
    "vbench_main",
    "videophy2_main",
    "wmreward_main",
]
