from __future__ import annotations

from .ball_block import main as ball_block_main, score_case as score_ball_block_case
from .cosmos_reason1 import main as cosmos_reason1_main, score_case as score_cosmos_reason1_case
from .pdi import main as pdi_main, score_case as score_pdi_case
from .phyground import main as phyground_main, score_case as score_phyground_case
from .proxy import main as proxy_main, score_case as score_proxy_case
from .videophy2 import main as videophy2_main, score_case as score_videophy2_case
from .wmreward import main as wmreward_main, score_case as score_wmreward_case

__all__ = [
    "ball_block_main",
    "cosmos_reason1_main",
    "pdi_main",
    "phyground_main",
    "proxy_main",
    "score_ball_block_case",
    "score_cosmos_reason1_case",
    "score_pdi_case",
    "score_phyground_case",
    "score_proxy_case",
    "score_videophy2_case",
    "score_wmreward_case",
    "videophy2_main",
    "wmreward_main",
]
