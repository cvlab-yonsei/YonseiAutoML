"""
YSAutoML: Zero-Shot NAS (AutoFormer AZ-NAS)
--------------------------------------------

This package provides functions for:
  1. Searching architectures (run_search_zeroshot)
  2. Retraining searched subnets (run_retrain_zeroshot)

Usage:
    from ysautoml.network.zeroshot.autoformer import (
        run_search_zeroshot,
        run_retrain_zeroshot,
    )
"""

from .api import (
    run_search_zeroshot,
    run_retrain_zeroshot,
)

__all__ = [
    "run_search_zeroshot",
    "run_retrain_zeroshot",
]
