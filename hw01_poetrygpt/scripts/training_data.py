"""Compatibility imports for historical data-preparation scripts.

The input contract lives in poetrygpt.serialization. Keep this path because
release builders also run directly as ``python scripts/<builder>.py``.
"""
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from poetrygpt.serialization import (
    IGNORE, SPECIAL_TOKENS, encode_content, encode_prompt, encode_sft,
    pad_batch, pretrain_stream, pretrain_windows, special_ids,
)

__all__ = [
    "IGNORE", "SPECIAL_TOKENS", "encode_content", "encode_prompt", "encode_sft",
    "pad_batch", "pretrain_stream", "pretrain_windows", "special_ids",
]
