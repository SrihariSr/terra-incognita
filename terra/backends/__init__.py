"""Model backends behind one interface (see base.py).

hf.py           PyTorch / Hugging Face (Qwen2.5-7B-Instruct)
mlx_backend.py  MLX on Apple silicon (Gemma 4 12B and 31B, 4-bit)
fake.py         a tiny fake model for tests (TERRA_FAKE_LM=1)
"""

from .base import BaseLM, Patch  # noqa: F401
