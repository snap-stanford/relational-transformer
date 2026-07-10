"""Relational Transformer (RT) -- a foundation model for relational data.

    from rt import RelationalTransformer

    # Load from the HuggingFace Hub or a local checkpoint directory:
    model = RelationalTransformer.from_pretrained("stanford-star/rt-j/classification")
    model = RelationalTransformer.from_pretrained("/path/to/checkpoint")

Lower-level helpers (load_rt_model, resolve_checkpoint, save_model) live in
:mod:`rt.checkpoints`.
"""

from importlib.metadata import PackageNotFoundError, version

from rt.model import RelationalTransformer

try:
    __version__ = version("relational-transformer")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["RelationalTransformer", "__version__"]
