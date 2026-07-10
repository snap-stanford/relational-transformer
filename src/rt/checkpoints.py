"""Load RT checkpoints from a local path *or* the HuggingFace Hub — one interface.

A released checkpoint is a directory (a local folder or a Hub model repo
``org/repo[/subdir]``) holding:

* ``model.safetensors`` -- the weights, a flat ``state_dict`` stored with
  :func:`safetensors.torch.save_file` (the format ``scripts/pretrain.py``
  writes); and
* ``config.json``       -- the model dims + the text-embedding model used, so a
  loader needs no out-of-band knowledge.

For backward compatibility, legacy ``model.pt`` checkpoints (a
``{"model": state_dict}`` pickle, the older release/training format) still load:
if no ``.safetensors`` is found, or ``config["checkpoint_file"]`` ends in
``.pt``, the weights are read with ``torch.load(...)["model"]``.

This mirrors :mod:`rt.pre`'s local-or-Hub *data* resolver, so a checkpoint
reference like ``stanford-star/rt-j/classification`` "just works" (downloaded and cached
on demand), while a local training run's checkpoint dir works with the same call.
A local path always wins, so iterating locally never triggers a download.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rt.pre import resolve_repo

try:
    _RT_VERSION = version("relational-transformer")
except PackageNotFoundError:  # running from a source tree without an install
    _RT_VERSION = None

# huggingface_hub user-agent so Hub downloads are attributed to this library.
_HF_UA = {"library_name": "relational-transformer", "library_version": _RT_VERSION}

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
LEGACY_MODEL_FILE = "model.pt"
WEIGHT_SUFFIXES = (".safetensors", ".pt")
MODEL_DIM_KEYS = ("num_blocks", "d_model", "d_text", "num_heads", "d_ff")


def save_model(state_dict, path, metadata: dict | None = None) -> None:
    """Save a flat tensor ``state_dict`` to ``path`` as safetensors.

    ``metadata`` (e.g. ``{"step": 1000}``) is coerced to a str→str header, as
    safetensors metadata only holds strings.
    """
    from safetensors.torch import save_file

    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    save_file(state_dict, str(path), metadata=meta or None)


def load_model(path):
    """Load a flat tensor ``state_dict`` from a ``.safetensors`` (or legacy
    ``.pt``) checkpoint at ``path``."""
    path = Path(path)
    if path.suffix == ".pt":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    from safetensors.torch import load_file

    return load_file(str(path))


def resolve_checkpoint(
    spec, *, revision: str | None = None, subfolder: str | None = None
) -> tuple[dict, Path]:
    """Return ``(config, model_path)`` for a local or Hub checkpoint.

    ``spec`` may be: a local weights file (``model.safetensors`` or legacy
    ``model.pt``; config from a sibling ``config.json`` if present), a local
    directory, or a Hub ``org/repo[/subdir]``. ``subfolder`` selects a
    sub-directory within the repo/directory (the HuggingFace-idiomatic way to
    pick a checkpoint; equivalent to appending it to ``spec``). Within a
    directory, an explicit ``config["checkpoint_file"]`` wins, else
    ``model.safetensors`` is preferred over a legacy ``model.pt``.
    """
    p = Path(spec).expanduser()
    if p.is_file():
        cfg_path = p.with_name(CONFIG_FILE)
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        return config, p
    if p.is_dir():
        d = p / subfolder if subfolder else p
    elif str(spec).endswith(WEIGHT_SUFFIXES):
        # ``org/repo/<file>.pt`` -- a single weights file inside a Hub repo that
        # holds many checkpoints (e.g. ``stanford-star/rt-plurel``). The repo's
        # ``config.json`` (next to the file, else at the repo root) supplies the
        # model dims.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        repo_id, filename = resolve_repo(spec)
        if subfolder:
            filename = f"{subfolder}/{filename}"
        model_path = Path(
            hf_hub_download(repo_id, filename, revision=revision, **_HF_UA)
        )
        config = {}
        parent = str(Path(filename).parent)
        candidates = (
            [CONFIG_FILE] if parent == "." else [f"{parent}/{CONFIG_FILE}", CONFIG_FILE]
        )
        for cfg_name in candidates:
            try:
                cfg = hf_hub_download(repo_id, cfg_name, revision=revision, **_HF_UA)
            except EntryNotFoundError:
                continue
            config = json.loads(Path(cfg).read_text())
            break
        return config, model_path
    else:
        from huggingface_hub import snapshot_download

        repo_id, subdir = resolve_repo(spec)
        subdir = "/".join(part for part in (subdir, subfolder) if part)
        local = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[f"{subdir}/*"] if subdir else None,
            **_HF_UA,
        )
        d = Path(local) / subdir if subdir else Path(local)
    config = json.loads((d / CONFIG_FILE).read_text())
    fname = config.get("checkpoint_file")
    if fname is None:
        # Prefer safetensors, fall back to a legacy .pt if that is all there is.
        fname = MODEL_FILE if (d / MODEL_FILE).exists() else LEGACY_MODEL_FILE
    return config, d / fname


def load_rt_model(
    spec,
    *,
    device: str = "cpu",
    compile: bool = False,
    revision: str | None = None,
    subfolder: str | None = None,
    model_kwargs: dict | None = None,
):
    """Resolve a checkpoint, build the RelationalTransformer, load its weights.

    Returns ``(model, config)``. Thin backward-compatible wrapper around
    :meth:`rt.model.RelationalTransformer.from_pretrained` (the HuggingFace-style
    entry point, which returns just the model with ``config`` attached as
    ``model.config``). ``model_kwargs`` overrides/fills model dims when a
    checkpoint has no ``config.json`` (e.g. a raw internal ckpt during dev).
    """
    from rt.model import RelationalTransformer

    model = RelationalTransformer.from_pretrained(
        spec,
        device=device,
        compile=compile,
        revision=revision,
        subfolder=subfolder,
        **(model_kwargs or {}),
    )
    return model, model.config


def _adapt_state_dict(state_dict):
    """Rename pre-RT-J RMSNorm parameters (``<norm>.weight`` -> ``<norm>.scale``).

    Older checkpoints used ``nn.RMSNorm`` whose parameter is called ``weight``;
    :class:`rt.model.RMSNorm` calls it ``scale``. Only norm parameters are
    renamed; ``nn.Linear`` weights keep their names. No-op on new checkpoints.
    """

    def is_norm(key: str) -> bool:
        mod = key.rsplit(".", 2)
        return (
            "norms." in key
            or "norm_dict." in key
            or key.startswith("norm_out.")
            or (len(mod) >= 2 and mod[-2] in ("q_norm", "k_norm"))
        )

    return {
        (
            k[: -len(".weight")] + ".scale"
            if k.endswith(".weight") and is_norm(k)
            else k
        ): v
        for k, v in state_dict.items()
    }
