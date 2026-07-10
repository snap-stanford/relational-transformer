"""Public API: RelationalTransformer.from_pretrained + load_rt_model."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from rt import RelationalTransformer  # noqa: E402
from rt.checkpoints import (  # noqa: E402
    CONFIG_FILE,
    LEGACY_MODEL_FILE,
    MODEL_FILE,
    load_rt_model,
    save_model,
)


def test_from_pretrained_local(tiny_checkpoint):
    ckpt, src = tiny_checkpoint
    model = RelationalTransformer.from_pretrained(ckpt, device="cpu")
    assert isinstance(model, RelationalTransformer)
    assert model.config["embedding_model"] == "test-embed"  # config attached
    s1, s2 = src.state_dict(), model.state_dict()
    assert s1.keys() == s2.keys()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)  # weights round-trip


def test_load_rt_model_backcompat(tiny_checkpoint):
    ckpt, _ = tiny_checkpoint
    model, config = load_rt_model(str(ckpt))
    assert isinstance(model, RelationalTransformer)
    assert config["embedding_model"] == "test-embed"


def test_from_pretrained_subfolder(tmp_path, tiny_dims):
    src = RelationalTransformer(**tiny_dims, compile=False, materialize_attn_masks=True)
    (tmp_path / "classification").mkdir()
    save_model(src.state_dict(), tmp_path / "classification" / MODEL_FILE)
    (tmp_path / "classification" / CONFIG_FILE).write_text(
        json.dumps({"model": tiny_dims, "embedding_model": "sub"})
    )
    model = RelationalTransformer.from_pretrained(tmp_path, subfolder="classification")
    assert model.config["embedding_model"] == "sub"


def test_from_pretrained_model_kwargs(tmp_path, tiny_dims):
    # config.json without dims -> dims supplied via keyword args
    src = RelationalTransformer(**tiny_dims, compile=False, materialize_attn_masks=True)
    save_model(src.state_dict(), tmp_path / MODEL_FILE)
    (tmp_path / CONFIG_FILE).write_text(json.dumps({"embedding_model": "x"}))
    model = RelationalTransformer.from_pretrained(tmp_path, **tiny_dims)
    assert model.config["embedding_model"] == "x"


def test_compile_true_builds(tiny_dims):
    # Regression: __init__ must still torch.compile forward when compile=True
    # (an earlier refactor accidentally orphaned that line).
    m = RelationalTransformer(**tiny_dims, compile=True, materialize_attn_masks=True)
    assert callable(m.forward)


def test_from_pretrained_legacy_pt(tmp_path, tiny_dims):
    # Legacy .pt ({"model": state_dict}) with flat top-level dims (the
    # stanford-star/rt-plurel format); no safetensors -> falls back to model.pt.
    src = RelationalTransformer(**tiny_dims, compile=False, materialize_attn_masks=True)
    torch.save({"model": src.state_dict()}, tmp_path / LEGACY_MODEL_FILE)
    (tmp_path / CONFIG_FILE).write_text(json.dumps({**tiny_dims, "embedding_model": "pt"}))
    model = RelationalTransformer.from_pretrained(tmp_path)
    assert model.config["embedding_model"] == "pt"
    s1, s2 = src.state_dict(), model.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
