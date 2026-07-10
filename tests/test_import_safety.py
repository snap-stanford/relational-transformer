"""Import-time guarantees, checked in fresh subprocesses (sys.modules must be
pristine, so each assertion runs in its own interpreter)."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_rt_embed_imports_without_strictfire():
    # 3.13-safety: rt.embed must import without strictfire (which imports the
    # stdlib `pipes` module removed in 3.13). The heavy embedding deps
    # (orjson/ml_dtypes/sentence_transformers) are stubbed out.
    r = _run(
        """
        import sys, types
        for name, attrs in [
            ("orjson", {"loads": lambda *a, **k: None}),
            ("ml_dtypes", {"bfloat16": object()}),
            ("sentence_transformers", {"SentenceTransformer": object()}),
        ]:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
        import rt.embed
        assert "strictfire" not in sys.modules, "rt.embed imported strictfire"
        assert hasattr(rt.embed, "TextEmbedder")
        print("ok")
        """
    )
    assert r.returncode == 0, r.stderr
