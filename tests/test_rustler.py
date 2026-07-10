"""The compiled Rust engine: symbols + preprocess end-to-end."""

from __future__ import annotations


def test_extension_symbols():
    import rt._rustler as r

    assert hasattr(r, "Sampler")
    assert hasattr(r, "column_sem_types")
    assert hasattr(r, "preprocess")  # present only when built with --features pre


def test_preprocess_end_to_end(synthetic_dataset, tmp_path):
    from rt._rustler import preprocess

    out = tmp_path / "out"
    preprocess(str(synthetic_dataset), str(out), skip_tasks=True)

    # preprocess writes to <out>/<dataset name>/
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    files = {p.name for p in produced.rglob("*") if p.is_file()}
    # the on-disk format the Sampler consumes
    assert {"nodes.rkyv", "table_info.json", "column_index.json"} <= files
