#!/usr/bin/env python
"""Preprocess relbench-3.0.0 datasets into rustler's shareable on-disk format.

A dataset is addressed exactly like in the ``relbench`` loader: a local path, or
a HuggingFace Hub spec ``org/repo[/subdir]`` (e.g. ``stanford-star/the-join/join-act-mooc``
or ``stanford-star/relbench/rel-f1``). Hub datasets are downloaded (and cached) on
demand; local paths are used in place.

Pipeline per dataset:  download/resolve  ->  rustler `pre`  ->  text embeddings.
The result is a self-contained ``<out_dir>/<name>/`` directory (see ``meta.json``)
that can be used directly for training or uploaded to a Hub ``*-preprocessed`` repo
and consumed from there.

Subcommands::

    preprocess.py one   --dataset <spec> --out-dir <dir> [--upload-repo <repo>] ...
    preprocess.py many  --repo <hf repo> --out-dir <dir> [--shard i --num-shards N] ...
    preprocess.py list  --repo <hf repo>
    preprocess.py upload --pre-dir <dir>/<name> --repo <repo>      # one dataset
    preprocess.py upload --pre-dir <dir> --repo <repo> --bulk      # whole collection

Recommended sharing workflow for a large collection (e.g. the 650-dataset Join):
preprocess everything locally with ``many`` (skipping uploads), then push the whole
``out-dir`` in one resumable ``upload --bulk`` pass. ``--bulk`` uses
``upload_large_folder`` (batched commits, far fewer Hub API calls than per-dataset
``upload_folder``), which avoids the account rate limits that per-dataset uploads hit.

Build the preprocessor binary first: ``pixi run build-pre`` (or it is built
automatically by the ``preprocess`` pixi task).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 1024


# --------------------------------------------------------------------------- #
# Hub / local addressing  (mirrors relbench.hf so we need no relbench dep)
# --------------------------------------------------------------------------- #
def resolve_repo(spec: str) -> tuple[str, str]:
    """Split a Hub spec into ``(repo_id, subdir)``.

    ``"org/name"`` -> ``("org/name", "")``; ``"org/name/a/b"`` -> ``("org/name", "a/b")``.
    """
    parts = spec.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"{spec!r} is not a Hub 'org/name[/subdir]' spec or a local path."
        )
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def resolve_dataset_dir(spec: str, revision: str | None = None) -> Path:
    """Return a local directory holding the dataset (manifest.yaml + db/ + tasks/).

    A local path with a ``manifest.yaml`` is used as-is; otherwise ``spec`` is a Hub
    ``org/repo[/subdir]`` and only that sub-path is downloaded (and cached).
    """
    p = Path(spec).expanduser()
    if (p / "manifest.yaml").exists():
        return p
    repo_id, subdir = resolve_repo(spec)
    if not subdir:
        return Path(
            snapshot_download(repo_id=repo_id, revision=revision, repo_type="dataset")
        )
    # Scope the tree listing to just this subdir. snapshot_download's allow_patterns
    # path still recursively lists the *whole* repo, which is huge (and gets rate
    # limited) for big collection repos like the-join (hundreds of datasets); listing
    # only ``subdir`` keeps the API calls small and avoids HTTP 429s.
    api = HfApi()
    files = [
        e.path
        for e in api.list_repo_tree(
            repo_id, path_in_repo=subdir, recursive=True,
            repo_type="dataset", revision=revision,
        )
        if e.__class__.__name__ == "RepoFile"
    ]
    local_root = None
    for rel in files:
        local = hf_hub_download(
            repo_id, rel, revision=revision, repo_type="dataset",
        )
        if local_root is None:
            # hf_hub_download returns <cache>/<...>/snapshots/<rev>/<rel>; strip rel.
            local_root = Path(local)
            for _ in Path(rel).parts:
                local_root = local_root.parent
    return Path(local_root) / subdir


def dataset_name(dataset_dir: Path) -> str:
    """Read the dataset name from its manifest (the output subdirectory name)."""
    import yaml  # PyYAML ships with huggingface_hub's deps; fall back to a tiny parse

    text = (dataset_dir / "manifest.yaml").read_text()
    try:
        return yaml.safe_load(text)["name"]
    except Exception:
        for line in text.splitlines():
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip().strip("'\"")
    raise ValueError(f"no 'name' in {dataset_dir / 'manifest.yaml'}")


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def run_rustler_pre(
    dataset_dir: Path, out_dir: Path, source: str, skip_tasks: bool
) -> None:
    from rt._rustler import preprocess

    print(f"+ preprocess {dataset_dir} -> {out_dir}", flush=True)
    preprocess(str(dataset_dir), str(out_dir), source=source, skip_tasks=skip_tasks)


def embed_dataset(
    pre_dataset_dir: Path, embedding_model: str, batch_size: int
) -> int:
    """Compute text embeddings for a preprocessed dataset; return d_text."""
    # Lazy import so download/upload/list work without torch installed.
    from rt.embed import main as embed_main

    out_root = pre_dataset_dir.parent
    name = pre_dataset_dir.name
    embed_main(
        dataset_name=name,
        pre_dir=str(out_root),
        device=None,  # auto: all visible GPUs, else CPU
        batch_size=batch_size,
        embedding_model=embedding_model,
    )
    emb_path = pre_dataset_dir / f"text_emb_{embedding_model}.bin"
    num_text = len(json.loads((pre_dataset_dir / "text.json").read_text()))
    # bfloat16 -> 2 bytes/elem; the emb file is (num_text, d_text) row-major.
    d_text = emb_path.stat().st_size // (max(num_text, 1) * 2)
    return d_text


def _embeddings_done(pre_dataset_dir: Path) -> bool:
    """True once ``meta.json`` records its text-embedding files and they exist.
    Used by ``--skip-existing`` so a dataset whose embedding step was interrupted
    (meta.json present, but no ``.bin``) is reprocessed rather than skipped."""
    meta_path = pre_dataset_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        embs = json.loads(meta_path.read_text()).get("text_embeddings", {})
    except Exception:
        return False
    return bool(embs) and all(
        (pre_dataset_dir / e["file"]).exists() for e in embs.values()
    )


def update_meta_with_embeddings(
    pre_dataset_dir: Path, embedding_model: str, d_text: int
) -> None:
    meta_path = pre_dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("text_embeddings", {})[embedding_model] = {
        "file": f"text_emb_{embedding_model}.bin",
        "d_text": d_text,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def upload_dataset(pre_dataset_dir: Path, repo: str, private: bool) -> None:
    """Upload ``<out_dir>/<name>/`` to ``repo`` under ``<name>/`` on the Hub."""
    name = pre_dataset_dir.name
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"uploading {pre_dataset_dir} -> {repo}/{name}", flush=True)
    api.upload_folder(
        folder_path=str(pre_dataset_dir),
        path_in_repo=name,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"add preprocessed {name}",
    )
    print(f"uploaded {repo}/{name}", flush=True)


def bulk_upload(out_dir: Path, repo: str, private: bool) -> None:
    """Upload an entire preprocessed ``out_dir`` (all ``<name>/`` subdirs) in one
    resumable pass with ``upload_large_folder``.

    This is the recommended path for sharing a whole collection (e.g. the 650-dataset
    Join): it batches files and commits in chunks, so it uses far fewer Hub API calls
    than uploading each dataset with ``upload_folder`` -- which trips account-level
    rate limits on big collections. It is resumable: re-running picks up where an
    interrupted upload left off. Workflow: preprocess locally, then bulk-upload.
    """
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"bulk-uploading {out_dir} -> {repo} (upload_large_folder)", flush=True)
    api.upload_large_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(out_dir),
    )
    print(f"bulk-uploaded {out_dir} -> {repo}", flush=True)


def preprocess_one(
    spec: str,
    out_dir: Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    skip_tasks: bool = False,
    no_embed: bool = False,
    upload_repo: str | None = None,
    private: bool = True,
    revision: str | None = None,
) -> Path:
    dataset_dir = resolve_dataset_dir(spec, revision=revision)
    name = dataset_name(dataset_dir)
    pre_dataset_dir = out_dir / name
    print(f"=== preprocessing {name} ({spec}) -> {pre_dataset_dir} ===", flush=True)

    run_rustler_pre(dataset_dir, out_dir, source=spec, skip_tasks=skip_tasks)
    if not no_embed:
        d_text = embed_dataset(pre_dataset_dir, embedding_model, batch_size)
        update_meta_with_embeddings(pre_dataset_dir, embedding_model, d_text)
    if upload_repo:
        upload_dataset(pre_dataset_dir, upload_repo, private=private)
    return pre_dataset_dir


# --------------------------------------------------------------------------- #
# Listing a collection repo (e.g. the-join's join-*/ datasets)
# --------------------------------------------------------------------------- #
def list_datasets(repo: str, revision: str | None = None) -> list[str]:
    """Top-level dataset subdirectories of a Hub collection repo (those with a
    manifest.yaml), as ``org/repo/<subdir>`` specs."""
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    subdirs = sorted({
        f.split("/", 1)[0]
        for f in files
        if f.endswith("/manifest.yaml") and f.count("/") == 1
    })
    return [f"{repo}/{d}" for d in subdirs]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", required=True, help="preprocessed-data output root")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--skip-tasks", action="store_true", help="ingest db tables only")
    p.add_argument("--no-embed", action="store_true", help="skip text embeddings")
    p.add_argument("--upload-repo", default=None, help="Hub repo to upload result to")
    p.add_argument("--public", action="store_true", help="make uploaded repo public")
    p.add_argument("--revision", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_one = sub.add_parser("one", help="preprocess a single dataset")
    p_one.add_argument("--dataset", required=True, help="local path or org/repo[/subdir]")
    _add_common(p_one)

    p_many = sub.add_parser("many", help="preprocess all datasets in a collection repo")
    p_many.add_argument("--repo", required=True, help="Hub collection repo, e.g. stanford-star/the-join")
    p_many.add_argument("--shard", type=int, default=0, help="this shard index")
    p_many.add_argument("--num-shards", type=int, default=1, help="total shards (for slurm arrays)")
    p_many.add_argument("--skip-existing", action="store_true",
                        help="skip datasets whose output meta.json already exists")
    _add_common(p_many)

    p_list = sub.add_parser("list", help="list dataset specs in a collection repo")
    p_list.add_argument("--repo", required=True)
    p_list.add_argument("--revision", default=None)

    p_up = sub.add_parser("upload", help="upload an already-preprocessed dataset dir")
    p_up.add_argument("--pre-dir", required=True,
                      help="path to <out_dir>/<name>, or the whole <out_dir> with --bulk")
    p_up.add_argument("--repo", required=True)
    p_up.add_argument("--public", action="store_true")
    p_up.add_argument("--bulk", action="store_true",
                      help="upload the whole out-dir (all datasets) in one resumable "
                           "upload_large_folder pass -- recommended for big collections")

    args = parser.parse_args()

    if args.cmd == "list":
        for spec in list_datasets(args.repo, revision=args.revision):
            print(spec)
        return

    if args.cmd == "upload":
        pre_dir = Path(args.pre_dir).expanduser()
        if args.bulk:
            bulk_upload(pre_dir, args.repo, private=not args.public)
        else:
            upload_dataset(pre_dir, args.repo, private=not args.public)
        return

    if args.cmd == "one":
        preprocess_one(
            args.dataset, Path(args.out_dir).expanduser(),
            embedding_model=args.embedding_model, batch_size=args.batch_size,
            skip_tasks=args.skip_tasks, no_embed=args.no_embed,
            upload_repo=args.upload_repo, private=not args.public, revision=args.revision,
        )
        return

    if args.cmd == "many":
        specs = list_datasets(args.repo, revision=args.revision)
        shard = specs[args.shard :: args.num_shards]
        print(f"shard {args.shard}/{args.num_shards}: {len(shard)} of {len(specs)} datasets",
              flush=True)
        out_dir = Path(args.out_dir).expanduser()
        failures = []
        for i, spec in enumerate(shard):
            name = spec.rsplit("/", 1)[-1]
            if args.skip_existing and _embeddings_done(out_dir / name):
                print(f"[{i + 1}/{len(shard)}] skip existing {name}", flush=True)
                continue
            print(f"[{i + 1}/{len(shard)}] {spec}", flush=True)
            try:
                preprocess_one(
                    spec, out_dir,
                    embedding_model=args.embedding_model, batch_size=args.batch_size,
                    skip_tasks=args.skip_tasks, no_embed=args.no_embed,
                    upload_repo=args.upload_repo, private=not args.public,
                    revision=args.revision,
                )
            except Exception as e:  # one bad dataset shouldn't sink the shard
                print(f"  FAILED {spec}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                failures.append(spec)
        if failures:
            print(f"\n{len(failures)} failure(s):", file=sys.stderr)
            for s in failures:
                print(f"  {s}", file=sys.stderr)
            sys.exit(1)
        return


if __name__ == "__main__":
    main()
