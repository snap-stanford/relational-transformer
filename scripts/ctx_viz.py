"""Interactive web UI for inspecting rustler batch tensors.

Run:
    pixi run python scripts/ctx_viz.py
    # then open http://localhost:8765 in your browser

The server keeps a small LRU cache of `RustlerDataset` objects keyed by
the structural parameters that require a rebuild (db, table, target,
split, ctx sizes, bfs widths, walks, embedding model, shuffle seed, ...).
Per-request parameters that don't require a rebuild (context_seed,
mask_prob_max, item_idx, ctx_size within the configured cap) are applied
on the fly so iterating in the UI is fast.
"""

from __future__ import annotations

import argparse
import errno
import json
import socket
import sys
import threading
import traceback
from collections import OrderedDict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from rt.data import MAX_F2P_NBRS, RustlerDataset, get_column_index
from rt.pre import resolve_pre_dir, resolve_repo

SEM_TYPE_NAMES = ["number", "text", "datetime", "boolean"]
INT_MIN = np.iinfo(np.int32).min  # rustler uses i32::MIN as missing-timestamp sentinel

DEFAULT_PRE_ROOT = Path("pre")


# ---------------------------------------------------------------------------
# Per-db assets (text vocab, table info, column index) cached in-process.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _load_text_json(db_dir: str) -> list[str]:
    with open(Path(db_dir) / "text.json") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _load_table_info(db_dir: str) -> dict:
    with open(Path(db_dir) / "table_info.json") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _load_column_index(db_dir: str) -> dict:
    with open(Path(db_dir) / "column_index.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sampler cache.
# ---------------------------------------------------------------------------
# Building a Sampler is expensive (it mmaps the rkyv blobs, computes table
# ranges, subsamples items). We rebuild only when something structural
# changes; mask_prob_max and context_seed are mutated on the cached object.

_DATASET_CACHE: "OrderedDict[tuple, RustlerDataset]" = OrderedDict()
_DATASET_CACHE_LOCK = threading.Lock()
_DATASET_CACHE_MAX = 4


def _get_or_build_dataset(
    *,
    pre_root: str,
    db_name: str,
    table_name: str,
    target_column: str,
    split: str,
    columns_to_drop: tuple[str, ...],
    ctx_cap: int,
    local_ctx_size: int,
    bfs_width: int,
    num_walks: int,
    walk_length: int,
    prefer_latest: bool,
    embedding_model: str,
    d_text: int,
    items_per_task: int,
    bool_as_num: bool,
    skip_text_cols: bool,
    balance_labels: bool,
    ablate_schema_semantics: bool,
    shuffle_seed: int,
) -> RustlerDataset:
    key = (
        pre_root,
        db_name,
        table_name,
        target_column,
        split,
        columns_to_drop,
        ctx_cap,
        local_ctx_size,
        bfs_width,
        num_walks,
        walk_length,
        prefer_latest,
        embedding_model,
        d_text,
        items_per_task,
        bool_as_num,
        skip_text_cols,
        balance_labels,
        ablate_schema_semantics,
        shuffle_seed,
    )
    with _DATASET_CACHE_LOCK:
        if key in _DATASET_CACHE:
            _DATASET_CACHE.move_to_end(key)
            return _DATASET_CACHE[key]

    # local_ctx_sizes must contain values <= ctx_cap; we pass exactly one.
    local_ctx_sizes = [min(local_ctx_size, ctx_cap)]
    bfs_widths = [bfs_width]

    ds = RustlerDataset(
        tasks=[(db_name, table_name, target_column, split, list(columns_to_drop))],
        pre_dir=pre_root,
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_sizes=local_ctx_sizes,
        bfs_widths=bfs_widths,
        num_walks=num_walks,
        walk_length=walk_length,
        prefer_latest=[prefer_latest],
        mask_prob_max=0.0,  # mutated per-request via set_mask_prob_max_py
        embedding_model=embedding_model,
        d_text=d_text,
        shuffle_seed=shuffle_seed,
        context_seed=0,  # context_seed is folded into step at request time
        items_per_task=items_per_task,
        quiet=True,
        bool_as_num=bool_as_num,
        ignore_data_errors=False,
        skip_text_cols=skip_text_cols,
        mmap_populate=False,
        balance_labels=[balance_labels],
        timeout_per_item=3600.0,
        ablate_schema_semantics=ablate_schema_semantics,
        vector_db_path=None,
        train_only_fallback=False,
    )

    with _DATASET_CACHE_LOCK:
        _DATASET_CACHE[key] = ds
        _DATASET_CACHE.move_to_end(key)
        while len(_DATASET_CACHE) > _DATASET_CACHE_MAX:
            _DATASET_CACHE.popitem(last=False)
    return ds


# ---------------------------------------------------------------------------
# Context construction.
# ---------------------------------------------------------------------------


def _decode_value(sem_type: str, t: dict, text_vocab: list[str]) -> object:
    """Pull the human-readable value out of a token, in normalized form.

    The rustler-side number/datetime/boolean values are z-score normalized
    per column at preprocess time, so we display the raw normalized value;
    text cells we resolve back to the original string via text.json.
    """
    if sem_type == "text":
        idx = t["class_value_idx"]
        if 0 <= idx < len(text_vocab):
            return text_vocab[idx]
        return None
    if sem_type == "number":
        v = float(t["number_value"])
        return None if np.isnan(v) else v
    if sem_type == "datetime":
        v = float(t["datetime_value"])
        return None if np.isnan(v) else v
    if sem_type == "boolean":
        v = float(t["boolean_value"])
        return None if np.isnan(v) else v
    return None


def _format_value(sem_type: str, val: object) -> str:
    if val is None:
        return "—"
    if sem_type == "number":
        return f"{val:+.4g}"
    if sem_type == "boolean":
        return f"{val:+.4g}"
    if sem_type == "datetime":
        return f"{val:+.4g}"
    if sem_type == "text":
        s = str(val)
        return s if len(s) <= 80 else s[:77] + "…"
    return str(val)


def _build_context_payload(
    *,
    ds: RustlerDataset,
    db_name: str,
    db_dir: str,
    text_vocab: list[str],
    table_info: dict,
    column_index: dict,
    item_idx: int | None,
    bs: int,
    ctx_size: int,
    context_seed: int,
    mask_prob_max: float,
    target_column: str,
    target_column_idx: int,
    table_name: str,
    split: str,
) -> dict:
    """Generate one item's worth of context and return a JSON-serializable payload."""
    # Mutate seed/mask in place. The sampler's context_seed is folded
    # together with `step` (in eval-mode it's just used; in train-mode
    # it's combined with step). We treat eval-mode (item_idx given) as
    # canonical here, and use train-mode only when item_idx is None.
    ds.sampler.set_mask_prob_max_py(mask_prob_max)

    # For deterministic stepping, set the step explicitly when in train mode.
    if item_idx is None:
        ds.sampler.set_step_py(int(context_seed))
        ds.sampler.set_stride_py(0)
    # Note: eval-mode (item_idx given) doesn't use step/stride, but it
    # uses the sampler's `context_seed`. We can't trivially mutate that
    # without rebuilding, so eval-mode reproduces using a fixed seed
    # baked into the dataset. The "context_seed" knob primarily affects
    # train-mode here (item_idx == None).

    tup = ds.sampler.batch_py(item_idx, bs, ctx_size)
    batch = ds._process_batch(tup)

    # Pull row 0 of the batch.
    seq_len = ctx_size
    fields = {
        "node_idxs": batch["node_idxs"][0].tolist(),
        "is_padding": batch["is_padding"][0].tolist(),
        "is_task_nodes": batch["is_task_nodes"][0].tolist(),
        "is_targets": batch["is_targets"][0].tolist(),
        "sem_types": batch["sem_types"][0].tolist(),
        "col_name_idxs": batch["col_name_idxs"][0].tolist(),
        "table_name_idxs": batch["table_name_idxs"][0].tolist(),
        "class_value_idxs": batch["class_value_idxs"][0].tolist(),
        "timestamps": batch["timestamps"][0].tolist(),
        "f2p_nbr_idxs": batch["f2p_nbr_idxs"][0].tolist(),
        "number_values": batch["number_values"][0].squeeze(-1).float().tolist(),
        "datetime_values": batch["datetime_values"][0].squeeze(-1).float().tolist(),
        "boolean_values": batch["boolean_values"][0].squeeze(-1).float().tolist(),
        "seed_node_idxs": batch["seed_node_idxs"][0].tolist(),
        "bfs_depths": batch["bfs_depths"][0].tolist(),
        "batch_mask": batch["batch_mask"].tolist(),
    }

    # Build per-cell records.
    # Pre-compute table-name index → table-base-name lookup for legend coloring.
    table_offsets = sorted(
        (
            (info["node_idx_offset"], int(info["num_nodes"]), key)
            for key, info in table_info.items()
        ),
        key=lambda kv: kv[0],
    )

    def lookup_table_for_node(node_idx: int) -> tuple[str, str, bool]:
        """Return (display_name, full_key, is_task_table) for a node index.

        full_key is e.g. 'races:Db' or 'driver-position:Train'; display
        name is just 'races' / 'driver-position'.
        """
        prev_key = None
        for off, num, key in table_offsets:
            if node_idx >= off:
                prev_key = key
            else:
                break
        if prev_key is None:
            return "<unknown>", "", False
        base, _, ttype = prev_key.partition(":")
        return base, prev_key, (ttype != "Db")

    tokens = []
    real_token_count = 0
    primary_target_idx: int | None = None
    primary_target_token = None
    masked_feature_count = 0

    # f2p_nbr_idxs[0] is shape (S, MAX_F2P_NBRS) after `.tolist()` →
    # already a list of lists.
    f2p_arr = fields["f2p_nbr_idxs"]

    for i in range(seq_len):
        is_pad = bool(fields["is_padding"][i])
        node_idx = int(fields["node_idxs"][i])
        sem_type = SEM_TYPE_NAMES[int(fields["sem_types"][i])] if not is_pad else None
        col_name_idx = int(fields["col_name_idxs"][i])
        col_name = (
            text_vocab[col_name_idx]
            if 0 <= col_name_idx < len(text_vocab)
            else f"<idx:{col_name_idx}>"
        )
        # Strip the " of <table>" suffix for terse display.
        col_short = col_name.split(" of ", 1)[0] if " of " in col_name else col_name
        tbl_name_idx = int(fields["table_name_idxs"][i])
        tbl_text = (
            text_vocab[tbl_name_idx]
            if 0 <= tbl_name_idx < len(text_vocab)
            else f"<idx:{tbl_name_idx}>"
        )

        table_display, table_full, is_task_table = (
            lookup_table_for_node(node_idx) if not is_pad else ("", "", False)
        )

        f2p_list = [int(x) for x in f2p_arr[i] if int(x) != -1]

        ts = int(fields["timestamps"][i])
        ts_display = None if ts == INT_MIN else ts

        is_target = bool(fields["is_targets"][i])
        # Primary target = the cell whose node + column is the actual task
        # target (always emitted at position 0 by the rustler sampler). Any
        # other is_target flags came from `mask_prob` and are "masked
        # features": predicted alongside the primary target during training.
        is_primary = (
            is_target
            and not is_pad
            and col_name_idx == target_column_idx
            and (primary_target_idx is None)
        )

        seed_idx = int(fields["seed_node_idxs"][i])
        bfs_depth = int(fields["bfs_depths"][i])
        token: dict = {
            "i": i,
            "node_idx": node_idx,
            "is_padding": is_pad,
            "is_task_node": bool(fields["is_task_nodes"][i]),
            "is_target": is_target,
            "is_primary_target": is_primary,
            "is_masked_feature": is_target and not is_primary and not is_pad,
            "sem_type": sem_type,
            "col_name": col_name,
            "col_short": col_short,
            "col_name_idx": col_name_idx,
            "table_name_idx": tbl_name_idx,
            "table_text": tbl_text,
            "table_display": table_display,
            "table_full_key": table_full,
            "is_task_table": is_task_table,
            "class_value_idx": int(fields["class_value_idxs"][i]),
            "number_value": float(fields["number_values"][i]),
            "datetime_value": float(fields["datetime_values"][i]),
            "boolean_value": float(fields["boolean_values"][i]),
            "f2p_nbr_idxs": f2p_list,
            "timestamp": ts_display,
            "seed_node_idx": seed_idx if seed_idx != -1 else None,
            "bfs_depth": bfs_depth if bfs_depth != -1 else None,
        }
        if not is_pad:
            real_token_count += 1
            decoded = _decode_value(sem_type, token, text_vocab)
            token["value"] = decoded
            token["value_str"] = _format_value(sem_type, decoded)
            if is_primary:
                primary_target_idx = i
                primary_target_token = token
            elif token["is_masked_feature"]:
                masked_feature_count += 1
        else:
            token["value"] = None
            token["value_str"] = ""
        tokens.append(token)

    # Group cells by node for the row-view.
    by_node: "OrderedDict[int, list[int]]" = OrderedDict()
    for t in tokens:
        if t["is_padding"]:
            continue
        by_node.setdefault(t["node_idx"], []).append(t["i"])

    nodes_meta = []
    for node_idx, idxs in by_node.items():
        first = tokens[idxs[0]]
        # All cells of a node share the same seed/depth (added together
        # in one inner loop per BFS hit), so the first cell's values are
        # canonical for the whole node.
        nodes_meta.append(
            {
                "node_idx": node_idx,
                "table_display": first["table_display"],
                "table_full_key": first["table_full_key"],
                "is_task_table": first["is_task_table"],
                "is_target_node": any(tokens[k]["is_target"] for k in idxs),
                "is_task_node": any(tokens[k]["is_task_node"] for k in idxs),
                "is_primary_target_node": any(
                    tokens[k]["is_primary_target"] for k in idxs
                ),
                "is_seed": first["seed_node_idx"] == node_idx,
                "seed_node_idx": first["seed_node_idx"],
                "bfs_depth": first["bfs_depth"],
                "timestamp": first["timestamp"],
                "cell_idxs": idxs,
            }
        )

    # Per-seed aggregate. Used by the radial-shell graph layout: the target
    # seed sits at center, other seeds sit on a ring around it, and each
    # seed's BFS expansion sits in concentric shells around that seed.
    by_seed: "OrderedDict[int, list[int]]" = OrderedDict()
    for n in nodes_meta:
        s = n["seed_node_idx"]
        if s is None:
            continue
        by_seed.setdefault(s, []).append(n["node_idx"])
    primary_target_node_idx = (
        primary_target_token["node_idx"] if primary_target_token else None
    )
    seeds_meta: list[dict] = []
    for seed_idx, member_node_idxs in by_seed.items():
        seed_node_meta = next(
            (n for n in nodes_meta if n["node_idx"] == seed_idx), None
        )
        depths = [
            n["bfs_depth"]
            for n in nodes_meta
            if n["seed_node_idx"] == seed_idx and n["bfs_depth"] is not None
        ]
        seeds_meta.append(
            {
                "seed_node_idx": seed_idx,
                "is_target_seed": seed_idx == primary_target_node_idx,
                "table_display": (
                    seed_node_meta["table_display"] if seed_node_meta else None
                ),
                "is_task_table": (
                    seed_node_meta["is_task_table"] if seed_node_meta else False
                ),
                "max_depth": max(depths) if depths else 0,
                "node_count": len(member_node_idxs),
                "member_node_idxs": member_node_idxs,
            }
        )
    seeds_meta.sort(key=lambda s: (not s["is_target_seed"], -s["node_count"]))

    # Group cells by column key (table+col) for the column-view.
    by_col: "OrderedDict[tuple[str, str], list[int]]" = OrderedDict()
    for t in tokens:
        if t["is_padding"]:
            continue
        key = (t["table_display"], t["col_short"])
        by_col.setdefault(key, []).append(t["i"])
    cols_meta = []
    for (tbl, col), idxs in by_col.items():
        sem_types = sorted(
            {tokens[k]["sem_type"] for k in idxs if tokens[k]["sem_type"]}
        )
        cols_meta.append(
            {
                "table_display": tbl,
                "col_short": col,
                "sem_types": sem_types,
                "cell_idxs": idxs,
            }
        )

    # Build the graph payload (for the network view).
    node_set = {n["node_idx"] for n in nodes_meta}
    edges = []
    seen_edges: set[tuple[int, int]] = set()
    for t in tokens:
        if t["is_padding"]:
            continue
        for nbr in t["f2p_nbr_idxs"]:
            if nbr in node_set and nbr != t["node_idx"]:
                e = (t["node_idx"], nbr)
                if e not in seen_edges:
                    seen_edges.add(e)
                    edges.append({"source": t["node_idx"], "target": nbr})

    # Per-table summary (for legend coloring).
    table_summary: dict[str, dict] = {}
    for n in nodes_meta:
        s = table_summary.setdefault(
            n["table_display"],
            {
                "table_display": n["table_display"],
                "is_task": n["is_task_table"],
                "node_count": 0,
                "cell_count": 0,
            },
        )
        s["node_count"] += 1
        s["cell_count"] += len(n["cell_idxs"])

    # Sem-type counts for stats.
    sem_counts = {st: 0 for st in SEM_TYPE_NAMES}
    target_count = 0
    task_token_count = 0
    for t in tokens:
        if t["is_padding"]:
            continue
        if t["sem_type"]:
            sem_counts[t["sem_type"]] += 1
        if t["is_target"]:
            target_count += 1
        if t["is_task_node"]:
            task_token_count += 1

    payload = {
        "request": {
            "db_name": db_name,
            "table_name": table_name,
            "target_column": target_column,
            "target_column_idx": target_column_idx,
            "split": split,
            "ctx_size": ctx_size,
            "item_idx": item_idx,
            "context_seed": context_seed,
            "mask_prob_max": mask_prob_max,
        },
        "num_items": ds.sampler.num_items,
        "tokens": tokens,
        "nodes_meta": nodes_meta,
        "cols_meta": cols_meta,
        "seeds_meta": seeds_meta,
        "table_summary": list(table_summary.values()),
        "graph": {
            "node_ids": list(node_set),
            "edges": edges,
        },
        "stats": {
            "seq_len": seq_len,
            "real_tokens": real_token_count,
            "padding_tokens": seq_len - real_token_count,
            "padding_pct": round((seq_len - real_token_count) / seq_len * 100, 2),
            "num_nodes": len(nodes_meta),
            "num_columns": len(cols_meta),
            "num_tables": len(table_summary),
            "num_edges": len(edges),
            "num_seeds": len(seeds_meta),
            "sem_counts": sem_counts,
            "target_count": target_count,
            "task_token_count": task_token_count,
            "masked_feature_count": masked_feature_count,
        },
        "target_token_idx": primary_target_idx,
        "target_token": primary_target_token,
        "max_f2p_nbrs": MAX_F2P_NBRS,
    }
    return payload


# ---------------------------------------------------------------------------
# Helpers for browsing what's available on disk.
# ---------------------------------------------------------------------------


def _list_pre_roots(top_root: Path) -> list[str]:
    """Return immediate sub-dirs of `top_root` that look like 'pre roots'.

    A pre-root is a directory containing one or more dataset directories,
    each with a `nodes.rkyv` file. We also include `top_root` itself if it
    has an immediate dataset (e.g., when DEFAULT_PRE_ROOT *is* the root).
    """
    roots: list[str] = []
    if not top_root.exists():
        return roots
    # Always offer top_root itself.
    if any(
        (top_root / d / "nodes.rkyv").exists()
        for d in _safe_iterdir(top_root)
        if (top_root / d).is_dir()
    ):
        roots.append(str(top_root))
    for d in _safe_iterdir(top_root):
        sub = top_root / d
        if not sub.is_dir():
            continue
        if (sub / "nodes.rkyv").exists():
            continue  # this is a dataset dir, not a root
        # Check whether sub has dataset children.
        for c in _safe_iterdir(sub):
            if (sub / c / "nodes.rkyv").exists():
                roots.append(str(sub))
                break
    return sorted(set(roots))


def _safe_iterdir(p: Path) -> list[str]:
    try:
        return [c.name for c in p.iterdir()]
    except (FileNotFoundError, PermissionError):
        return []


def _list_dbs_in_root(pre_root: Path) -> list[str]:
    out: list[str] = []
    if not pre_root.exists():
        return out
    for child in sorted(_safe_iterdir(pre_root)):
        full = pre_root / child
        if not full.is_dir():
            continue
        if (full / "nodes.rkyv").exists():
            out.append(child)
        else:
            for grand in sorted(_safe_iterdir(full)):
                gfull = full / grand
                if (gfull / "nodes.rkyv").exists():
                    out.append(f"{child}/{grand}")
    return out


def _root_is_hf(root_arg: str) -> bool:
    """A root spec is HuggingFace-hosted if it is not an existing local dir."""
    return not Path(root_arg).expanduser().exists()


def _list_dbs(root_arg: str) -> list[str]:
    """Databases under a root, whether a local directory or a HuggingFace repo."""
    if not _root_is_hf(root_arg):
        return _list_dbs_in_root(Path(root_arg).expanduser())
    from huggingface_hub import HfApi

    repo_id, subdir = resolve_repo(root_arg)
    prefix = f"{subdir}/" if subdir else ""
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return sorted(
        {
            f[len(prefix) :].split("/", 1)[0]
            for f in files
            if f.startswith(prefix) and f.endswith("/nodes.rkyv")
        }
    )


def _resolve_db_dir(
    root_arg: str,
    db: str,
    *,
    metadata_only: bool,
    embedding_model: str = "all-MiniLM-L12-v2",
) -> Path:
    """Return a LOCAL directory for `db`, downloading from HF on demand.

    For HF roots, `metadata_only=True` fetches just the schema files (fast
    browsing); `False` fetches the node blobs + embeddings needed to build
    contexts. Local roots are used in place.
    """
    if not _root_is_hf(root_arg):
        return Path(root_arg).expanduser() / db
    local_root = resolve_pre_dir(
        root_arg,
        [db],
        embedding_model,
        include_text=not metadata_only,
        metadata_only=metadata_only,
    )
    return Path(local_root) / db


def _list_tables(db_dir: Path, split: str) -> list[dict]:
    """Tables in this DB available for the given split (or 'auto' for any)."""
    info = _load_table_info(str(db_dir))
    by_base: dict[str, dict[str, dict]] = {}
    for full_key, meta in info.items():
        base, _, ttype = full_key.partition(":")
        by_base.setdefault(base, {})[ttype] = meta
    out = []
    for base, ttypes in sorted(by_base.items()):
        item = {
            "name": base,
            "splits": sorted(ttypes.keys()),
            "is_task_table": any(t != "Db" for t in ttypes.keys()),
            "num_nodes_by_split": {
                t: int(meta["num_nodes"]) for t, meta in ttypes.items()
            },
        }
        out.append(item)
    return out


def _columns_for_table(db_dir: Path, table_name: str) -> list[str]:
    """All columns belonging to `table_name`, parsed from column_index.json."""
    ci = _load_column_index(str(db_dir))
    out = []
    for key, idx in ci.items():
        col, _, tbl = key.partition(" of ")
        if tbl == table_name:
            out.append({"name": col, "idx": int(idx)})
    return sorted(out, key=lambda d: d["name"])


# ---------------------------------------------------------------------------
# HTTP server.
# ---------------------------------------------------------------------------


class CtxVizServer(ThreadingHTTPServer):
    # Re-bind without TIME_WAIT delay after a previous run on the same port
    # exits — the common cause of EADDRINUSE on quick restart cycles.
    allow_reuse_address = True


class CtxVizHandler(BaseHTTPRequestHandler):
    server_version = "ctx-viz/0.1"

    # Suppress per-request stderr logging (override of default).
    def log_message(self, format, *args):  # noqa: A002
        if self.server.quiet:
            return
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    # ---- routing ----

    def do_GET(self):  # noqa: N802
        try:
            url = urlparse(self.path)
            if url.path == "/":
                self._send_html(INDEX_HTML)
            elif url.path == "/api/list_pre_roots":
                root_arg = str(self.server.root_arg)
                if _root_is_hf(root_arg):
                    self._send_json({"default_root": root_arg, "pre_roots": [root_arg]})
                else:
                    pre_root = Path(root_arg).expanduser()
                    self._send_json(
                        {
                            "default_root": str(pre_root),
                            "pre_roots": _list_pre_roots(pre_root.parent)
                            + ([str(pre_root)] if pre_root.exists() else []),
                        }
                    )
            elif url.path == "/api/list_dbs":
                qs = parse_qs(url.query)
                root = qs.get("root", [str(self.server.root_arg)])[0]
                self._send_json({"dbs": _list_dbs(root)})
            elif url.path == "/api/list_tables":
                qs = parse_qs(url.query)
                root = qs.get("root", [str(self.server.root_arg)])[0]
                db = qs["db"][0]
                db_dir = _resolve_db_dir(root, db, metadata_only=True)
                tables = _list_tables(db_dir, split=qs.get("split", ["auto"])[0])
                self._send_json({"tables": tables})
            elif url.path == "/api/list_columns":
                qs = parse_qs(url.query)
                root = qs.get("root", [str(self.server.root_arg)])[0]
                db = qs["db"][0]
                tbl = qs["table"][0]
                db_dir = _resolve_db_dir(root, db, metadata_only=True)
                self._send_json({"columns": _columns_for_table(db_dir, tbl)})
            elif url.path == "/api/health":
                self._send_json({"ok": True})
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"unknown path: {url.path}")
        except BaseException as e:  # incl. pyo3 PanicException
            self._send_error(e)

    def do_POST(self):  # noqa: N802
        try:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            req = json.loads(body or b"{}")
            if url.path == "/api/build":
                payload = self._handle_build(req)
                self._send_json(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"unknown path: {url.path}")
        except BaseException as e:  # incl. pyo3 PanicException
            self._send_error(e)

    # ---- handlers ----

    def _handle_build(self, req: dict) -> dict:
        root_arg = req.get("pre_root") or str(self.server.root_arg)
        db_name = req["db_name"]
        table_name = req["table_name"]
        target_column = req["target_column"]
        split = req.get("split", "val")
        columns_to_drop = tuple(req.get("columns_to_drop") or [])
        ctx_size = int(req.get("ctx_size", 256))
        local_ctx_size = int(req.get("local_ctx_size", ctx_size))
        bfs_width = int(req.get("bfs_width", 128))
        num_walks = int(req.get("num_walks", 0))
        walk_length = int(req.get("walk_length", 4))
        prefer_latest = bool(req.get("prefer_latest", False))
        embedding_model = req.get("embedding_model", "all-MiniLM-L12-v2")
        d_text = int(req.get("d_text", 384))
        items_per_task = int(req.get("items_per_task", -1))
        bool_as_num = bool(req.get("bool_as_num", False))
        skip_text_cols = bool(req.get("skip_text_cols", False))
        balance_labels = bool(req.get("balance_labels", False))
        ablate_schema_semantics = bool(req.get("ablate_schema_semantics", False))
        shuffle_seed = int(req.get("shuffle_seed", 0))
        context_seed = int(req.get("context_seed", 0))
        mask_prob_max = float(req.get("mask_prob_max", 0.0))
        item_idx = req.get("item_idx", None)
        if item_idx is not None and item_idx != "":
            item_idx = int(item_idx)
        else:
            item_idx = None

        db_dir = _resolve_db_dir(
            root_arg, db_name, metadata_only=False, embedding_model=embedding_model
        )
        if not db_dir.exists():
            raise FileNotFoundError(f"db dir does not exist: {db_dir}")
        pre_root = db_dir.parent

        ds = _get_or_build_dataset(
            pre_root=str(pre_root),
            db_name=db_name,
            table_name=table_name,
            target_column=target_column,
            split=split,
            columns_to_drop=columns_to_drop,
            ctx_cap=ctx_size,
            local_ctx_size=local_ctx_size,
            bfs_width=bfs_width,
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest=prefer_latest,
            embedding_model=embedding_model,
            d_text=d_text,
            items_per_task=items_per_task,
            bool_as_num=bool_as_num,
            skip_text_cols=skip_text_cols,
            balance_labels=balance_labels,
            ablate_schema_semantics=ablate_schema_semantics,
            shuffle_seed=shuffle_seed,
        )

        target_column_idx = get_column_index(
            target_column, table_name, db_name, str(pre_root)
        )

        return _build_context_payload(
            ds=ds,
            db_name=db_name,
            db_dir=str(db_dir),
            text_vocab=_load_text_json(str(db_dir)),
            table_info=_load_table_info(str(db_dir)),
            column_index=_load_column_index(str(db_dir)),
            item_idx=item_idx,
            bs=1,
            ctx_size=ctx_size,
            context_seed=context_seed,
            mask_prob_max=mask_prob_max,
            target_column=target_column,
            target_column_idx=target_column_idx,
            table_name=table_name,
            split=split,
        )

    # ---- response helpers ----

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj) -> None:
        body = json.dumps(obj, default=_json_default, allow_nan=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, exc: BaseException) -> None:
        traceback.print_exc()
        body = json.dumps(
            {
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        ).encode("utf-8")
        self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _json_default(o):
    """Coerce numpy scalars and convert non-finite floats to None."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def _bind_with_fallback(host: str, port: int, max_tries: int) -> CtxVizServer:
    """Try `port`, then fall through to free ports if it's taken.

    On a shared box another user can hold the requested port; rather than
    erroring out, walk forward up to `max_tries` ports and finally fall
    back to OS-assigned (port=0). The actual port is read off the bound
    socket, so the printed URL reflects what was used.
    """
    last_err: OSError | None = None
    for offset in range(max_tries):
        try:
            return CtxVizServer((host, port + offset), CtxVizHandler)
        except OSError as e:
            if e.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            last_err = e
            continue
    # Final fallback: let the OS pick.
    try:
        return CtxVizServer((host, 0), CtxVizHandler)
    except OSError as e:
        raise (last_err or e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Bind to all interfaces by default so a browser on a laptop can hit a
    # remote workstation directly (e.g. http://<remote-host>:8765).
    # If you'd rather keep it local-only, pass `--host 127.0.0.1`.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--pre-root",
        default=str(DEFAULT_PRE_ROOT),
        help="directory containing your pre-processed datasets",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-request HTTP logs",
    )
    parser.add_argument(
        "--no-port-fallback",
        action="store_true",
        help="fail loudly if --port is unavailable instead of trying nearby ports",
    )
    args = parser.parse_args()

    pre_root = Path(args.pre_root).expanduser()
    if args.no_port_fallback:
        server = CtxVizServer((args.host, args.port), CtxVizHandler)
    else:
        server = _bind_with_fallback(args.host, args.port, max_tries=20)
    server.pre_root = pre_root  # type: ignore[attr-defined]
    server.root_arg = args.pre_root  # raw spec: local dir or HF repo
    server.quiet = args.quiet  # type: ignore[attr-defined]

    actual_port = server.server_address[1]
    if actual_port != args.port:
        print(
            f"\n  \033[33m[note]\033[0m port {args.port} was taken; using {actual_port}",
            flush=True,
        )
    print(f"\n  ctx-viz running on {args.host}:{actual_port}", flush=True)
    if args.host in ("0.0.0.0", "::"):
        fqdn = socket.getfqdn()
        print(
            f"    local:   \033[1;36mhttp://127.0.0.1:{actual_port}\033[0m",
            flush=True,
        )
        print(
            f"    network: \033[1;36mhttp://{fqdn}:{actual_port}\033[0m",
            flush=True,
        )
    else:
        print(
            f"    open:    \033[1;36mhttp://{args.host}:{actual_port}\033[0m",
            flush=True,
        )
    print(f"  pre-root: {pre_root}\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


# ---------------------------------------------------------------------------
# Frontend (single-page HTML, embedded for zero-deps deploy).
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ctx-viz</title>
<script>
  // Apply saved theme before first paint to avoid a flash. Default = light.
  (function () {
    var t = null;
    try { t = localStorage.getItem("ctxviz-theme"); } catch (e) {}
    if (t === "dark") document.documentElement.dataset.theme = "dark";
  })();
</script>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  /* ----- design tokens (light = default, dark = data-theme="dark") ----- */
  :root {
    /* light theme */
    --bg: #f7f8fa;
    --bg-2: #ffffff;
    --bg-3: #f0f2f5;
    --bg-4: #e6e9ee;
    --line: #d8dde5;
    --line-2: #c2c8d2;
    --text: #1f2430;
    --text-dim: #525968;
    --text-faint: #8990a0;
    --accent: #0969da;
    --accent-2: #8250df;
    --good: #1a7f37;
    --warn: #bf8700;
    --bad: #cf222e;
    --target: #c97800;
    --task: #8250df;
    --pad: #c2c8d2;
    --code-bg: #eef0f3;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
    --radius: 6px;
    --mono: ui-monospace, "JetBrains Mono", "Fira Code", "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter",
            "Segoe UI", system-ui, sans-serif;
  }
  :root[data-theme="dark"] {
    --bg: #0f1115;
    --bg-2: #161922;
    --bg-3: #1d2230;
    --bg-4: #232a3b;
    --line: #2d3548;
    --line-2: #3a445b;
    --text: #d8dee9;
    --text-dim: #9aa3b8;
    --text-faint: #6b758c;
    --accent: #79b8ff;
    --accent-2: #b392f0;
    --good: #85e89d;
    --warn: #ffab70;
    --bad: #f97583;
    --target: #ffd166;
    --task: #b392f0;
    --pad: #404a63;
    --code-bg: #0b0d11;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 13px;
    overflow: hidden;
  }
  button, input, select, textarea {
    font-family: inherit;
    font-size: inherit;
  }
  /* ----- layout ----- */
  #app {
    display: grid;
    grid-template-columns: 320px 1fr;
    grid-template-rows: 100vh;
    height: 100vh;
  }
  #sidebar {
    background: var(--bg-2);
    border-right: 1px solid var(--line);
    overflow-y: auto;
    padding: 14px 14px 64px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  #main {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #topbar {
    background: var(--bg-2);
    border-bottom: 1px solid var(--line);
    padding: 8px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-shrink: 0;
  }
  #stats { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  .stat {
    display: flex; flex-direction: column; align-items: flex-start;
    line-height: 1.05;
  }
  .stat .v { color: var(--text); font-weight: 600; font-family: var(--mono); }
  .stat .l { color: var(--text-faint); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }

  #tabs {
    display: flex;
    gap: 0;
    background: var(--bg-2);
    border-bottom: 1px solid var(--line);
    flex-shrink: 0;
  }
  .tab {
    padding: 8px 14px;
    color: var(--text-dim);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    user-select: none;
  }
  .tab:hover { color: var(--text); background: var(--bg-3); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  #content {
    flex: 1;
    overflow: hidden;
    position: relative;
  }
  .pane {
    display: none;
    height: 100%;
    width: 100%;
    overflow: auto;
  }
  .pane.active { display: block; }

  /* ----- sidebar form ----- */
  h1 {
    font-size: 14px;
    font-weight: 700;
    margin: 0 0 4px;
    letter-spacing: 0.04em;
  }
  h1 small { color: var(--text-faint); font-weight: 400; font-size: 11px; }
  fieldset {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 8px 10px 10px;
    margin: 0;
    background: var(--bg-3);
  }
  legend {
    padding: 0 6px;
    color: var(--text-dim);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }
  label.row {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 8px;
  }
  label.row > span {
    color: var(--text-dim);
    font-size: 11px;
  }
  label.row.flex {
    flex-direction: row;
    align-items: center;
    gap: 8px;
  }
  label.row.flex > span { flex: 1; }
  label.row > span code { color: var(--accent-2); }
  input[type=text], input[type=number], select, textarea {
    background: var(--bg-4);
    color: var(--text);
    border: 1px solid var(--line-2);
    border-radius: 4px;
    padding: 5px 7px;
    width: 100%;
    outline: none;
  }
  input[type=text]:focus, input[type=number]:focus, select:focus {
    border-color: var(--accent);
  }
  input[type=checkbox] { accent-color: var(--accent); }
  button.primary, button.action {
    background: var(--accent);
    color: #0a0d12;
    border: none;
    border-radius: 4px;
    padding: 8px 12px;
    cursor: pointer;
    font-weight: 600;
  }
  button.primary:hover, button.action:hover { filter: brightness(1.08); }
  button.ghost {
    background: transparent;
    border: 1px solid var(--line-2);
    color: var(--text);
    border-radius: 4px;
    padding: 6px 8px;
    cursor: pointer;
  }
  button.ghost:hover { background: var(--bg-4); border-color: var(--line); }
  .btn-row { display: flex; gap: 8px; }
  .helper { color: var(--text-faint); font-size: 11px; margin-top: 2px; }

  /* ----- legend ----- */
  #legend {
    padding: 6px 14px;
    background: var(--bg-2);
    border-bottom: 1px solid var(--line);
    display: flex;
    gap: 14px;
    align-items: center;
    flex-wrap: wrap;
    flex-shrink: 0;
    max-height: 100px;
    overflow: auto;
  }
  .swatch {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-dim);
    font-size: 12px;
  }
  .swatch .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }
  .swatch.task .dot { border: 2px solid var(--task); border-radius: 2px; }
  .swatch.target .dot { border: 2px solid var(--target); }

  /* ----- graph ----- */
  #graph-pane {
    background: var(--bg);
    position: relative;
  }
  #graph-svg { width: 100%; height: 100%; cursor: grab; }
  #graph-svg.dragging { cursor: grabbing; }
  #graph-svg .link {
    stroke: var(--text-faint);
    stroke-opacity: 0.3;
    stroke-width: 1;
  }
  #graph-svg .node circle, #graph-svg .node rect {
    stroke: var(--line-2);
    stroke-width: 1;
  }
  #graph-svg .node.target circle, #graph-svg .node.target rect {
    stroke: var(--target);
    stroke-width: 3;
  }
  #graph-svg .node text {
    fill: var(--text-faint);
    font-size: 10px;
    font-family: var(--mono);
    pointer-events: none;
  }
  #graph-svg .node.target text {
    fill: var(--target);
    font-weight: 700;
  }
  #graph-overlay {
    position: absolute;
    bottom: 12px; left: 12px;
    background: var(--bg-2);
    opacity: 0.92;
    border: 1px solid var(--line);
    padding: 8px 10px;
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-dim);
    backdrop-filter: blur(4px);
    pointer-events: none;
  }

  /* ----- sequence pane ----- */
  #sequence-pane { padding: 12px; }
  .seq-tape {
    width: 100%;
    height: 38px;
    margin-bottom: 10px;
    border: 1px solid var(--line);
    border-radius: 3px;
    background: var(--bg-2);
    display: block;
  }
  .seq-tape .tape-cell { cursor: pointer; }
  .seq-tape-helper { color: var(--text-faint); font-size: 11px; margin-bottom: 6px; }
  .cell-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(74px, 1fr));
    gap: 4px;
  }
  .cell {
    background: var(--bg-3);
    border: 1px solid var(--line);
    border-left: 3px solid var(--pad);
    border-radius: 3px;
    padding: 4px 6px;
    font-family: var(--mono);
    font-size: 10.5px;
    line-height: 1.25;
    color: var(--text);
    cursor: pointer;
    overflow: hidden;
    min-height: 56px;
    transition: transform 0.05s linear, border-color 0.05s linear;
  }
  .cell:hover {
    transform: translateY(-1px);
    border-color: var(--accent);
  }
  .cell.padding {
    background: transparent;
    color: var(--text-faint);
    border-left-color: var(--pad);
    opacity: 0.4;
  }
  .cell.target {
    background: linear-gradient(0deg, rgba(255, 209, 102, 0.30), rgba(255, 209, 102, 0.10));
    border-color: var(--target);
    box-shadow: inset 0 0 0 2px var(--target);
  }
  .cell.masked {
    background: linear-gradient(0deg, rgba(179, 146, 240, 0.18), rgba(179, 146, 240, 0.06));
    border-style: dashed;
    border-color: var(--task);
  }
  .cell.task { border-top: 2px dashed var(--task); }
  .cell .head {
    color: var(--text-faint);
    font-size: 9px;
    display: flex;
    justify-content: space-between;
  }
  .cell .col { color: var(--text-dim); font-size: 9.5px; margin-top: 1px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cell .val { color: var(--text); margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cell .sem { font-size: 9px; color: var(--text-faint); }
  .badge {
    display: inline-block; padding: 0 4px;
    background: var(--bg-4); color: var(--text-dim);
    border-radius: 3px; font-size: 9px;
    font-family: var(--mono);
  }
  .badge.target { background: var(--target); color: #2c1f00; }
  .badge.task { background: var(--task); color: #1a0d33; }

  /* ----- by-row pane ----- */
  #row-pane { padding: 12px; display: flex; flex-direction: column; gap: 10px; }
  .node-row {
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .node-row .row-head {
    padding: 6px 10px;
    background: var(--bg-3);
    border-bottom: 1px solid var(--line);
    display: flex;
    gap: 10px;
    align-items: center;
  }
  .node-row.target .row-head { box-shadow: inset 4px 0 0 var(--target); }
  .node-row.task .row-head { box-shadow: inset 4px 0 0 var(--task); }
  .node-row.target.task .row-head { box-shadow: inset 4px 0 0 var(--target), inset 8px 0 0 var(--task); }
  .row-head .table-name {
    font-family: var(--mono);
    font-weight: 600;
  }
  .row-head .meta { color: var(--text-faint); font-size: 11px; font-family: var(--mono); }
  .cell-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 0;
  }
  .cell-list .kv {
    padding: 4px 8px;
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    overflow: hidden;
  }
  .cell-list .kv:hover { background: var(--bg-3); }
  .cell-list .kv.target { background: rgba(255, 209, 102, 0.07); border-bottom-color: var(--target); }
  .cell-list .kv .k { color: var(--text-dim); }
  .cell-list .kv .v { color: var(--text); }
  .cell-list .kv .sem { color: var(--text-faint); font-size: 9px; }

  /* ----- by-column pane ----- */
  #col-pane { padding: 12px; display: flex; flex-direction: column; gap: 8px; }
  .col-block {
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 8px 10px;
  }
  .col-block .col-head {
    display: flex; gap: 10px; align-items: baseline;
    border-bottom: 1px solid var(--line);
    padding-bottom: 4px; margin-bottom: 6px;
  }
  .col-block .col-head .name { font-family: var(--mono); font-weight: 600; }
  .col-block .col-head .table { color: var(--text-faint); font-family: var(--mono); font-size: 11px; }
  .col-vals {
    display: flex; flex-wrap: wrap; gap: 4px;
  }
  .col-vals .pill {
    padding: 2px 6px;
    background: var(--bg-3);
    border: 1px solid var(--line);
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
  }
  .col-vals .pill:hover { background: var(--bg-4); }
  .col-vals .pill.target { background: rgba(255, 209, 102, 0.16); border-color: var(--target); color: var(--target); }
  .col-numdist {
    height: 28px;
    margin-top: 4px;
  }

  /* ----- raw tensors pane ----- */
  #raw-pane { padding: 12px; font-family: var(--mono); font-size: 11px; }
  .raw-table { width: 100%; border-collapse: collapse; }
  .raw-table th {
    background: var(--bg-3);
    color: var(--text-dim);
    text-align: left;
    padding: 4px 6px;
    border-bottom: 1px solid var(--line);
    position: sticky;
    top: 0;
  }
  .raw-table td {
    padding: 2px 6px;
    border-bottom: 1px solid var(--bg-3);
  }
  .raw-table tr:hover { background: var(--bg-2); }
  .raw-table tr.target { background: rgba(255, 209, 102, 0.07); }
  .raw-table tr.padding { color: var(--text-faint); }
  .raw-table .num { text-align: right; }
  .raw-table .center { text-align: center; }

  /* ----- attention masks pane ----- */
  #masks-pane { padding: 12px; }
  .masks-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .masks-grid .mask-block {
    background: var(--bg-2);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 8px 10px;
  }
  .mask-block h4 { margin: 0 0 4px; font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em;}
  .mask-block p { margin: 0; color: var(--text-faint); font-size: 11px; }
  .mask-canvas { display: block; margin-top: 6px; image-rendering: pixelated; width: 100%; height: auto; max-height: 320px; }

  /* ----- popover (cell details) ----- */
  #popover {
    position: absolute;
    background: var(--bg-3);
    border: 1px solid var(--line-2);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 8px 10px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text);
    pointer-events: none;
    z-index: 10;
    max-width: 360px;
    display: none;
    line-height: 1.4;
  }
  #popover .row { display: flex; gap: 8px; }
  #popover .row .k { color: var(--text-faint); min-width: 90px; }
  #popover .row .v { color: var(--text); flex: 1; word-break: break-word; }

  /* ----- spinner ----- */
  #spinner {
    display: none;
    position: absolute; inset: 0;
    background: var(--bg);
    opacity: 0.7;
    align-items: center; justify-content: center;
    z-index: 20;
  }
  #spinner.show { display: flex; }
  .spinner-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1s infinite ease-in-out;
    margin: 0 4px;
  }
  .spinner-dot:nth-child(2) { animation-delay: 0.15s; }
  .spinner-dot:nth-child(3) { animation-delay: 0.3s; }
  @keyframes pulse { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }

  /* ----- error banner ----- */
  #error {
    display: none;
    background: var(--bad);
    color: #2a0808;
    padding: 8px 12px;
    font-family: var(--mono);
    font-size: 12px;
    white-space: pre-wrap;
    border-radius: 4px;
    margin: 6px 14px;
  }
  #error.show { display: block; }

  /* ----- mini scrollbars ----- */
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 5px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

  /* ----- empty state ----- */
  .empty {
    display: flex;
    flex-direction: column;
    height: 100%;
    align-items: center;
    justify-content: center;
    color: var(--text-faint);
    text-align: center;
    padding: 32px;
  }
  .empty h3 { margin: 0 0 8px; color: var(--text-dim); font-weight: 600; }
  .empty p { margin: 0; max-width: 480px; }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <h1>ctx-viz <small>v0.1</small></h1>
    <div class="helper">Visualizes a single rustler batch tensor for inspection.</div>

    <fieldset>
      <legend>Dataset</legend>
      <label class="row"><span>Pre-root</span>
        <select id="f-pre-root"></select>
      </label>
      <label class="row"><span>Database</span>
        <select id="f-db"></select>
      </label>
      <label class="row"><span>Table (task)</span>
        <select id="f-table"></select>
      </label>
      <label class="row"><span>Split</span>
        <select id="f-split"></select>
      </label>
      <label class="row"><span>Target column</span>
        <select id="f-target"></select>
      </label>
      <label class="row"><span>Drop columns (leakage)</span>
        <input type="text" id="f-drop" placeholder="comma-separated" />
      </label>
    </fieldset>

    <fieldset>
      <legend>Context</legend>
      <label class="row"><span>ctx_size</span>
        <input type="number" id="f-ctx" value="256" min="8" max="32768" />
      </label>
      <label class="row"><span>local_ctx_size</span>
        <input type="number" id="f-local" value="256" min="8" max="32768" />
      </label>
      <label class="row"><span>bfs_width</span>
        <input type="number" id="f-bfs" value="128" min="1" max="32768" />
      </label>
      <label class="row"><span>num_walks</span>
        <input type="number" id="f-walks" value="0" min="0" max="100000" />
      </label>
      <label class="row"><span>walk_length</span>
        <input type="number" id="f-walklen" value="4" min="1" max="100" />
      </label>
      <label class="row flex">
        <input type="checkbox" id="f-prefer-latest" />
        <span>prefer_latest</span>
      </label>
      <label class="row flex">
        <input type="checkbox" id="f-skip-text" />
        <span>skip_text_cols</span>
      </label>
      <label class="row flex">
        <input type="checkbox" id="f-balance-labels" />
        <span>balance_labels</span>
      </label>
      <label class="row flex">
        <input type="checkbox" id="f-bool-as-num" />
        <span>bool_as_num</span>
      </label>
    </fieldset>

    <fieldset>
      <legend>Item & Seed</legend>
      <label class="row"><span>item_idx <span class="helper" id="f-item-helper"></span></span>
        <input type="number" id="f-item" value="0" min="0" />
      </label>
      <label class="row"><span>shuffle_seed</span>
        <input type="number" id="f-shuffle" value="0" />
      </label>
      <label class="row"><span>context_seed</span>
        <input type="number" id="f-ctxseed" value="0" />
      </label>
      <label class="row"><span>mask_prob_max</span>
        <input type="number" id="f-mask" value="0.0" step="0.05" min="0" max="1" />
      </label>
      <label class="row"><span>items_per_task</span>
        <input type="number" id="f-ipt" value="-1" />
      </label>
    </fieldset>

    <fieldset>
      <legend>Embedding</legend>
      <label class="row"><span>embedding_model</span>
        <input type="text" id="f-emb" value="all-MiniLM-L12-v2" />
      </label>
      <label class="row"><span>d_text</span>
        <input type="number" id="f-dtext" value="384" />
      </label>
    </fieldset>

    <div class="btn-row">
      <button class="primary" id="f-build" style="flex:1;">Build context</button>
      <button class="ghost" id="f-prev" title="prev item">◀</button>
      <button class="ghost" id="f-next" title="next item">▶</button>
      <button class="ghost" id="f-rand" title="random item">⚄</button>
    </div>
  </aside>

  <main id="main">
    <div id="topbar">
      <div id="stats"></div>
      <div id="actions" style="display:flex; gap:8px; align-items:center;">
        <button class="ghost" id="f-theme" title="toggle light/dark theme" style="min-width:36px;">☀</button>
        <button class="ghost" id="f-export" title="export current payload as JSON">Export JSON</button>
      </div>
    </div>
    <div id="error"></div>
    <div id="legend"></div>
    <div id="tabs">
      <div class="tab active" data-pane="graph-pane">Graph</div>
      <div class="tab" data-pane="sequence-pane">Sequence</div>
      <div class="tab" data-pane="row-pane">By Row</div>
      <div class="tab" data-pane="col-pane">By Column</div>
      <div class="tab" data-pane="masks-pane">Attention Masks</div>
      <div class="tab" data-pane="raw-pane">Raw Tokens</div>
    </div>
    <div id="content">
      <div id="graph-pane" class="pane active">
        <svg id="graph-svg"></svg>
        <div id="graph-overlay">★ primary target at center · ◆ seed nodes around it · BFS shells fan outward · scroll to zoom · drag canvas/nodes · click a node for its row</div>
      </div>
      <div id="sequence-pane" class="pane"></div>
      <div id="row-pane" class="pane"></div>
      <div id="col-pane" class="pane"></div>
      <div id="masks-pane" class="pane"></div>
      <div id="raw-pane" class="pane"></div>
      <div id="spinner"><div class="spinner-dot"></div><div class="spinner-dot"></div><div class="spinner-dot"></div></div>
    </div>
  </main>
</div>
<div id="popover"></div>

<script>
"use strict";

// ---- state ----
const state = {
  preRoot: null,
  preRoots: [],
  db: null,
  dbs: [],
  table: null,
  tables: [],
  split: null,
  target: null,
  cols: [],
  payload: null,
  tableColors: new Map(),
  hoverPin: null,
};

const $ = (id) => document.getElementById(id);
const ce = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
const fmt = (v, d=4) => {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? v.toLocaleString(undefined, {maximumFractionDigits: d}) : "NaN";
  if (typeof v === "string") return v.length > 60 ? v.slice(0, 57) + "…" : v;
  return String(v);
};
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ---- color palette by table (deterministic by name) ----
const tablePalette = d3.schemeCategory10.concat(d3.schemeSet3, d3.schemePastel1);
function colorForTable(tableName, isTask) {
  if (state.tableColors.has(tableName)) return state.tableColors.get(tableName);
  // Hash to a stable index.
  let h = 5381;
  for (let i = 0; i < tableName.length; i++) h = ((h << 5) + h) ^ tableName.charCodeAt(i);
  const idx = Math.abs(h) % tablePalette.length;
  let c = tablePalette[idx];
  if (isTask) {
    // Slightly brighten/saturate for task tables so they pop.
    c = d3.color(c).brighter(0.4).formatHex();
  }
  state.tableColors.set(tableName, c);
  return c;
}

// ---- API helpers ----
async function api(path, opts={}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({error: "HTTPError", message: res.statusText}));
    throw new Error(`${err.error}: ${err.message}`);
  }
  return res.json();
}

function showError(msg) {
  const el = $("error");
  el.textContent = msg;
  el.classList.add("show");
}
function clearError() {
  const el = $("error");
  el.classList.remove("show");
  el.textContent = "";
}

function setSpinner(on) {
  $("spinner").classList.toggle("show", !!on);
}

// ---- sidebar wiring ----
async function init() {
  // Pre-roots
  const r = await api("/api/list_pre_roots");
  state.preRoots = r.pre_roots && r.pre_roots.length ? r.pre_roots : [r.default_root];
  const sel = $("f-pre-root");
  for (const x of state.preRoots) {
    const o = ce("option"); o.value = x; o.textContent = x; sel.appendChild(o);
  }
  state.preRoot = r.default_root;
  sel.value = state.preRoot;
  sel.addEventListener("change", () => { state.preRoot = sel.value; loadDbs(); });
  await loadDbs();

  // Buttons
  $("f-build").addEventListener("click", build);
  $("f-prev").addEventListener("click", () => stepItem(-1));
  $("f-next").addEventListener("click", () => stepItem(1));
  $("f-rand").addEventListener("click", () => {
    if (!state.payload) return;
    const n = state.payload.num_items;
    if (!n) return;
    $("f-item").value = Math.floor(Math.random() * n);
    build();
  });
  $("f-export").addEventListener("click", exportJSON);

  // Theme toggle (light default; persists to localStorage).
  syncThemeButton();
  $("f-theme").addEventListener("click", toggleTheme);

  // Tab switching
  document.querySelectorAll("#tabs .tab").forEach(t => {
    t.addEventListener("click", () => {
      document.querySelectorAll("#tabs .tab").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".pane").forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      $(t.dataset.pane).classList.add("active");
      // Re-render layout-sensitive panes.
      if (t.dataset.pane === "graph-pane") renderGraph();
      if (t.dataset.pane === "masks-pane") renderMasks();
    });
  });

  // Field bindings — auto-refresh ctx-related field on Enter.
  document.querySelectorAll("#sidebar input, #sidebar select").forEach(el => {
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") build();
    });
  });

  // Auto-refresh on item change (it's cheap)
  $("f-item").addEventListener("change", build);
  $("f-ctxseed").addEventListener("change", build);
  $("f-mask").addEventListener("change", build);

  // Popover
  document.addEventListener("mousemove", (e) => {
    if (state.hoverPin) {
      const p = $("popover");
      p.style.left = (e.pageX + 14) + "px";
      p.style.top = (e.pageY + 14) + "px";
    }
  });
}

async function loadDbs() {
  const r = await api(`/api/list_dbs?root=${encodeURIComponent(state.preRoot)}`);
  state.dbs = r.dbs;
  const sel = $("f-db");
  sel.innerHTML = "";
  for (const x of state.dbs) {
    const o = ce("option"); o.value = x; o.textContent = x; sel.appendChild(o);
  }
  // Prefer rel-f1 by default (small + fast) if present.
  const preferred = state.dbs.find(x => /\brel-f1$/.test(x)) || state.dbs[0];
  if (preferred) {
    sel.value = preferred;
    state.db = preferred;
  }
  sel.addEventListener("change", () => { state.db = sel.value; loadTables(); });
  if (state.db) await loadTables();
}

async function loadTables() {
  const r = await api(`/api/list_tables?root=${encodeURIComponent(state.preRoot)}&db=${encodeURIComponent(state.db)}`);
  state.tables = r.tables;
  const sel = $("f-table");
  sel.innerHTML = "";
  // Show task tables first (those with non-Db splits), then Db-only tables.
  const sorted = [...state.tables].sort((a, b) => {
    if (a.is_task_table === b.is_task_table) return a.name.localeCompare(b.name);
    return a.is_task_table ? -1 : 1;
  });
  for (const t of sorted) {
    const o = ce("option");
    o.value = t.name;
    const splits = t.splits.join(",");
    const tag = t.is_task_table ? " (task)" : "";
    o.textContent = `${t.name} [${splits}]${tag}`;
    sel.appendChild(o);
  }
  // Prefer a small task table for instant gratification on rel-f1.
  const preferredTable = sorted.find(t => t.is_task_table) || sorted[0];
  if (preferredTable) {
    sel.value = preferredTable.name;
    state.table = preferredTable.name;
  }
  sel.addEventListener("change", () => { state.table = sel.value; loadSplits(); });
  if (state.table) await loadSplits();
}

async function loadSplits() {
  const splits = (state.tables.find(t => t.name === state.table) || {splits: ["Db"]}).splits;
  const sel = $("f-split");
  sel.innerHTML = "";
  // The python API uses lowercased train/val/test or 'db' for non-task tables.
  const candidates = splits.includes("Val") ? ["val", "train", "test"]
                   : splits.includes("Train") ? ["train", "test"]
                   : ["val", "train", "test", "db"];
  for (const x of candidates) {
    const o = ce("option"); o.value = x; o.textContent = x; sel.appendChild(o);
  }
  sel.value = candidates[0];
  state.split = sel.value;
  sel.addEventListener("change", () => { state.split = sel.value; build(); });
  await loadColumns();
}

async function loadColumns() {
  const r = await api(`/api/list_columns?root=${encodeURIComponent(state.preRoot)}&db=${encodeURIComponent(state.db)}&table=${encodeURIComponent(state.table)}`);
  state.cols = r.columns;
  const sel = $("f-target");
  sel.innerHTML = "";
  for (const c of state.cols) {
    const o = ce("option"); o.value = c.name; o.textContent = c.name; sel.appendChild(o);
  }
  // Prefer a column that contains "churn", "position", "outcome", "did_not_finish".
  const preferred = state.cols.find(c => /churn|position|outcome|did_not_finish|target|ctr|ltv|sales|score|qualifying/i.test(c.name)) || state.cols[0];
  if (preferred) {
    sel.value = preferred.name;
    state.target = preferred.name;
  }
  sel.addEventListener("change", () => { state.target = sel.value; });
}

// ---- build action ----
async function build() {
  if (!state.db || !state.table || !state.target) return;
  state.target = $("f-target").value;
  state.split = $("f-split").value;
  setSpinner(true);
  clearError();
  try {
    const drop = $("f-drop").value.split(",").map(s => s.trim()).filter(Boolean);
    const req = {
      pre_root: state.preRoot,
      db_name: state.db,
      table_name: state.table,
      target_column: state.target,
      split: state.split,
      columns_to_drop: drop,
      ctx_size: parseInt($("f-ctx").value),
      local_ctx_size: parseInt($("f-local").value),
      bfs_width: parseInt($("f-bfs").value),
      num_walks: parseInt($("f-walks").value),
      walk_length: parseInt($("f-walklen").value),
      prefer_latest: $("f-prefer-latest").checked,
      skip_text_cols: $("f-skip-text").checked,
      balance_labels: $("f-balance-labels").checked,
      bool_as_num: $("f-bool-as-num").checked,
      items_per_task: parseInt($("f-ipt").value),
      shuffle_seed: parseInt($("f-shuffle").value),
      context_seed: parseInt($("f-ctxseed").value),
      mask_prob_max: parseFloat($("f-mask").value),
      embedding_model: $("f-emb").value,
      d_text: parseInt($("f-dtext").value),
      item_idx: $("f-item").value,
    };
    const payload = await api("/api/build", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(req),
    });
    state.payload = payload;
    $("f-item-helper").textContent = `(0..${(payload.num_items || 1) - 1})`;
    renderAll();
  } catch (e) {
    showError(e.message);
  } finally {
    setSpinner(false);
  }
}

function stepItem(delta) {
  if (!state.payload) return build();
  const max = (state.payload.num_items || 1) - 1;
  const cur = parseInt($("f-item").value || "0");
  const next = Math.max(0, Math.min(max, cur + delta));
  $("f-item").value = next;
  build();
}

function exportJSON() {
  if (!state.payload) return;
  const blob = new Blob([JSON.stringify(state.payload, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ctx_${state.db.replace(/\//g,'-')}_${state.table}_${state.split}_item${state.payload.request.item_idx ?? "rand"}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- theme ----
function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}
function syncThemeButton() {
  const dark = currentTheme() === "dark";
  const btn = $("f-theme");
  btn.textContent = dark ? "☾" : "☀";
  btn.title = dark ? "switch to light theme" : "switch to dark theme";
}
function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  if (next === "dark") document.documentElement.dataset.theme = "dark";
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem("ctxviz-theme", next); } catch (e) {}
  syncThemeButton();
  // Mask canvas palette depends on theme — repaint if currently shown.
  if (state.payload && $("masks-pane").classList.contains("active")) renderMasks();
}

// ---- top stats + legend ----
function renderStats() {
  const p = state.payload;
  const s = p.stats;
  const tgt = p.target_token;
  const tgtVal = tgt ? tgt.value_str : "—";
  const tgtCol = tgt ? `${tgt.table_display}.${tgt.col_short}` : "—";
  const stats = [
    ["Item", `${(p.request.item_idx ?? "?")}/${p.num_items}`],
    ["Target", `${tgtCol} = ${tgtVal}`],
    ["Real / pad", `${s.real_tokens} / ${s.padding_tokens} (${s.padding_pct}%)`],
    ["Nodes", s.num_nodes],
    ["Tables", s.num_tables],
    ["Columns", s.num_columns],
    ["Edges", s.num_edges],
    ["Masked", `1+${s.masked_feature_count}`],
    ["Task tok", s.task_token_count],
    ["Sem", `n=${s.sem_counts.number} t=${s.sem_counts.text} d=${s.sem_counts.datetime} b=${s.sem_counts.boolean}`],
  ];
  const wrap = $("stats");
  wrap.innerHTML = "";
  for (const [l, v] of stats) {
    const e = ce("div", "stat");
    const lv = ce("div", "v"); lv.textContent = v;
    const ll = ce("div", "l"); ll.textContent = l;
    e.appendChild(lv); e.appendChild(ll);
    wrap.appendChild(e);
  }
}
function renderLegend() {
  const ts = state.payload.table_summary.slice().sort((a, b) => b.cell_count - a.cell_count);
  const wrap = $("legend");
  wrap.innerHTML = "";
  for (const t of ts) {
    const sw = ce("div", "swatch" + (t.is_task ? " task" : ""));
    const dot = ce("span", "dot");
    dot.style.background = colorForTable(t.table_display, t.is_task);
    if (t.is_task) dot.style.borderRadius = "2px";
    sw.appendChild(dot);
    const lab = ce("span");
    lab.textContent = `${t.table_display} (${t.node_count}n / ${t.cell_count}c)`;
    sw.appendChild(lab);
    wrap.appendChild(sw);
  }
  wrap.appendChild(buildLegendKey("Target", "var(--target)", "•"));
  wrap.appendChild(buildLegendKey("Task token", "var(--task)", "▢"));
}
function buildLegendKey(label, color, sym) {
  const sw = ce("div", "swatch");
  const dot = ce("span", "dot");
  dot.style.background = color;
  sw.appendChild(dot);
  const lab = ce("span");
  lab.textContent = label;
  sw.appendChild(lab);
  return sw;
}

// ---- popover ----
function showPopover(token, evt) {
  if (!token) return hidePopover();
  const p = $("popover");
  let role = "—";
  if (token.is_primary_target) role = "★ PRIMARY TARGET";
  else if (token.is_masked_feature) role = "masked feature";
  else if (token.is_task_node) role = "task-node feature";
  else if (token.is_padding) role = "padding";
  else role = "context cell";
  const rows = [
    ["i (seq pos)", token.i],
    ["role", role],
    ["node_idx", token.node_idx],
    ["table", token.table_display + (token.is_task_table ? " (task)" : "")],
    ["table_name_idx", token.table_name_idx],
    ["col", token.col_short],
    ["col_full", token.col_name],
    ["col_name_idx", token.col_name_idx],
    ["sem_type", token.sem_type],
    ["value", token.value_str],
    ["value (raw)", token.sem_type === "number" ? token.number_value
                  : token.sem_type === "datetime" ? token.datetime_value
                  : token.sem_type === "boolean" ? token.boolean_value
                  : `text_idx=${token.class_value_idx}`],
    ["is_target", token.is_target],
    ["is_task_node", token.is_task_node],
    ["is_padding", token.is_padding],
    ["timestamp", token.timestamp ?? "—"],
    ["f2p_nbrs", token.f2p_nbr_idxs && token.f2p_nbr_idxs.length ? token.f2p_nbr_idxs.join(", ") : "—"],
  ];
  p.innerHTML = rows.map(([k, v]) => `<div class="row"><div class="k">${k}</div><div class="v">${escapeHTML(String(v))}</div></div>`).join("");
  p.style.display = "block";
  state.hoverPin = true;
  if (evt) {
    p.style.left = (evt.pageX + 14) + "px";
    p.style.top = (evt.pageY + 14) + "px";
  }
}
function hidePopover() {
  $("popover").style.display = "none";
  state.hoverPin = null;
}
function escapeHTML(s) {
  return s.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

// ---- sequence pane ----
function renderSequence() {
  const pane = $("sequence-pane");
  pane.innerHTML = "";

  // Sequence tape: each position is a thin vertical bar across the width.
  const tokens = state.payload.tokens;
  const S = tokens.length;
  const tapeHelp = ce("div", "seq-tape-helper");
  tapeHelp.textContent = `Sequence tape (${S} positions, colored by table; hover/click to jump)`;
  pane.appendChild(tapeHelp);

  const tape = ce("div", "seq-tape");
  tape.style.position = "relative";
  pane.appendChild(tape);
  const tapeWidth = pane.clientWidth - 24;
  // Use SVG for the tape so we can size cells precisely.
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", tapeWidth);
  svg.setAttribute("height", 36);
  svg.setAttribute("viewBox", `0 0 ${S} 36`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.style.width = "100%";
  svg.style.height = "100%";
  for (let i = 0; i < S; i++) {
    const t = tokens[i];
    const r = document.createElementNS(ns, "rect");
    r.setAttribute("class", "tape-cell");
    r.setAttribute("x", String(i));
    r.setAttribute("y", "4");
    r.setAttribute("width", "1");
    r.setAttribute("height", "28");
    if (t.is_padding) {
      r.style.fill = "var(--pad)";
      r.setAttribute("opacity", "0.5");
    } else {
      r.setAttribute("fill", colorForTable(t.table_display, t.is_task_table));
    }
    if (t.is_primary_target) {
      r.style.stroke = "var(--target)";
      r.setAttribute("stroke-width", "0.5");
      r.setAttribute("y", "0");
      r.setAttribute("height", "36");
    }
    if (t.is_masked_feature) {
      const m = document.createElementNS(ns, "rect");
      m.setAttribute("x", String(i));
      m.setAttribute("y", "32");
      m.setAttribute("width", "1");
      m.setAttribute("height", "4");
      m.style.fill = "var(--task)";
      svg.appendChild(m);
    }
    r.addEventListener("mouseenter", (e) => showPopover(t, e));
    r.addEventListener("mouseleave", hidePopover);
    r.addEventListener("click", () => {
      const cells = pane.querySelectorAll(".cell-grid .cell");
      if (cells[i]) {
        cells[i].scrollIntoView({behavior: "smooth", block: "center"});
        cells[i].animate(
          [{boxShadow: "inset 0 0 0 2px var(--accent)"}, {boxShadow: "inset 0 0 0 0 transparent"}],
          {duration: 800}
        );
      }
    });
    svg.appendChild(r);
  }
  tape.appendChild(svg);

  const grid = ce("div", "cell-grid");
  for (const t of tokens) {
    const cell = ce("div", "cell");
    if (t.is_padding) cell.classList.add("padding");
    if (t.is_primary_target) cell.classList.add("target");
    if (t.is_masked_feature) cell.classList.add("masked");
    if (t.is_task_node && !t.is_padding) cell.classList.add("task");
    if (!t.is_padding) cell.style.borderLeftColor = colorForTable(t.table_display, t.is_task_table);
    if (t.is_padding) {
      cell.innerHTML = `<div class="head"><span>${t.i}</span><span>pad</span></div>`;
    } else {
      const tgtBadge = t.is_primary_target
        ? '<span class="badge target">★</span>'
        : (t.is_masked_feature ? '<span class="badge task">msk</span>' : '');
      const taskBadge = t.is_task_node ? '<span class="badge task">tsk</span>' : '';
      cell.innerHTML = `
        <div class="head">
          <span>${t.i}</span>
          <span>${tgtBadge}${taskBadge}</span>
        </div>
        <div class="col" title="${escapeHTML(t.col_name)}">${escapeHTML(t.col_short)}</div>
        <div class="val" title="${escapeHTML(t.value_str || '')}">${escapeHTML(t.value_str || '')}</div>
        <div class="sem">${t.sem_type ?? ''} · n${t.node_idx}</div>
      `;
    }
    cell.addEventListener("mouseenter", (e) => showPopover(t, e));
    cell.addEventListener("mouseleave", hidePopover);
    cell.addEventListener("click", () => {
      // Switch to row view & scroll to this node.
      switchTab("row-pane");
      setTimeout(() => {
        const el = document.querySelector(`[data-node-idx='${t.node_idx}']`);
        if (el) el.scrollIntoView({behavior: "smooth", block: "center"});
      }, 50);
    });
    grid.appendChild(cell);
  }
  pane.appendChild(grid);
}

// ---- by-row pane ----
function renderRows() {
  const pane = $("row-pane");
  pane.innerHTML = "";
  const tokens = state.payload.tokens;
  // Groups already split server-side.
  const groups = state.payload.nodes_meta.slice();
  // Sort: target first, then task, then by-table.
  groups.sort((a, b) => {
    if (a.is_target_node !== b.is_target_node) return a.is_target_node ? -1 : 1;
    if (a.is_task_node !== b.is_task_node) return a.is_task_node ? -1 : 1;
    if (a.table_display !== b.table_display) return a.table_display.localeCompare(b.table_display);
    return a.node_idx - b.node_idx;
  });
  for (const g of groups) {
    const block = ce("div", "node-row");
    block.dataset.nodeIdx = g.node_idx;
    if (g.is_target_node) block.classList.add("target");
    if (g.is_task_node) block.classList.add("task");
    const head = ce("div", "row-head");
    const dot = ce("span", "dot");
    dot.style.cssText = `width:10px;height:10px;border-radius:50%;background:${colorForTable(g.table_display, g.is_task_table)};display:inline-block;`;
    head.appendChild(dot);
    const tn = ce("span", "table-name");
    tn.textContent = g.table_display;
    head.appendChild(tn);
    const meta = ce("span", "meta");
    meta.textContent = `node ${g.node_idx} · ${g.cell_idxs.length} cells${g.timestamp ? ` · ts=${g.timestamp}` : ""}${g.is_target_node ? " · TARGET" : ""}${g.is_task_node ? " · task" : ""}`;
    head.appendChild(meta);
    block.appendChild(head);

    const list = ce("div", "cell-list");
    for (const i of g.cell_idxs) {
      const t = tokens[i];
      const kv = ce("div", "kv");
      if (t.is_primary_target) kv.classList.add("target");
      if (t.is_masked_feature) kv.style.background = "rgba(179, 146, 240, 0.10)";
      const tgt = t.is_primary_target ? ' <span class="badge target">★</span>'
                : t.is_masked_feature ? ' <span class="badge task">msk</span>' : '';
      kv.innerHTML = `<div class="k" title="${escapeHTML(t.col_name)}">${escapeHTML(t.col_short)} <span class="sem">${t.sem_type}</span>${tgt}</div><div class="v">${escapeHTML(t.value_str || '—')}</div>`;
      kv.addEventListener("mouseenter", (e) => showPopover(t, e));
      kv.addEventListener("mouseleave", hidePopover);
      list.appendChild(kv);
    }
    block.appendChild(list);
    pane.appendChild(block);
  }
}

// ---- by-column pane ----
function renderColumns() {
  const pane = $("col-pane");
  pane.innerHTML = "";
  const tokens = state.payload.tokens;
  const cols = state.payload.cols_meta.slice();
  cols.sort((a, b) => {
    if (a.table_display !== b.table_display) return a.table_display.localeCompare(b.table_display);
    return a.col_short.localeCompare(b.col_short);
  });
  for (const c of cols) {
    const block = ce("div", "col-block");
    const head = ce("div", "col-head");
    const dot = ce("span");
    dot.style.cssText = `width:10px;height:10px;border-radius:50%;background:${colorForTable(c.table_display, false)};display:inline-block;margin-right:4px;`;
    head.appendChild(dot);
    const name = ce("span", "name");
    name.textContent = c.col_short;
    head.appendChild(name);
    const tbl = ce("span", "table");
    tbl.textContent = `· ${c.table_display} · ${c.cell_idxs.length}× · ${c.sem_types.join("/")}`;
    head.appendChild(tbl);
    block.appendChild(head);

    // Pills (or histogram if numeric).
    if (c.sem_types.includes("number") || c.sem_types.includes("datetime") || c.sem_types.includes("boolean")) {
      const cv = ce("div");
      cv.className = "col-numdist";
      block.appendChild(cv);
      drawNumDist(cv, c.cell_idxs.map(i => tokens[i]), tokens);
    }
    const valWrap = ce("div", "col-vals");
    for (const i of c.cell_idxs.slice(0, 400)) {
      const t = tokens[i];
      const pill = ce("span", "pill");
      if (t.is_primary_target) pill.classList.add("target");
      if (t.is_masked_feature) pill.style.borderColor = "var(--task)";
      pill.textContent = (t.is_primary_target ? "★ " : (t.is_masked_feature ? "▣ " : "")) + (t.value_str || "—");
      pill.title = `pos ${i} · node ${t.node_idx}${t.timestamp ? ' · ts ' + t.timestamp : ''}`;
      pill.addEventListener("mouseenter", (e) => showPopover(t, e));
      pill.addEventListener("mouseleave", hidePopover);
      valWrap.appendChild(pill);
    }
    if (c.cell_idxs.length > 400) {
      const more = ce("span", "pill");
      more.textContent = `+${c.cell_idxs.length - 400} more`;
      more.style.color = "var(--text-faint)";
      valWrap.appendChild(more);
    }
    block.appendChild(valWrap);
    pane.appendChild(block);
  }
}

function drawNumDist(container, tokens, _allTokens) {
  // Mini sparkline-style strip: each cell is one tick.
  const w = container.clientWidth || 600;
  const h = 28;
  const svg = d3.select(container).html("").append("svg").attr("width", w).attr("height", h);
  const vals = tokens.map(t => {
    if (t.sem_type === "number") return t.number_value;
    if (t.sem_type === "datetime") return t.datetime_value;
    if (t.sem_type === "boolean") return t.boolean_value;
    return null;
  }).filter(v => v !== null && Number.isFinite(v));
  if (!vals.length) return;
  const ext = d3.extent(vals);
  if (ext[0] === ext[1]) ext[1] = ext[0] + 1;
  const x = d3.scaleLinear().domain(ext).range([2, w - 2]);
  svg.append("line").attr("x1", 0).attr("x2", w).attr("y1", h - 2).attr("y2", h - 2)
     .style("stroke", "var(--line)").attr("stroke-width", 1);
  for (const t of tokens) {
    let v;
    if (t.sem_type === "number") v = t.number_value;
    else if (t.sem_type === "datetime") v = t.datetime_value;
    else if (t.sem_type === "boolean") v = t.boolean_value;
    else continue;
    if (!Number.isFinite(v)) continue;
    const cx = x(v);
    svg.append("line")
       .attr("x1", cx).attr("x2", cx).attr("y1", 6).attr("y2", h - 4)
       .style("stroke", t.is_primary_target ? "var(--target)" : (t.is_masked_feature ? "var(--task)" : "var(--accent)"))
       .attr("stroke-width", t.is_primary_target ? 2 : 1)
       .attr("opacity", t.is_primary_target ? 1 : (t.is_masked_feature ? 0.85 : 0.55));
  }
  svg.append("text").text(fmt(ext[0])).attr("x", 0).attr("y", 8)
     .style("fill", "var(--text-faint)").style("font-family", "var(--mono)").attr("font-size", 9);
  svg.append("text").text(fmt(ext[1])).attr("x", w).attr("y", 8)
     .attr("text-anchor", "end").style("fill", "var(--text-faint)").style("font-family", "var(--mono)").attr("font-size", 9);
}

// ---- raw pane ----
function renderRaw() {
  const pane = $("raw-pane");
  pane.innerHTML = "";
  const tbl = ce("table", "raw-table");
  const cols = ["i", "node_idx", "table", "col", "sem", "value", "tgt", "task", "ts", "f2p"];
  const thead = ce("thead");
  const trh = ce("tr");
  for (const c of cols) {
    const th = ce("th"); th.textContent = c; trh.appendChild(th);
  }
  thead.appendChild(trh);
  tbl.appendChild(thead);
  const tbody = ce("tbody");
  for (const t of state.payload.tokens) {
    const tr = ce("tr");
    if (t.is_primary_target) tr.classList.add("target");
    if (t.is_padding) tr.classList.add("padding");
    if (t.is_masked_feature) tr.style.background = "rgba(179,146,240,0.07)";
    const tgtMark = t.is_primary_target ? "★" : (t.is_masked_feature ? "msk" : "");
    const cells = [
      String(t.i),
      String(t.node_idx),
      t.is_padding ? "—" : `${t.table_display}`,
      t.is_padding ? "—" : t.col_short,
      t.is_padding ? "—" : (t.sem_type || "—"),
      t.is_padding ? "—" : (t.value_str || "—"),
      t.is_padding ? "—" : tgtMark,
      t.is_padding ? "—" : (t.is_task_node ? "✓" : ""),
      t.timestamp == null ? "—" : String(t.timestamp),
      t.f2p_nbr_idxs.length ? t.f2p_nbr_idxs.join(",") : "—",
    ];
    for (let i = 0; i < cells.length; i++) {
      const td = ce("td");
      if (i === 0 || i === 1 || i === 8) td.classList.add("num");
      if (i === 6 || i === 7) td.classList.add("center");
      td.textContent = cells[i];
      tr.appendChild(td);
    }
    tr.addEventListener("mouseenter", (e) => showPopover(t, e));
    tr.addEventListener("mouseleave", hidePopover);
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
  pane.appendChild(tbl);
}

// ---- attention masks pane ----
function renderMasks() {
  const pane = $("masks-pane");
  pane.innerHTML = "";
  if (!state.payload) return;
  const tokens = state.payload.tokens;
  const S = tokens.length;
  // Compute the three boolean masks the model uses, modulo padding.
  // (we mirror the python code in src/rt/model.py).
  const isPad = tokens.map(t => t.is_padding);
  const node = tokens.map(t => t.node_idx);
  const f2p = tokens.map(t => t.f2p_nbr_idxs);
  const col = tokens.map(t => t.col_name_idx);
  const tbl = tokens.map(t => t.table_name_idx);

  // Sort cells by col_name_idx (matching forward()), padding at the end.
  const order = d3.range(S).sort((a, b) => {
    const ka = isPad[a] ? 1e18 : col[a];
    const kb = isPad[b] ? 1e18 : col[b];
    return ka - kb;
  });
  const featM = new Uint8Array(S * S);
  const nbrM = new Uint8Array(S * S);
  const colM = new Uint8Array(S * S);
  for (let qi = 0; qi < S; qi++) {
    const q = order[qi];
    if (isPad[q]) continue;
    for (let ki = 0; ki < S; ki++) {
      const k = order[ki];
      if (isPad[k]) continue;
      const sameNode = node[q] === node[k];
      const kvInQF2P = f2p[q].includes(node[k]);
      const qInKF2P = f2p[k].includes(node[q]);
      const sameCT = (col[q] === col[k]) && (tbl[q] === tbl[k]);
      if (sameNode || kvInQF2P) featM[qi * S + ki] = 1;
      if (qInKF2P) nbrM[qi * S + ki] = 1;
      if (sameCT) colM[qi * S + ki] = 1;
    }
  }

  // Compose 3 mask blocks.
  pane.innerHTML = `<div class="masks-grid">
    <div class="mask-block"><h4>FEAT mask</h4><p>same-node | kv ∈ q.f2p — captures “my row + the rows I point to”</p><canvas class="mask-canvas" id="mask-feat"></canvas></div>
    <div class="mask-block"><h4>NBR mask</h4><p>q ∈ kv.f2p — “rows that point to me” (reverse FK)</p><canvas class="mask-canvas" id="mask-nbr"></canvas></div>
    <div class="mask-block"><h4>COL mask</h4><p>same (table, column) — “other cells from this column across the context”</p><canvas class="mask-canvas" id="mask-col"></canvas></div>
  </div>
  <div class="helper" style="margin-top:8px;">
    Cells reordered by col_name_idx (matches model.forward()). Pixel grid is q (row) × kv (col); white = attend, black = mask. Padding excluded.
  </div>`;
  drawMask("mask-feat", featM, S);
  drawMask("mask-nbr", nbrM, S);
  drawMask("mask-col", colM, S);
}

function drawMask(canvasId, m, S) {
  const cv = document.getElementById(canvasId);
  cv.width = S;
  cv.height = S;
  const ctx = cv.getContext("2d");
  const id = ctx.createImageData(S, S);
  // Invert palette in light mode so "attend" pixels are dark on a light
  // page bg (visible) and "mask" pixels are light (visually quiet).
  const dark = document.documentElement.dataset.theme === "dark";
  const onColor  = dark ? 230 : 25;   // attend
  const offColor = dark ? 25  : 235;  // mask
  for (let i = 0; i < S * S; i++) {
    const v = m[i] ? onColor : offColor;
    id.data[i*4 + 0] = v;
    id.data[i*4 + 1] = v;
    id.data[i*4 + 2] = v;
    id.data[i*4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

// ---- graph pane ----
let graphSim = null;
let graphZoom = null;
function renderGraph() {
  const svg = d3.select("#graph-svg");
  svg.selectAll("*").remove();
  if (!state.payload) return;
  const w = svg.node().clientWidth || 800;
  const h = svg.node().clientHeight || 600;
  svg.attr("viewBox", `0 0 ${w} ${h}`);
  const root = svg.append("g").attr("class", "root");

  const nodes_meta = state.payload.nodes_meta;
  const seeds_meta = state.payload.seeds_meta || [];
  const edges = state.payload.graph.edges;
  const primaryTargetNodeIdx = state.payload.target_token ? state.payload.target_token.node_idx : null;

  // Compute deterministic shell positions:
  //   target seed → page center
  //   other seeds → on a ring at radius R_seeds around target
  //   each non-seed node → on an arc around its seed at radius depth*R_shell
  // For non-target seeds the arc fans outward (away from target) so shells
  // grow into empty space; for the target seed they go around 360°.
  const cx = w / 2, cy = h / 2;
  const R = Math.min(w, h);
  const rSeeds = R * 0.30;            // seed-ring radius
  const rShellStep = Math.max(28, R * 0.07); // BFS shell spacing
  const targetSeed = seeds_meta.find(s => s.is_target_seed);
  const otherSeeds = seeds_meta.filter(s => !s.is_target_seed);

  const seedAngle = new Map();   // seed_node_idx → outward angle θ
  const seedPos = new Map();     // seed_node_idx → {x, y}
  if (targetSeed) {
    seedPos.set(targetSeed.seed_node_idx, {x: cx, y: cy});
    seedAngle.set(targetSeed.seed_node_idx, null); // full 360 fan
  }
  // Distribute other seeds evenly around the ring; start near 12 o'clock
  // and go clockwise so the layout is reproducible.
  otherSeeds.forEach((s, i) => {
    const θ = (-Math.PI / 2) + (2 * Math.PI) * (i / Math.max(1, otherSeeds.length));
    seedPos.set(s.seed_node_idx, {x: cx + rSeeds * Math.cos(θ), y: cy + rSeeds * Math.sin(θ)});
    seedAngle.set(s.seed_node_idx, θ);
  });

  // Group non-seed nodes by (seed, depth) so we can spread same-shell nodes
  // around an arc around their seed.
  // shellGroups: Map<seed, Map<depth, [node_meta...]>>
  const shellGroups = new Map();
  for (const n of nodes_meta) {
    const s = n.seed_node_idx;
    if (s == null) continue;
    if (!shellGroups.has(s)) shellGroups.set(s, new Map());
    const byDepth = shellGroups.get(s);
    if (!byDepth.has(n.bfs_depth)) byDepth.set(n.bfs_depth, []);
    byDepth.get(n.bfs_depth).push(n);
  }

  // Compute the final position per node.
  const pos = new Map(); // node_idx → {x, y}
  for (const [seedIdx, byDepth] of shellGroups) {
    const center = seedPos.get(seedIdx) || {x: cx, y: cy};
    const θ_outward = seedAngle.get(seedIdx);
    const depths = [...byDepth.keys()].sort((a, b) => a - b);
    for (const d of depths) {
      const members = byDepth.get(d);
      if (d === 0) {
        // Just the seed — place at center
        pos.set(members[0].node_idx, {x: center.x, y: center.y});
        continue;
      }
      const r = d * rShellStep;
      const N = members.length;
      // For target seed, fan around full 2π (seeded by node_idx hash so it's
      // stable). For non-target seeds, fan over a narrower outward arc that
      // shrinks as the seed-ring fills.
      let θ0, span;
      if (θ_outward === null) {
        θ0 = -Math.PI / 2;             // start at top
        span = 2 * Math.PI;            // full circle
      } else {
        θ0 = θ_outward;
        // narrower arc when there are many sibling seeds
        const widthFactor = otherSeeds.length <= 1
          ? 1.6
          : Math.min(1.4, (2 * Math.PI / otherSeeds.length) * 0.85);
        span = Math.PI * widthFactor;
      }
      members.sort((a, b) => a.node_idx - b.node_idx); // stable ordering
      for (let i = 0; i < N; i++) {
        const t = N === 1 ? 0.5 : i / (N - 1);
        const θ = (θ_outward === null)
          ? (θ0 + span * t)
          : (θ0 - span / 2 + span * t);
        pos.set(members[i].node_idx, {
          x: center.x + r * Math.cos(θ),
          y: center.y + r * Math.sin(θ),
        });
      }
    }
  }

  // Build node + link payloads with the precomputed positions baked in.
  const nodes = nodes_meta.map(n => {
    const p = pos.get(n.node_idx) || {x: cx, y: cy};
    return {
      id: n.node_idx,
      table: n.table_display,
      is_task: n.is_task_node || n.is_task_table,
      is_target: n.is_target_node,
      is_primary: n.node_idx === primaryTargetNodeIdx,
      is_seed: n.is_seed,
      seed: n.seed_node_idx,
      depth: n.bfs_depth,
      cells: n.cell_idxs.length,
      full_meta: n,
      x: p.x, y: p.y,
      // pin position; we use a tiny simulation only to allow drag interaction.
      fx: p.x, fy: p.y,
    };
  });
  const nodeById = new Map(nodes.map(n => [n.id, n]));
  const links = edges.map(e => ({source: nodeById.get(e.source), target: nodeById.get(e.target)}))
                     .filter(l => l.source && l.target);

  // Optional: faint shell-rings to make the layout legible.
  const ringG = root.append("g").attr("class", "shell-rings");
  for (const [seedIdx, byDepth] of shellGroups) {
    const center = seedPos.get(seedIdx) || {x: cx, y: cy};
    const maxDepth = Math.max(...byDepth.keys());
    for (let d = 1; d <= maxDepth; d++) {
      ringG.append("circle")
        .attr("cx", center.x).attr("cy", center.y)
        .attr("r", d * rShellStep)
        .attr("fill", "none")
        .style("stroke", "var(--line)")
        .attr("stroke-dasharray", "2,3")
        .attr("stroke-width", 0.8)
        .attr("opacity", 0.55);
    }
  }

  // Edges
  const link = root.append("g").attr("class", "links")
    .selectAll(".link")
    .data(links).enter().append("line").attr("class", "link")
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);

  const nodeG = root.append("g").attr("class", "nodes")
    .selectAll(".node")
    .data(nodes).enter().append("g")
    .attr("class", d => "node" + (d.is_primary ? " target" : ""))
    .attr("transform", d => `translate(${d.x},${d.y})`);

  // Circle for db nodes, rounded square for task nodes; seeds get a slight
  // size bump so they read as the "centers" of their shells.
  nodeG.each(function(d) {
    const sel = d3.select(this);
    let r = Math.max(5, Math.min(18, 4 + Math.sqrt(d.cells)));
    if (d.is_seed) r += 2;
    if (d.is_task) {
      sel.append("rect")
         .attr("x", -r).attr("y", -r).attr("width", 2*r).attr("height", 2*r).attr("rx", 3)
         .attr("fill", colorForTable(d.table, true));
    } else {
      sel.append("circle")
         .attr("r", r).attr("fill", colorForTable(d.table, false));
    }
    if (d.is_seed && !d.is_primary) {
      sel.select("rect, circle").style("stroke", "var(--accent-2)").attr("stroke-width", 2);
    }
    if (d.is_primary) {
      sel.select("rect, circle").style("stroke", "var(--target)").attr("stroke-width", 3);
    }
    if (d.is_seed || d.is_primary) {
      sel.append("text")
        .attr("dy", -r - 4)
        .attr("text-anchor", "middle")
        .text(d.is_primary ? `★ ${d.table}` : `◆ ${d.table}`);
    } else {
      // Compact label only when a seed/primary; otherwise leave nodes unlabeled
      // to keep the BFS shells tidy. Tooltip + click still expose details.
    }
  });

  // Drag updates the pinned position so simulation does no work.
  nodeG.call(d3.drag()
    .on("drag", (e, d) => {
      d.fx = e.x; d.fy = e.y; d.x = e.x; d.y = e.y;
      d3.select(this).attr("transform", `translate(${d.x},${d.y})`);
      // Translate the dragged node directly and update incident edges.
      const grp = d3.select(e.sourceEvent.target.closest(".node"));
      grp.attr("transform", `translate(${d.x},${d.y})`);
      link.filter(l => l.source.id === d.id || l.target.id === d.id)
        .attr("x1", l => l.source.x).attr("y1", l => l.source.y)
        .attr("x2", l => l.target.x).attr("y2", l => l.target.y);
    })
  );

  nodeG.on("mouseenter", (evt, d) => {
    const t = state.payload.tokens[d.full_meta.cell_idxs[0]];
    const role = d.is_primary ? "primary target"
              : d.is_seed     ? `seed (depth=0, owns ${seeds_meta.find(s => s.seed_node_idx === d.seed)?.node_count || "?"} nodes)`
              : `BFS depth=${d.depth} (seed=${d.seed})`;
    showPopover({...t, value_str: `${d.full_meta.cell_idxs.length} cells · ${role}`}, evt);
  }).on("mouseleave", hidePopover)
    .on("dblclick", (evt, d) => {
      const tx = -d.x + w/2, ty = -d.y + h/2;
      svg.transition().duration(450).call(graphZoom.transform, d3.zoomIdentity.translate(tx, ty).scale(1.2));
    })
    .on("click", (evt, d) => {
      switchTab("row-pane");
      setTimeout(() => {
        const el = document.querySelector(`[data-node-idx='${d.id}']`);
        if (el) el.scrollIntoView({behavior: "smooth", block: "center"});
      }, 50);
    });

  graphZoom = d3.zoom().scaleExtent([0.1, 8]).on("zoom", (e) => root.attr("transform", e.transform));
  svg.call(graphZoom);
}

// ---- entry ----
function switchTab(paneId) {
  document.querySelectorAll("#tabs .tab").forEach(x => x.classList.toggle("active", x.dataset.pane === paneId));
  document.querySelectorAll(".pane").forEach(x => x.classList.toggle("active", x.id === paneId));
  if (paneId === "graph-pane") renderGraph();
  if (paneId === "masks-pane") renderMasks();
}

function renderAll() {
  if (!state.payload) return;
  // Re-color tables (preserve previous mappings; new ones get assigned as needed).
  for (const t of state.payload.table_summary) colorForTable(t.table_display, t.is_task);
  renderStats();
  renderLegend();
  renderSequence();
  renderRows();
  renderColumns();
  renderRaw();
  renderGraph();
  renderMasks();
}

window.addEventListener("resize", () => {
  if (state.payload) {
    if ($("graph-pane").classList.contains("active")) renderGraph();
    if ($("masks-pane").classList.contains("active")) renderMasks();
  }
});

init().catch(e => showError(e.message));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
