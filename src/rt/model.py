from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention, dynamic=False)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


class MaskedAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        legacy_attn=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads
        # legacy_attn reproduces the pre-RT-J attention (RT / rt-plurel
        # checkpoints): plain 1/sqrt(d) softmax scaling, no learned per-head
        # scale, no log(kv_size) length scaling, no output gate.
        self.legacy_attn = legacy_attn

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.wo.weight)

        if not legacy_attn:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
            self.wg = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.wg.weight)

    def forward(self, x, block_mask, kv_sizes):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if not self.legacy_attn:
            # clamp_min(1) so kv_size=0 (queries with all-masked keys) gives
            # log(1)=0 instead of log(1e-6)=-13.8. flex_attention already
            # zeros the output for fully-masked queries; this just removes
            # the wrong-sign numerical hazard on q for those rows. Has no
            # effect when kv_size >= 1.
            q = (
                q
                * self.scale
                * torch.log(rearrange(kv_sizes.clamp_min(1.0), "b s 1 -> b 1 s 1"))
            )

        v = v.to(q.dtype)

        attn_out = flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=(self.head_dim**-0.5 if self.legacy_attn else 1.0 / self.head_dim),
        )
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        if not self.legacy_attn:
            gate = 2 * torch.sigmoid(self.wg(x))
            attn_out = gate * attn_out

        output = self.wo(attn_out)
        return output


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        nn.init.zeros_(self.w2.weight)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        legacy_attn=False,
    ):
        super().__init__()
        self.attn_types = ["col", "feat", "nbr"]
        self.norms = nn.ModuleDict(
            {
                attn_type: RMSNorm(d_model, eps=1e-6)
                for attn_type in self.attn_types + ["ffn"]
            }
        )

        self.attns = nn.ModuleDict()

        for attn_type in self.attn_types:
            self.attns[attn_type] = MaskedAttention(d_model, num_heads, legacy_attn)

        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, block_masks, kv_sizes):
        for attn in self.attn_types:
            x = x + self.attns[attn](
                self.norms[attn](x),
                block_mask=block_masks[attn],
                kv_sizes=kv_sizes[attn],
            )

        x = x + self.ffn(self.norms["ffn"](x))
        return x


def _make_block_mask(mask, batch_size, q_seq_len, kv_seq_len, device):
    def _mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod=_mod,
        B=batch_size,
        H=None,
        Q_LEN=q_seq_len,
        KV_LEN=kv_seq_len,
        device=device,
        _compile=True,
    )


def _kv_sizes(node_idxs, f2p_nbr_idxs, col_name_idxs, table_name_idxs, is_padding):
    # Exact per-query key counts for the three attention types, computed without
    # ever materializing a (B, S, S) tensor. Peak memory is O(B*S*F).
    #
    # Strategy: per batch row, sort the relevant key tensors and look up counts
    # via searchsorted (count of value v in sorted list = upper - lower bound).
    B, S = node_idxs.shape
    Fnbr = f2p_nbr_idxs.shape[-1]
    INT64_MAX = torch.iinfo(torch.int64).max
    SENTINEL = INT64_MAX  # padding/invalid keys are shifted past any real value

    not_pad = (~is_padding).long()  # (B, S)

    # Sorted node ids per batch row, with padding tokens shifted to the sentinel
    # so they never match a real query.
    node_sortable = node_idxs.to(torch.int64).masked_fill(is_padding, SENTINEL)
    sorted_nodes, _ = node_sortable.sort(dim=-1)  # (B, S)

    # ---- col: kv tokens with the same (col_name_idxs, table_name_idxs) and not padding.
    # Pack (col, table) into one 64-bit key; both indices are small non-negative vocab ids.
    combined = col_name_idxs.to(torch.int64) * (1 << 32) + table_name_idxs.to(
        torch.int64
    )
    combined_q = combined.masked_fill(is_padding, SENTINEL)
    sorted_combined, _ = combined_q.sort(dim=-1)
    lo_c = torch.searchsorted(sorted_combined, combined_q, side="left")
    hi_c = torch.searchsorted(sorted_combined, combined_q, side="right")
    col_count = (hi_c - lo_c) * not_pad

    # ---- feat: kv tokens whose node id is in {node[q]} ∪ f2p_nbr_idxs[q,:], not padding.
    # Build the candidate set per query, dedupe via sort+is_first, then sum counts of each
    # unique non-(-1) candidate looked up in sorted_nodes.
    candidates = torch.cat([node_idxs.unsqueeze(-1), f2p_nbr_idxs], dim=-1).to(
        torch.int64
    )  # (B, S, F+1)
    sorted_cand, _ = candidates.sort(dim=-1)
    is_first = torch.cat(
        [
            torch.ones_like(sorted_cand[..., :1], dtype=torch.bool),
            sorted_cand[..., 1:] != sorted_cand[..., :-1],
        ],
        dim=-1,
    )
    keep = is_first & (sorted_cand >= 0)  # exclude -1 (no neighbor) sentinels
    sorted_cand_flat = sorted_cand.reshape(B, S * (Fnbr + 1))
    lo_f = torch.searchsorted(sorted_nodes, sorted_cand_flat, side="left").view(
        B, S, Fnbr + 1
    )
    hi_f = torch.searchsorted(sorted_nodes, sorted_cand_flat, side="right").view(
        B, S, Fnbr + 1
    )
    feat_count = ((hi_f - lo_f) * keep.long()).sum(-1) * not_pad

    # ---- nbr: kv tokens whose f2p list contains node[q], not padding.
    # Build a per-batch flat list of unique non-(-1) f2p values from non-padding kv,
    # sort it, then look up node[q] counts.
    sorted_f, _ = f2p_nbr_idxs.to(torch.int64).sort(dim=-1)  # (B, S, F)
    is_first_f = torch.cat(
        [
            torch.ones_like(sorted_f[..., :1], dtype=torch.bool),
            sorted_f[..., 1:] != sorted_f[..., :-1],
        ],
        dim=-1,
    )
    keep_f = is_first_f & (sorted_f >= 0) & (~is_padding).unsqueeze(-1)
    masked_f = torch.where(keep_f, sorted_f, torch.full_like(sorted_f, SENTINEL))
    flat = masked_f.view(B, S * Fnbr)
    flat_sorted, _ = flat.sort(dim=-1)
    node_q = node_idxs.to(torch.int64)
    lo_n = torch.searchsorted(flat_sorted, node_q, side="left")
    hi_n = torch.searchsorted(flat_sorted, node_q, side="right")
    nbr_count = (hi_n - lo_n) * not_pad

    return {
        "feat": feat_count.unsqueeze(-1).bfloat16(),
        "nbr": nbr_count.unsqueeze(-1).bfloat16(),
        "col": col_count.unsqueeze(-1).bfloat16(),
    }


SEM_TYPE_NAMES = ["number", "text", "datetime", "boolean"]


class RelationalTransformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        d_text,
        num_heads,
        d_ff,
        compile,
        materialize_attn_masks,
        legacy_attn=False,
    ):
        super().__init__()
        self.materialize_attn_masks = materialize_attn_masks
        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, bias=True),
                "text": nn.Linear(d_text, d_model, bias=True),
                "datetime": nn.Linear(1, d_model, bias=True),
                "col_name": nn.Linear(d_text, d_model, bias=True),
                "boolean": nn.Linear(1, d_model, bias=True),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, bias=True),
                "text": nn.Linear(d_model, d_text, bias=True),
                "datetime": nn.Linear(d_model, 1, bias=True),
                "boolean": nn.Linear(d_model, 1, bias=True),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {
                "number": RMSNorm(d_model, eps=1e-6),
                "text": RMSNorm(d_model, eps=1e-6),
                "datetime": RMSNorm(d_model, eps=1e-6),
                "col_name": RMSNorm(d_model, eps=1e-6),
                "boolean": RMSNorm(d_model, eps=1e-6),
            }
        )
        self.mask_embs = nn.ParameterDict(
            {
                t: nn.Parameter(torch.randn(d_model))
                for t in ["number", "text", "datetime", "boolean"]
            }
        )
        self.blocks = nn.ModuleList(
            [
                RelationalBlock(d_model, num_heads, d_ff, legacy_attn)
                for i in range(num_blocks)
            ]
        )
        self.norm_out = RMSNorm(d_model, eps=1e-6)
        self.d_model = d_model

        # zero-init output weights
        for module in self.dec_dict.values():
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

        if compile:
            self.forward = torch.compile(self.forward, dynamic=False)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path,
        *,
        device: str = "cpu",
        compile: bool = False,
        revision: str | None = None,
        subfolder: str | None = None,
        **model_kwargs,
    ):
        """Load a pretrained RT model from a local path *or* the HuggingFace Hub.

        ``model_id_or_path`` may be a Hub repo (``org/repo[/subdir]``), a local
        checkpoint directory, or a local weights file; a sibling ``config.json``
        supplies the model dims and the text-embedding model. A Hub reference is
        downloaded and cached on demand; a local path is used as-is and never
        triggers a download. ``subfolder`` selects a sub-directory within the
        repo/directory (the HuggingFace-idiomatic way to pick one of several
        checkpoints; equivalent to appending it to ``model_id_or_path``). Extra
        keyword args fill/override model dims for checkpoints that ship without a
        ``config.json``.

        Returns the model, moved to ``device``, with its resolved ``config`` dict
        attached as ``model.config``.

            model = RelationalTransformer.from_pretrained("stanford-star/rt-j", subfolder="classification")
            model = RelationalTransformer.from_pretrained("stanford-star/rt-j/classification")
            model = RelationalTransformer.from_pretrained("/path/to/checkpoint")
        """
        from rt.checkpoints import (
            MODEL_DIM_KEYS,
            _adapt_state_dict,
            load_model,
            resolve_checkpoint,
        )

        config, model_path = resolve_checkpoint(
            model_id_or_path, revision=revision, subfolder=subfolder
        )
        # Model dims live under config["model"]; older release configs (e.g.
        # ``stanford-star/rt-plurel``) carry them flat at the top level.
        flat = {k: config[k] for k in MODEL_DIM_KEYS if k in config}
        m = {**flat, **config.get("model", {}), **model_kwargs}
        missing = [k for k in MODEL_DIM_KEYS if k not in m]
        if missing:
            raise ValueError(
                f"checkpoint {model_id_or_path!r} is missing model dims {missing}; "
                f"provide a config.json or pass them as keyword args."
            )
        state_dict = _adapt_state_dict(load_model(model_path))
        # Pre-RT-J checkpoints have no attention gate (wg): run them with the
        # legacy attention math they were trained with.
        legacy_attn = not any(k.endswith(".wg.weight") for k in state_dict)
        model = cls(
            num_blocks=m["num_blocks"],
            d_model=m["d_model"],
            d_text=m["d_text"],
            num_heads=m["num_heads"],
            d_ff=m["d_ff"],
            compile=compile,
            materialize_attn_masks=m.get("materialize_attn_masks", True),
            legacy_attn=legacy_attn,
        )
        model.load_state_dict(state_dict)
        model.config = config
        return model.to(device)

    def forward(self, batch, return_embeddings):
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]

        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device

        # Sort cells by column index (padding stays at end)
        sort_keys = col_name_idxs.masked_fill(
            is_padding, torch.iinfo(col_name_idxs.dtype).max
        )
        sort_idxs = sort_keys.argsort(dim=-1, stable=True)
        si = sort_idxs.unsqueeze(-1)
        node_idxs = node_idxs.gather(1, sort_idxs)
        f2p_nbr_idxs = f2p_nbr_idxs.gather(1, si.expand_as(f2p_nbr_idxs))
        col_name_idxs = col_name_idxs.gather(1, sort_idxs)
        table_name_idxs = table_name_idxs.gather(1, sort_idxs)
        is_padding = is_padding.gather(1, sort_idxs)
        col_name_values = batch["col_name_values"].gather(
            1, si.expand_as(batch["col_name_values"])
        )
        sem_types = batch["sem_types"].gather(1, sort_idxs)
        is_targets = batch["is_targets"].gather(1, sort_idxs)
        type_values = {}
        for t in ["number", "text", "datetime", "boolean"]:
            k = t + "_values"
            type_values[t] = batch[k].gather(1, si.expand_as(batch[k]))

        if self.materialize_attn_masks:
            # Materialize (B, S, S) pairwise masks, then convert to block masks.
            pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])
            same_node = node_idxs[:, :, None] == node_idxs[:, None, :]
            kv_in_f2p = (
                node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]
            ).any(-1)
            q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(
                -1
            )
            same_col_table = (
                col_name_idxs[:, :, None] == col_name_idxs[:, None, :]
            ) & (table_name_idxs[:, :, None] == table_name_idxs[:, None, :])

            attn_masks = {
                "feat": (same_node | kv_in_f2p) & pad,
                "nbr": q_in_f2p & pad,
                "col": same_col_table & pad,
            }
            kv_sizes = {
                attn_type: attn_masks[attn_type].sum(dim=-1, keepdim=True).bfloat16()
                for attn_type in attn_masks
            }
            for attn_type in attn_masks:
                attn_masks[attn_type] = attn_masks[attn_type].contiguous()

            make_block_mask = partial(
                _make_block_mask,
                batch_size=batch_size,
                q_seq_len=seq_len,
                kv_seq_len=seq_len,
                device=device,
            )
            block_masks = {
                attn_type: make_block_mask(attn_mask)
                for attn_type, attn_mask in attn_masks.items()
            }
        else:
            # Build sparse block masks via flex_attention's mask_mod path: closures index
            # into per-token tensors directly. create_block_mask samples mask_mod at block
            # granularity, never materializing a (B, S, S) tensor.
            node_idxs_c = node_idxs.contiguous()
            f2p_nbr_idxs_c = f2p_nbr_idxs.contiguous()
            col_name_idxs_c = col_name_idxs.contiguous()
            table_name_idxs_c = table_name_idxs.contiguous()
            is_padding_c = is_padding.contiguous()

            def feat_mask_mod(b, h, q_idx, kv_idx):
                not_pad = (~is_padding_c[b, q_idx]) & (~is_padding_c[b, kv_idx])
                same_node = node_idxs_c[b, q_idx] == node_idxs_c[b, kv_idx]
                kv_node = node_idxs_c[b, kv_idx]
                in_nbrs = (f2p_nbr_idxs_c[b, q_idx] == kv_node).any(dim=-1)
                return (same_node | in_nbrs) & not_pad

            def nbr_mask_mod(b, h, q_idx, kv_idx):
                not_pad = (~is_padding_c[b, q_idx]) & (~is_padding_c[b, kv_idx])
                q_node = node_idxs_c[b, q_idx]
                in_nbrs = (f2p_nbr_idxs_c[b, kv_idx] == q_node).any(dim=-1)
                return in_nbrs & not_pad

            def col_mask_mod(b, h, q_idx, kv_idx):
                not_pad = (~is_padding_c[b, q_idx]) & (~is_padding_c[b, kv_idx])
                same_col = col_name_idxs_c[b, q_idx] == col_name_idxs_c[b, kv_idx]
                same_table = table_name_idxs_c[b, q_idx] == table_name_idxs_c[b, kv_idx]
                return same_col & same_table & not_pad

            make_bm = partial(
                create_block_mask,
                B=batch_size,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                _compile=True,
            )
            block_masks = {
                "feat": make_bm(feat_mask_mod),
                "nbr": make_bm(nbr_mask_mod),
                "col": make_bm(col_mask_mod),
            }

            # Exact per-query key counts, computed analytically without (B, S, S).
            kv_sizes = _kv_sizes(
                node_idxs_c,
                f2p_nbr_idxs_c,
                col_name_idxs_c,
                table_name_idxs_c,
                is_padding_c,
            )

        x = 0
        x = x + (
            self.norm_dict["col_name"](self.enc_dict["col_name"](col_name_values))
            * (~is_padding)[..., None]
        )

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            # fill in nan values with 0s # FIXME: check rustler (ctu only)
            type_values[t] = torch.where(
                torch.isnan(type_values[t]),
                torch.zeros_like(type_values[t]),
                type_values[t],
            )
            t_values = type_values[t]
            x = x + (
                self.norm_dict[t](self.enc_dict[t](t_values))
                * ((sem_types == i) & ~is_targets & ~is_padding)[..., None]
            )
            x = x + (
                self.mask_embs[t]
                * ((sem_types == i) & is_targets & ~is_padding)[..., None]
            )

        for i, block in enumerate(self.blocks):
            x = block(x, block_masks, kv_sizes)

        x = self.norm_out(x)

        if return_embeddings:
            return x

        yhat_out = {"number": None, "text": None, "datetime": None, "boolean": None}

        B, S, _ = x.shape
        masks = is_targets.bool()  # (B,S) where to train

        loss_per_seq = x.new_zeros(B)
        sem_type_names = ["number", "text", "datetime", "boolean"]
        sem_type_losses = {}

        for i, t in enumerate(sem_type_names):
            yhat = self.dec_dict[t](x)  # (B,S, D_t)
            y = type_values[t]  # (B,S, D_y)
            sem_type_mask = (sem_types == i) & masks  # (B,S) mask for this type

            if t in ("number", "text", "datetime"):
                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)  # (B, S)
            elif t == "boolean":
                loss_t = F.binary_cross_entropy_with_logits(
                    yhat, (y > 0).float(), reduction="none"
                ).mean(-1)  # (B, S)

            # Per-sem_type average loss (clamp avoids 0/0 when type is absent)
            n_tokens = sem_type_mask.sum()
            sem_type_losses[t] = (loss_t * sem_type_mask).sum() / n_tokens.clamp(min=1)

            # Sum loss per sequence for this type
            loss_per_seq = loss_per_seq + (loss_t * sem_type_mask).sum(dim=1)  # (B,)

            yhat_out[t] = yhat

        # Normalize by number of masks per sequence, then average across sequences
        masks_per_seq = masks.sum(dim=1).float()  # (B,)
        loss_per_seq = loss_per_seq / masks_per_seq  # (B,)
        loss_out = loss_per_seq.mean()  # scalar

        return loss_out, yhat_out, sem_type_losses, is_targets

    def predict(self, batch, eval_ctx_sizes, device, task, bool_as_num):
        """Eval-mode predictions at multiple context sizes.

        batch: dict of CPU tensors (B, S_max).
        Returns dict mapping ctx_size → (B,) prediction tensor (CPU), with
        one entry per batch row. Rows with no target (phantom rows from
        last-batch overshoot, batch_mask=false) get 0.0; the caller filters
        them out using batch_mask after gather.
        """
        val_key = "boolean" if task.task_type == "clf" and not bool_as_num else "number"
        preds = {}
        for ctx_size in eval_ctx_sizes:
            trunc = {
                k: v[:, :ctx_size].to(device, non_blocking=True)
                for k, v in batch.items()
            }
            _, yhat_dict, _, sorted_is_targets = self(trunc, return_embeddings=False)
            # Each real row has exactly one target position (eval invariant).
            # Collapse (B, S, 1) → (B,) by summing the per-row target entry.
            # Phantom rows have no target → sum is 0, filtered by batch_mask.
            yhat = yhat_dict[val_key].squeeze(-1)  # (B, S)
            preds[ctx_size] = (yhat * sorted_is_targets.to(yhat.dtype)).sum(dim=1).cpu()
        return preds
