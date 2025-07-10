
import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention import SDPBackend, sdpa_kernel

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)


m_f2p = None
m_p2f = None
m_col = None


def mask_mod_f2p(b, h, q_idx, kv_idx):
    return m_f2p[b, q_idx, kv_idx]


def mask_mod_p2f(b, h, q_idx, kv_idx):
    return m_p2f[b, q_idx, kv_idx]


def mask_mod_col(b, h, q_idx, kv_idx):
    return m_col[b, q_idx, kv_idx]


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, block_mask):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if block_mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        else:
            x = flex_attention(q, k, v, block_mask=block_mask)

        x = rearrange(x, "b h s d -> b s (h d)")
        x = self.wo(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.norm_attn_f2p = nn.RMSNorm(d_model)
        self.attn_f2p = Attention(d_model, num_heads)

        self.norm_attn_p2f = nn.RMSNorm(d_model)
        self.attn_p2f = Attention(d_model, num_heads)

        self.norm_attn_col = nn.RMSNorm(d_model)
        self.attn_col = Attention(d_model, num_heads)

        self.norm_attn = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, num_heads)

        self.norm_ffn = nn.RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, bm_f2p, bm_p2f, bm_col):
        x = x + self.attn_col(self.norm_attn_col(x), block_mask=bm_col)
        x = x + self.attn_f2p(self.norm_attn_f2p(x), block_mask=bm_f2p)
        x = x + self.attn_p2f(self.norm_attn_p2f(x), block_mask=bm_p2f)
        x = x + self.attn(self.norm_attn(x), block_mask=None)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class MLP(nn.Module):
    def __init__(self, d_in, d_mlp, d_out):
        super().__init__()

        self.w1 = nn.Linear(d_in, d_mlp, bias=True)
        self.w2 = nn.Linear(d_mlp, d_out, bias=True)
        self.w3 = nn.Linear(d_in, d_mlp, bias=True)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        d_text,
        num_heads,
        d_ff,
        loss,
    ):
        super().__init__()
        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, bias=True),
                "text": nn.Linear(d_text, d_model, bias=True),
                "datetime": nn.Linear(1, d_model, bias=True),
                "table_name": nn.Linear(d_text, d_model, bias=True),
                "col_name": nn.Linear(d_text, d_model, bias=True),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, bias=True),
                "text": nn.Linear(d_model, d_text, bias=True),
                "datetime": nn.Linear(d_model, 1, bias=True),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {
                "number": nn.RMSNorm(d_model),
                "text": nn.RMSNorm(d_model),
                "datetime": nn.RMSNorm(d_model),
                "table_name": nn.RMSNorm(d_model),
                "col_name": nn.RMSNorm(d_model),
            }
        )
        self.mask_embs = nn.ParameterDict(
            {
                t: nn.Parameter(torch.randn(d_model))
                for t in ["number", "text", "datetime"]
            }
        )
        self.layers = nn.ModuleList(
            [Layer(d_model, num_heads, d_ff) for i in range(num_layers)]
        )
        self.norm_out = nn.RMSNorm(d_model)
        self.d_model = d_model
        self.loss = loss

    def forward(self, batch):
        global m_f2p, m_p2f, m_col

        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        batch_size, seq_len = node_idxs.shape

        m_f2p = (node_idxs[:, :, None] == node_idxs[:, None, :]) | (
            node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]
        ).any(-1)
        m_p2f = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(-1)
        m_col = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )

        # TODO: avoid graph break?
        bm_f2p = create_block_mask(
            mask_mod=mask_mod_f2p,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=node_idxs.device,
            _compile=True,
        )
        bm_p2f = create_block_mask(
            mask_mod=mask_mod_p2f,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=node_idxs.device,
            _compile=True,
        )

        bm_col = create_block_mask(
            mask_mod=mask_mod_col,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=node_idxs.device,
            _compile=True,
        )

        x = 0
        for t in ["table_name", "col_name"]:
            x = x + self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))

        for i, t in enumerate(["number", "text", "datetime"]):
            x = x + (
                self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~batch["masks"])[..., None]
            )
            x = x + (
                self.mask_embs[t]
                * ((batch["sem_types"] == i) & batch["masks"])[..., None]
            )

        for i, layer in enumerate(self.layers):
            x = layer(x, bm_f2p, bm_p2f, bm_col)

        x = self.norm_out(x)

        loss_out = 0
        for i, t in enumerate(["number", "text", "datetime"]):
            yhat = self.dec_dict[t](x)
            y = batch[t + "_values"]
            if self.loss == "bce":
                loss = F.binary_cross_entropy_with_logits(
                    yhat, (y > 0).float(), reduction="none"
                ).mean(-1)
            elif self.loss == "mse":
                loss = F.mse_loss(yhat, y, reduction="none").mean(-1)
            elif self.loss == "huber":
                loss = F.huber_loss(yhat, y, reduction="none").mean(-1)
            else:
                raise ValueError(f"Unknown loss: {loss}")
            loss_out = (
                loss_out + (loss * ((batch["sem_types"] == i) & batch["masks"])).sum()
            )
            if t == "number":
                yhat_out = yhat

        loss_out = loss_out / batch["masks"].sum()

        return loss_out, yhat_out
