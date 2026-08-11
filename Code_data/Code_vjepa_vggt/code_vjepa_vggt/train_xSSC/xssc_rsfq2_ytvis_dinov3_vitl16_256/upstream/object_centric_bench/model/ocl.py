"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from .basic import MLP


class SlotAttention(nn.Module):
    """"""

    def __init__(
        self, num_iter, embed_dim, ffn_dim, dropout=0, kv_dim=None, trunc_bp=None
    ):
        """
        - dropout: only works in self.ffn; a bit is beneficial
        """
        super().__init__()
        kv_dim = kv_dim or embed_dim
        assert trunc_bp in ["bi-level", None]
        self.num_iter = num_iter
        self.trunc_bp = trunc_bp
        self.norm1q = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1kv = nn.LayerNorm(kv_dim)
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False)
        # self.dropout = nn.Dropout(dropout)  # always bad for attention
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], None, dropout)

    def forward(self, input, query, smask=None, num_iter=None):
        """
        input: in shape (b,h*w,c)
        query: in shape (b,n,c)
        smask: slots' mask, shape=(b,n), dtype=bool. True means there is a valid slot.
        """
        b, n, c = query.shape
        self_num_iter = num_iter or self.num_iter
        kv = self.norm1kv(input)
        k = self.proj_k(kv)
        v = self.proj_v(kv)
        q = query
        for _ in range(self_num_iter):
            if _ + 1 == self_num_iter:
                if self.trunc_bp == "bi-level":  # BO-QSA
                    q = q.detach() + query - query.detach()
            x = q
            q = self.norm1q(q)
            q = self.proj_q(q)
            u, a = __class__.inverted_scaled_dot_product_attention(q, k, v, smask)
            y = self.rnn(u.flatten(0, 1), x.flatten(0, 1)).view(b, n, -1)
            z = self.norm2(y)
            q = y + self.ffn(z)  # droppath on ffn seems harmful
        return q, a

    @staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, smask=None, eps=1e-5):
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        if smask is not None:
            logit = logit.where(smask[:, :, None], -pt.inf)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        return o, a0

    """@staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, eps=1e-5, h=4):
        q = rearrange(q, "b q (h d) -> (b h) q d", h=h)
        k = rearrange(k, "b k (h d) -> (b h) k d", h=h)
        v = rearrange(v, "b k (h d) -> (b h) k d", h=h)
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        o = rearrange(o, "(b h) q d -> b q (h d)", h=h)
        return o, a0"""


class CartesianPositionalEmbedding2d(nn.Module):
    """"""

    def __init__(self, resolut: list, embed_dim: int):
        super().__init__()
        assert len(resolut) == 2
        self._pe = nn.Parameter(
            __class__.meshgrid(resolut)[None, ...], requires_grad=False
        )
        self.project = nn.Linear(4, embed_dim)

    @staticmethod
    def meshgrid(resolut, low=-1, high=1):
        assert len(resolut) == 2
        yx = [pt.linspace(low, high, _ + 1) for _ in resolut]
        yx = [(_[:-1] + _[1:]) / 2 for _ in yx]
        grid_y, grid_x = pt.meshgrid(*yx)
        return pt.stack([grid_y, grid_x, 1 - grid_y, 1 - grid_x], 2)

    def forward(self, input):
        """
        input: in shape (b,h,w,c)
        output: in shape (b,h,w,c)
        """
        max_h, max_w = input.shape[1:3]
        output = input + self.project(self._pe[:, :max_h, :max_w, :])
        return output

    @property
    def pe(self):
        return self.project(self._pe)  # .flatten(1, -2)


class LearntPositionalEmbedding(nn.Module):
    """Support any dimension. Must be channel-last.
    PositionalEncoding: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self,
        resolut: list,
        embed_dim: int,
        in_dim: int = 0,
        spatial_shape=None,
    ):
        super().__init__()
        self.resolut = resolut
        self.embed_dim = embed_dim
        self.spatial_shape = (
            None
            if spatial_shape is None
            else tuple(int(value) for value in spatial_shape)
        )
        if self.spatial_shape is not None:
            if len(resolut) != 1 or len(self.spatial_shape) != 2:
                raise ValueError(
                    "spatial_shape requires one flattened resolution dimension"
                )
            if self.spatial_shape[0] * self.spatial_shape[1] != resolut[0]:
                raise ValueError(
                    f"spatial_shape {self.spatial_shape} does not match {resolut}"
                )
        if in_dim:
            self._pe = nn.Parameter(pt.zeros(1, *resolut, in_dim), requires_grad=True)
            self._project = nn.Linear(in_dim, embed_dim)
        else:
            self._pe = nn.Parameter(
                pt.zeros(1, *resolut, embed_dim), requires_grad=True
            )
        nn.init.trunc_normal_(self._pe)

    @property
    def pe(self):
        if hasattr(self, "_project"):
            return self._project(self._pe)
        return self._pe

    def forward(self, input, retp=False):
        """
        input: in shape (b,*r,c)
        output: in shape (b,*r,c)
        """
        max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
        pe = eval(f"self.pe[:, {max_r}, :]")
        output = input + pe
        if retp:
            return output, pe
        return output

    def interpolate_2d(self, height, width):
        """Interpolate a flattened learned 2D grid without changing parameters."""
        if self.spatial_shape is None:
            raise RuntimeError("interpolate_2d requires spatial_shape at construction")
        height, width = int(height), int(width)
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid target positional grid: {(height, width)}")
        base_height, base_width = self.spatial_shape
        pe = self.pe.reshape(1, base_height, base_width, self.embed_dim)
        if (height, width) == self.spatial_shape:
            return pe.flatten(1, 2)
        pe = pe.permute(0, 3, 1, 2)
        pe = ptnf.interpolate(
            pe,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return pe.permute(0, 2, 3, 1).flatten(1, 2)

    def extra_repr(self):
        return f"{self.resolut}, {self.embed_dim}, spatial_shape={self.spatial_shape}"


class NormalSeparat(nn.Module):
    """Separate gaussians as queries."""

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, num, dim))
        self.logstd = nn.Parameter(
            (pt.ones(1, num, dim) * dim**-0.5).log()
        )  # scheduled std cause nan in dinosaur; here is learnt
        nn.init.xavier_uniform_(self.mean[0, :, :])  # very important

    def forward(self, b):
        smpl = self.mean.expand(b, -1, -1)
        if self.training:
            randn = pt.randn_like(smpl)
            smpl = smpl + randn * self.logstd.exp()
        return smpl

    def extra_repr(self):
        return f"1, {self.num}, {self.dim}"


class NormalShared(nn.Module):
    """Shared gaussian as queries."""

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, 1, dim))
        self.logstd = nn.Parameter(pt.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.mean)
        nn.init.xavier_uniform_(self.logstd)

    def forward(self, b, n=None):
        self_num = self.num
        if n is not None:
            self_num = n
        smpl = self.mean.expand(b, self_num, -1)
        randn = pt.randn_like(smpl)
        smpl = smpl + randn * self.logstd.exp()
        return smpl
