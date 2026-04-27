from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.layers import DropPath


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def batch_operation(batch_size, fn, inputs, dim, **kwargs):
    # args: list of tensors
    # dim: the dimension to split
    out = []
    for i in range(0, inputs.shape[dim], batch_size):
        out.append(fn(inputs[i:i+batch_size], **kwargs))
    return torch.cat(out, dim=dim)    

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class MLPHead(nn.Module):
    def __init__(self, dim, dim_out, drop_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim_out)
        )

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # (batch * heads, query_dim, context_dim)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))

class PointEmbed(nn.Module):
    def __init__(
        self,
        hidden_dim=48,
        dim=128,
        rgb_embed_dim=0,
        part_embed_dim=0,
        mat_embed_dim=0,
        num_part_classes=0,
        num_mmid=0,
        dim_mat_input=6,
    ):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        self.rgb_embedding_dim = 12
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        rgb_e = torch.pow(2, torch.arange(self.rgb_embedding_dim // 6)).float() * np.pi
        rgb_e = torch.stack([
            torch.cat([rgb_e, torch.zeros(self.rgb_embedding_dim // 6),
                        torch.zeros(self.rgb_embedding_dim // 6)]),
            torch.cat([torch.zeros(self.rgb_embedding_dim // 6), rgb_e,
                        torch.zeros(self.rgb_embedding_dim // 6)]),
            torch.cat([torch.zeros(self.rgb_embedding_dim // 6),
                        torch.zeros(self.rgb_embedding_dim // 6), rgb_e]),
        ])
        self.register_buffer('rgb_basis', rgb_e)

        if rgb_embed_dim > 0:
            rgb_embed_dim += self.rgb_embedding_dim

        mlp_input_dim = self.embedding_dim + 3 + rgb_embed_dim + part_embed_dim + mat_embed_dim

        self.mlp = nn.Linear(
            mlp_input_dim, dim
        )

        if part_embed_dim > 0:
            self.part_embed = nn.Embedding(num_part_classes, part_embed_dim - 1)    # 1 for verification
        if mat_embed_dim > 0:
            self.mmid_embed = nn.Embedding(num_mmid, mat_embed_dim - dim_mat_input) # 5 for e, nu, sigma, phi, rho, 1 for verification

        self.rgb_embed_dim = rgb_embed_dim
        self.part_embed_dim = part_embed_dim
        self.mat_embed_dim = mat_embed_dim

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input, rgb=None, part=None, mat=None):
        # input: B x N x 3
        # rgb: B x N x 3
        # part: B x N
        # mat: B x N x 6 (e, nu, sigma, phi, rho, mmid)
        features = [self.embed(input, self.basis), input]
        if self.rgb_embed_dim > 0:
            if rgb is None:
                rgb_features = torch.zeros(input.shape[0], input.shape[1], self.rgb_embed_dim).to(input.device)
            else:
                one_embed = torch.ones(input.shape[0], input.shape[1], 1).to(input.device)
                rgb_features = torch.cat([self.embed(rgb, self.rgb_basis), rgb, one_embed], dim=2)
            features.append(rgb_features)
        if self.part_embed_dim > 0:
            if part is None:
                part_embed = torch.zeros(input.shape[0], input.shape[1], self.part_embed_dim).to(input.device)
            else:
                part_embed = self.part_embed(part)
                one_embed = torch.ones(input.shape[0], input.shape[1], 1).to(input.device)
                part_embed = torch.cat([part_embed, one_embed], dim=2)
            features.append(part_embed)
        if self.mat_embed_dim > 0:
            if mat is None:
                mat_embed = torch.zeros(input.shape[0], input.shape[1], self.mat_embed_dim).to(input.device)
            else:
                mmid = mat[:, :, -1].long()
                mmid_embed = self.mmid_embed(mmid)  # B x N -> B x N x C_mmid
                one_embed = torch.ones(input.shape[0], input.shape[1], 1).to(input.device)
                mat_embed = torch.cat([mat[:, :, :-1], mmid_embed, one_embed], dim=2)
            features.append(mat_embed)
        features = torch.cat(features, dim=2)
        embed = self.mlp(features) # B x N x C
        return embed

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
