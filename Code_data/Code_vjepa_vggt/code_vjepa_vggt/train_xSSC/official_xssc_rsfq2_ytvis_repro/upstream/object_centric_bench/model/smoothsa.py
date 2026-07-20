"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

from einops import rearrange, repeat
import torch as pt
import torch.nn as nn


class SmoothSA(nn.Module):
    """
    Slot Attention with Re-Initialization and Self-Distillation.
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,  # trunc_bp=false: bad
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode
        __class__.reset_parameters(  # reset self.decode: no difference
            [self.encode_posit_embed, self.encode_project, self.aggregat]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def forward(self, input, condit=None):
        """
        - input: image, shape=(b,c,h,w)
        - condit: condition, shape=(b,n,c)
        """
        feature = self.encode_backbone(input).detach()  # (b,c,h,w)
        b, c, h, w = feature.shape

        encode = feature.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b,h*w,c)
        encode = self.encode_project(encode)

        qinit, query = self.initializ(encode, condit)  # (b,n,c)
        slotz, attenta = self.aggregat(encode, query)
        attenta = rearrange(attenta, "b n (h w) -> b n h w", h=h)

        clue = rearrange(feature, "b c h w -> b (h w) c")
        recon, attentd = self.decode(clue, slotz)  # (b,h*w,c)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        attentd = rearrange(attentd, "b n (h w) -> b n h w", h=h)

        return feature, qinit, slotz, attenta, recon, attentd


class SmoothSAVideo(SmoothSA):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,  # trunc_bp=false: bad
        transit,
        decode,
    ):
        super().__init__(
            encode_backbone,
            encode_posit_embed,
            encode_project,
            initializ,
            aggregat,
            decode,
        )
        self.transit = transit
        __class__.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.transit]
        )

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,n,c)
        """
        b, t, c0, h0, w0 = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        slotz = None
        attenta = []

        for i in range(t):
            if i == 0:
                qinit0, query_i = self.initializ(
                    encode[:, 0, :, :], None if condit is None else condit[:, 0, :, :]
                )  # (b,n,c)
            else:
                query_i = self.transit(slotz, encode[:, : i + 1, :, :])

            niter = None if i == 0 else 1
            slotz_i, attenta_i = self.aggregat(
                encode[:, i, :, :], query_i, num_iter=niter
            )

            slotz = (  # (b,i+1,n,c)
                slotz_i[:, None, :, :]
                if slotz is None
                else pt.concat([slotz, slotz_i[:, None, :, :]], 1)
            )
            attenta.append(attenta_i)  # t*(b,n,h*w)

        attenta = pt.stack(attenta, 1)  # (b,t,n,h*w)
        attenta = rearrange(attenta, "b t n (h w) -> b t n h w", h=h)

        clue = rearrange(feature, "b t c h w -> (b t) (h w) c")
        recon, attentd = self.decode(clue, slotz.flatten(0, 1))  # (b*t,h*w,c)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attentd = rearrange(attentd, "(b t) n (h w) -> b t n h w", b=b, h=h)

        return feature, qinit0, slotz, attenta, recon, attentd


class NormalSharedPreheated(nn.Module):  #  > normal separate

    def __init__(self, num, emb_dim, kv_dim):
        super().__init__()
        self.num = num
        self.emb_dim = emb_dim
        self.kv_dim = kv_dim

        # zero mean > xavier_uniform, xavier_normal or randn mean
        self.mean = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))
        self.logstd = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))

        self.qproj_kv = nn.Linear(kv_dim, emb_dim)  # > ln, fc, lnfc, fcln, mlp
        self.qinit = nn.TransformerDecoderLayer(  # > SwappedTransformerDecoderLayer, i.e., different qdim and kvdim
            emb_dim,
            # kv_dim,
            nhead=4,
            dim_feedforward=emb_dim * 4,
            dropout=0,  # 0 vs 0.1, 0.5: good for arifg
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.qinit.forward = forward_switch_sa_ca.__get__(self.qinit, type(self.qinit))
        if self.qinit.norm_first:
            del self.qinit.norm2  # good for arifg
            self.qinit.norm2 = lambda _: _

        self.logstd2 = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))
        self.register_buffer("detach_flag", pt.tensor(1, dtype=pt.bool))

    def forward(self, encode, n: int = None):
        b, hw, c = encode.shape
        self_num = self.num
        if n is not None:
            self_num = n

        mean = self.mean.expand(b, self_num, -1)
        randn = pt.randn_like(mean)  # better than not
        smpl = mean + randn * self.logstd.exp()

        if self.detach_flag:  # detach initial > always detach
            encode = encode.detach()
        qinit = self.qinit(smpl, self.qproj_kv(encode))
        # in training, start from smpl as qinit than switch to real qinit: bad

        if self.training:
            randn2 = pt.randn_like(qinit)  # better than not
            query = qinit.detach() + randn2 * self.logstd2.exp()
        else:
            query = qinit.detach()
        # align qinit with slotz > align qinit+std with slotz
        return qinit, query  # > query, query.detach()


from .basic import MLP


class NormalMlpPreheated(nn.Module):

    def __init__(self, in_dim, dims, kv_dim, mlpln="post", pad_value=-1):
        super().__init__()
        emb_dim = dims[-1]
        self.emb_dim = emb_dim
        self.kv_dim = kv_dim
        self.pad_value = pad_value

        self.mlp = MLP(in_dim, dims, mlpln, 0)
        self.logstd = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))

        self.qproj_kv = nn.Linear(kv_dim, emb_dim)
        self.qinit = nn.TransformerDecoderLayer(  # SwappedTransformerDecoderLayer
            emb_dim,
            # kv_dim,
            nhead=4,
            dim_feedforward=emb_dim * 4,
            dropout=0,  # 0 vs 0.1, 0.5: good for arifg
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=False,
        )
        self.qinit.forward = forward_switch_sa_ca.__get__(self.qinit, type(self.qinit))
        if self.qinit.norm_first:
            del self.qinit.norm2  # good for arifg
            self.qinit.norm2 = lambda _: _

        self.logstd2 = nn.Parameter(pt.zeros(1, 1, emb_dim, dtype=pt.float))

        self.register_buffer("detach_flag", pt.tensor(1, dtype=pt.bool))

    def forward(self, encode, condit):
        """
        - encode: shape=(b,h*w,c)
        - condit: shape=(b,n,c)
        """
        pad_flag = (condit == self.pad_value).all(2)  # (b,n)

        mean = self.mlp(condit)
        randn = pt.randn_like(mean)  # better than not
        smpl = (
            mean + randn * self.logstd.exp()
        )  # > share_randn0/1_on_pad (different on batch)

        if self.detach_flag:
            encode = encode.detach()
        qinit = self.qinit(smpl, self.qproj_kv(encode))

        if self.training:
            # stop-grad on padded slots
            ppad_flag = pt.concat([pt.zeros_like(pad_flag[:, :1]), pad_flag[:, :-1]], 1)
            # print(pad_flag)
            # print(ppad_flag)
            qinit = pt.where(ppad_flag[:, :, None], qinit.detach(), qinit)

            randn2 = pt.randn_like(qinit)  # better than not
            query = qinit.detach() + randn2 * self.logstd2.exp()
        else:
            query = qinit.detach()
        return qinit, query  # > query, query.detach()


def forward_switch_sa_ca(
    self,
    tgt,
    memory,
    tgt_mask=None,
    memory_mask=None,
    tgt_key_padding_mask=None,
    memory_key_padding_mask=None,
    tgt_is_causal: bool = False,
    memory_is_causal: bool = False,
):
    x = tgt
    if self.norm_first:
        x = x + self._mha_block(  # swape self-att and cross-att
            self.norm2(x),
            memory,
            memory_mask,
            memory_key_padding_mask,
            memory_is_causal,
        )
        x = x + self._sa_block(
            self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
        )
        x = x + self._ff_block(self.norm3(x))
    else:
        x = self.norm2(
            x
            + self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
            )
        )
        x = self.norm1(
            x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
        )
        x = self.norm3(x + self._ff_block(x))

    return x
