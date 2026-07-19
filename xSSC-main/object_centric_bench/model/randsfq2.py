"""
Copyright (c) 2026 Genera1Z
https://github.com/Genera1Z
"""

from einops import rearrange, repeat
import torch as pt
import torch.nn as nn

from .randsfq import RandSFQ


class RandSFQ2(RandSFQ):
    """
    Almost identical to `RandSFQ`,
    except code blocks wrapped by `### <<<` and `### >>>`
    """

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,s,c)
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
            if i == 0:  # (b,s,c)
                query_i = self.initializ(b if condit is None else condit[:, 0, :, :])
            else:  # slotz: [0,i); encode: [0,i]
                query_i = self.transit(slotz, encode[:, : i + 1, :, :])

            niter = None if i == 0 else 1
            slotz_i, attenta_i = self.aggregat(
                encode[:, i, :, :], query_i, num_iter=niter
            )

            slotz = (  # (b,i+1,s,c)
                slotz_i[:, None, :, :]
                if slotz is None
                else pt.concat([slotz, slotz_i[:, None, :, :]], 1)
            )
            attenta.append(attenta_i)  # t*(b,s,h*w)

        attenta = pt.stack(attenta, 1)  # (b,t,s,h*w)
        attenta = rearrange(attenta, "b t s (h w) -> b t s h w", h=h)

        ### <<<
        clue = rearrange(feature, "b t c h w -> b t (h w) c")
        recon, attentd, fsti = self.decode(clue, slotz)  # (b,t,h*w,c)
        if self.training:
            # feature0 = feature.clone()
            feature = feature.gather(
                1, fsti[:, :, None, None, None].expand(-1, -1, c, h, w)
            )
            # assert (
            #     (feature0 == feature).all([2, 3, 4])
            #     == (
            #         fsti
            #         == pt.arange(t, dtype=pt.long, device=feature.device)[
            #             None, :
            #         ].expand(b, -1)
            #     )
            # ).all()
        recon = rearrange(recon, "b t (h w) c -> b t c h w", b=b, h=h)
        attentd = rearrange(attentd, "b t s (h w) -> b t s h w", b=b, h=h)
        ### >>>

        return feature, slotz, attenta, recon, attentd


from .dias import ARRandTransformerDecoder


class MarkovRarDecoder(ARRandTransformerDecoder):
    """
    Markov Decision Process Random Auto-Regressive Transformer Decoder
    """

    def __init__(
        self,
        dt: int,  # max predict-back time difference; 0 <= dt < training clip length
        emb_dim,
        posit_embed,
        project1,
        project2,
        backbone,
        readout,
        prob=1,  # p of swap (static channels)
        rd=0.25,  # ratio of dynamic channels; rd=0: swap-all; rd=1: swap-none
    ):
        super().__init__(emb_dim, posit_embed, project1, project2, backbone, readout)
        """
        # for predict-past-future:
        self.te = nn.Embedding(dt * 2 + 1, emb_dim)
        """
        # for predict-past or predict-future
        self.te = nn.Embedding(dt + 1, emb_dim)
        self.dt = dt
        self.register_buffer("prob", pt.tensor(prob, dtype=pt.float), persistent=False)
        self.rd = rd

    def forward(self, input, slotz, smask=None, p=0.5):
        """
        - input: shape=(b,t,n=h*w,c)
        - slotz: shape=(b,t,s,c)
        - smask: shape=(b,t,s)
        """
        b, t, n, c = input.shape
        # assert n == self.posit_embed.pe.size(1)
        _, _, s, _ = slotz.shape
        bt = b * t
        device = input.device

        input = rearrange(input, "b t n c -> (b t) n c")
        slotz = rearrange(slotz, "b t s c -> (b t) s c")

        # as long as no `.contiguous()` right after, `.expand()` is faster than `repeat(...)`
        # TODO XXX disable masking in val for attent2 !!!

        tokens = self.project1(input)  # (b*t,n,c)

        if self.training:
            idxs = pt.vmap(  # (b*t,n)
                lambda _: pt.randperm(n, device=device), randomness="different"
            )(tokens)
            idxs_expanded = idxs[:, :, None].expand(-1, -1, c)

            idxs0 = pt.arange(0, n, device=device)[None, :]  # (1,n)
            keep1 = pt.randint(0, n - 1, [bt, 1], device=device)  # (b*t,1)
            keep2 = pt.ones(bt, 1, dtype=pt.long, device=device) * int(256 * 0.1) - 1
            cond = pt.rand(bt, 1, device=device) < p
            keep = pt.where(cond, keep1, keep2)
            mask = idxs0 < keep  # (b,n)

            # shuffle tokens
            tokens_shuffled = tokens.gather(1, idxs_expanded)  # (b*t,n,c)
            # mask tokens
            mask_token_expanded = self.mask_token.expand(bt, n, -1)
            tokens_masked = tokens_shuffled.where(mask[:, :, None], mask_token_expanded)

            # shuffle pe
            pe_expanded = self.posit_embed.pe[:, :n, :].expand(bt, -1, -1)  # (b*t,n,c)
            pe_shuffled = pe_expanded.gather(1, idxs_expanded)  # (b*t,n,c)
            query = tokens_masked + pe_shuffled

        else:
            query = tokens + self.posit_embed.pe[:, :n, :]

        memory = self.project2(slotz)  # (b*t,s,c)

        ### <<< reorder and index temporally
        ti0 = pt.arange(t, dtype=pt.long, device=device)[None, :].expand(b, -1)
        cs = int(memory.size(-1) * (1 - self.rd))
        cd = int(memory.size(-1) * self.rd)
        if self.training:
            """
            # predict past-present-future, without repeatition or omission
            ... = pt.randperm(t, dtype=pt.long, device=device)
            """
            # predict past-present (or present-future), with repetition or omission
            fsti = __class__.random_time_index(b, t, device, self.dt, mode="le")
            # if self.rd > 0:
            #     query0 = query.clone()
            #     memory0 = memory.clone()
            query_d = (
                query[:, :, cs:]
                .unflatten(0, [b, t])  # (b,t,n,c)
                .gather(1, fsti[:, :, None, None].expand(-1, -1, n, cd))
                .flatten(0, 1)  # (b*t,n,c)
            )
            memory_d = (
                memory[:, :, cs:]
                .unflatten(0, [b, t])  # (b,t,s,c)
                .gather(1, fsti[:, :, None, None].expand(-1, -1, s, cd))
                .flatten(0, 1)  # (b*t,s,c)
            )
            query = pt.concat([query[:, :, :cs], query_d], -1)
            memory = pt.concat([memory[:, :, :cs], memory_d], -1)
            # if self.rd > 0:
            #     assert ((query0 == query).all([1, 2]) == (ti0 == fsti).flatten()).all()
            #     assert (
            #         (memory0 == memory).all([1, 2]) == (ti0 == fsti).flatten()
            #     ).all()
        else:  # (b*t,1,c)
            fsti = ti0
        # assert (ti0 >= fsti).all() and (ti0 - fsti).le(self.dt).all()
        fste_s = self.te(ti0 - fsti).flatten(0, 1)[:, None, :cs]
        te0 = self.te.weight[0][None, None, cs:].expand(b * t, -1, -1)
        fste = pt.concat([fste_s, te0], -1)
        query = pt.concat([fste, query], dim=1)  # (b*t,1+h*w,c)
        memory = pt.concat([fste, memory], dim=1)  # (b*t,1+s,c)
        ### >>>

        autoreg = self.backbone(
            self.norm0(query),
            memory=memory,
            memory_key_padding_mask=None if smask is None else ~smask,
        )
        recon = self.readout(autoreg)  # (b*t,1+h*w,c)
        _, _, d = recon.shape

        ### <<< remove concated temb
        recon = recon[:, 1:, :]  # (b*t,h*w,c)
        self._attent = self._attent[:, 1:, 1:]  # (b*t,h*w,s)
        ### >>>

        if self.training:
            idxs_inverse = idxs.argsort(1)[:, :, None]
            recon = recon.gather(1, idxs_inverse.expand(-1, -1, d))
            attent = self._attent.gather(1, idxs_inverse.expand(-1, -1, s))
        else:
            attent = self._attent

        recon = rearrange(recon, "(b t) n c -> b t n c", b=b)
        attent = rearrange(attent, "(b t) n s -> b t s n", b=b)
        return recon, attent, fsti

    @pt.no_grad()
    @staticmethod
    def random_time_index(b, t, device, dt=None, mode="le"):
        """
        :param b: batch size
        :param t: total number of time steps
        :param dt: max time difference
        """
        if dt is None:
            dt = t - 1
        else:
            assert 0 <= dt < t
        if mode == "le":  # less-than or equal-to
            upper = pt.arange(t, device=device)[None, :]
            lower = (upper - dt).clamp(min=0)
        elif mode == "ge":  # greater-than or equal-to
            lower = pt.arange(t, device=device)[None, :]
            upper = (lower + dt).clamp(max=t - 1)
        else:
            raise ValueError
        range_size = upper - lower + 1
        t1 = (
            pt.rand(b, t, dtype=pt.float16, device=device) * range_size
        ).floor().long() + lower
        return t1
