"""
Copyright (c) 2026 Genera1Z
https://github.com/Genera1Z
"""

from einops import rearrange, repeat
import torch as pt

from .smoothsa import SmoothSAVideo


class SmoothSAVideo2(SmoothSAVideo):
    """
    Almost identical to `SmoothSAVideo`,
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
                qinit0, query_i = self.initializ(
                    encode[:, 0, :, :], None if condit is None else condit[:, 0, :, :]
                )
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

        return feature, qinit0, slotz, attenta, recon, attentd
