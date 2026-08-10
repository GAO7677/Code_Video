"""RandSFQ2 variant consuming native V-JEPA two-frame tubelets."""

from einops import rearrange
import torch as pt

from .randsfq2 import RandSFQ2


class RandSFQ2VJEPAVideo(RandSFQ2):
    """Run xSSC at the temporal resolution returned by a video backbone."""

    def forward(self, input, condit=None):
        """
        - input: raw video, shape=(b,t_raw,c,h,w)
        - condit: tubelet-aligned conditions, shape=(b,t_tubelet,s,c)
        """
        b = input.shape[0]
        feature = self.encode_backbone(input).detach()  # (b,t,c,h,w)
        b_feature, t, c, h, w = feature.shape
        if b_feature != b:
            raise RuntimeError(f"Backbone changed batch size: {b} -> {b_feature}")
        if condit is not None and condit.shape[:2] != (b, t):
            raise ValueError(
                "Conditions must be aligned to V-JEPA tubelets: "
                f"{tuple(condit.shape[:2])} != {(b, t)}"
            )

        encode = feature.permute(0, 1, 3, 4, 2).flatten(0, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = self.encode_project(encode.flatten(1, 2))
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b, t=t)

        slotz = None
        attenta = []
        for i in range(t):
            if i == 0:
                query_i = self.initializ(b if condit is None else condit[:, 0, :, :])
            else:
                query_i = self.transit(slotz, encode[:, : i + 1, :, :])

            niter = None if i == 0 else 1
            slotz_i, attenta_i = self.aggregat(
                encode[:, i, :, :], query_i, num_iter=niter
            )
            slotz = (
                slotz_i[:, None, :, :]
                if slotz is None
                else pt.concat([slotz, slotz_i[:, None, :, :]], 1)
            )
            attenta.append(attenta_i)

        attenta = pt.stack(attenta, 1)
        attenta = rearrange(attenta, "b t s (h w) -> b t s h w", h=h, w=w)

        clue = rearrange(feature, "b t c h w -> b t (h w) c")
        recon, attentd, fsti = self.decode(clue, slotz)
        if self.training:
            feature = feature.gather(
                1, fsti[:, :, None, None, None].expand(-1, -1, c, h, w)
            )
        recon = rearrange(recon, "b t (h w) c -> b t c h w", b=b, h=h, w=w)
        attentd = rearrange(
            attentd, "b t s (h w) -> b t s h w", b=b, h=h, w=w
        )
        return feature, slotz, attenta, recon, attentd
