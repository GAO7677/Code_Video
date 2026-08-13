"""RandSFQ2 variant consuming native V-JEPA two-frame tubelets."""

from einops import rearrange
import torch as pt

from .randsfq2 import RandSFQ2


class RandSFQ2VJEPAVideo(RandSFQ2):
    """Run xSSC at the temporal resolution returned by a video backbone."""

    def _extract_slots(self, input, initial_condit=None):
        """Encode a video and recurrently aggregate slots without the decoder.

        ``initial_condit`` is deliberately restricted to ``[B,S,C]``.  The
        slot-only Stage-1 path must not accept a time sequence of future GT
        conditions.
        """
        b = input.shape[0]
        feature = self.encode_backbone(input).detach()  # (b,t,c,h,w)
        b_feature, t, c, h, w = feature.shape
        if b_feature != b:
            raise RuntimeError(f"Backbone changed batch size: {b} -> {b_feature}")
        if initial_condit is not None and initial_condit.ndim != 3:
            raise ValueError(
                "Slot-only initial condition must be [B,S,C], got "
                f"{tuple(initial_condit.shape)}"
            )
        if initial_condit is not None and initial_condit.shape[0] != b:
            raise ValueError(
                "Initial condition batch does not match video batch: "
                f"{initial_condit.shape[0]} != {b}"
            )

        encode = feature.permute(0, 1, 3, 4, 2).flatten(0, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = self.encode_project(encode.flatten(1, 2))
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b, t=t)

        slotz = None
        attenta = []
        for i in range(t):
            if i == 0:
                query_i = self.initializ(b if initial_condit is None else initial_condit)
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
        return feature, slotz, attenta

    def extract_slot_trajectory(self, input, initial_condit=None):
        """Return causal backbone features, slots, and aggregation attention.

        This method never calls the feature-query decoder.  In particular, it
        cannot expose a predictor or metric to a real future V-JEPA query.
        """
        return self._extract_slots(input, initial_condit=initial_condit)

    def forward(self, input, condit=None):
        """
        - input: raw video, shape=(b,t_raw,c,h,w)
        - condit: tubelet-aligned conditions, shape=(b,t_tubelet,s,c)
        """
        if condit is not None and condit.ndim != 4:
            raise ValueError(
                "Training conditions must be [B,T,S,C], got "
                f"{tuple(condit.shape)}"
            )
        initial_condit = None if condit is None else condit[:, 0, :, :]
        feature, slotz, attenta = self._extract_slots(
            input, initial_condit=initial_condit
        )
        b, t, c, h, w = feature.shape
        if condit is not None and condit.shape[:2] != (b, t):
            raise ValueError(
                "Conditions must be aligned to V-JEPA tubelets: "
                f"{tuple(condit.shape[:2])} != {(b, t)}"
            )

        clue = rearrange(feature, "b t c h w -> b t (h w) c")
        # Legacy square checkpoints use the original flat positional embedding.
        # New aspect-ratio configs opt into 2-D interpolation by constructing the
        # positional embedding with a reference spatial_shape.
        decode_spatial_shape = (
            (h, w)
            if getattr(self.decode.posit_embed, "spatial_shape", None) is not None
            else None
        )
        recon, attentd, fsti = self.decode(
            clue, slotz, spatial_shape=decode_spatial_shape
        )
        if self.training:
            feature = feature.gather(
                1, fsti[:, :, None, None, None].expand(-1, -1, c, h, w)
            )
        recon = rearrange(recon, "b t (h w) c -> b t c h w", b=b, h=h, w=w)
        attentd = rearrange(
            attentd, "b t s (h w) -> b t s h w", b=b, h=h, w=w
        )
        return feature, slotz, attenta, recon, attentd
