from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import numpy as np


class BlockVarianceScaler(BaseEstimator, TransformerMixin):
    """
    Equalizes the total variance of different-sized blocks of features
    so they have equal weight in subsequent PCA steps.
    """

    def __init__(self, k):
        self.k = k  # Index where static features end

    def fit(self, X, y=None):
        # No fitting required, weights are based strictly on block sizes
        return self

    def transform(self, X, y=None):
        X_weighted = X.copy()
        c = X.shape[1]

        # Calculate block weights (inverse square root of number of features)
        weight_stat = 1.0 / np.sqrt(self.k)
        weight_dyn = 1.0 / np.sqrt(c - self.k)

        # Apply weights to the blocks
        X_weighted[:, : self.k] *= weight_stat
        X_weighted[:, self.k :] *= weight_dyn

        return X_weighted


class JointSplitPCA(BaseEstimator, TransformerMixin):
    """
    A custom transformer that fits a joint PCA and projects
    subsets of features (static vs dynamic) onto the shared basis.
    """

    def __init__(self, k, n_components=2):
        self.k = k  # Index where static features end and dynamic begin
        self.n_components = n_components
        self.pca_ = None
        self.W_ = None
        self.mu_ = None

    def fit(self, X, y=None):
        # 1. Fit joint PCA on the full data
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X)

        # Extract projection matrix and mean
        self.W_ = self.pca_.components_.T
        self.mu_ = self.pca_.mean_
        return self

    def transform(self, X, y=None):
        # 2. Prepare padded means
        mu_stat = np.zeros_like(self.mu_)
        mu_stat[: self.k] = self.mu_[: self.k]

        mu_dyn = np.zeros_like(self.mu_)
        mu_dyn[self.k :] = self.mu_[self.k :]

        # 3. Prepare padded data
        X_stat_padded = np.zeros_like(X)
        X_stat_padded[:, : self.k] = X[:, : self.k]

        X_dyn_padded = np.zeros_like(X)
        X_dyn_padded[:, self.k :] = X[:, self.k :]

        # 4. Project onto the shared basis
        Y_stat = (X_stat_padded - mu_stat) @ self.W_
        Y_dyn = (X_dyn_padded - mu_dyn) @ self.W_

        # 5. Return stacked array (t, 4) for pipeline compatibility
        return Y_stat, Y_dyn


####


from pathlib import Path
import pickle as pkl

from einops import rearrange
import cv2
import torch as pt

from object_centric_bench.util_datum import draw_segmentation_np


def main_plot_ttraj(ttraj_file=Path("ttraj.pkl")):
    import torch.nn.functional as ptnf
    from object_centric_bench.util_model import interpolat_argmax_attent
    from object_centric_bench.util_datum import generate_spectrum_colors

    IMAGENET_MEAN = np.array([[[123.675]], [[116.28]], [[103.53]]], "float32")
    IMAGENET_STD = np.array([[[58.395]], [[57.12]], [[57.375]]], "float32")

    def visualiz_frame_attent(frame, attent, save_file, color):
        """
        - frame: (c,h,w)
        - attent: (s,h,w)
        - color: (s,c)
        """
        c, h, w = frame.shape
        s, _, _ = attent.shape
        mean = pt.from_numpy(IMAGENET_MEAN).cuda()
        std = pt.from_numpy(IMAGENET_STD).cuda()
        frame = (pt.from_numpy(frame).cuda() * std + mean).cuda().clip(0, 255).byte()
        # attent = ptnf.interpolate(pt.from_numpy(attent).cuda(), size=[h, w], mode="bilinear")
        segment = interpolat_argmax_attent(pt.from_numpy(attent).cuda()[None], [h, w])[
            0
        ]
        segment = ptnf.one_hot(segment.long()).bool()
        frame = frame.permute(1, 2, 0).cpu().numpy()
        segment = segment.cpu().numpy()
        viz = draw_segmentation_np(frame, segment, alpha=0.5, color=color)
        cv2.imwrite(save_file, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    with open(ttraj_file, "rb") as f:
        ttraj = pkl.load(f)

    frame, slotz, memory, attentd = [
        ttraj[_] for _ in ["frame", "slotz", "memory", "attentd"]
    ]
    assert len(frame) == len(slotz) == len(memory) == len(attentd)

    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    def func_pca(x, pca=PCA, **kwd2):
        pp = Pipeline([("scaler", StandardScaler()), ("pca", pca(**kwd2))])
        y = pp.fit_transform(x)
        return y

    for ii, (fi, si, mi, ai) in enumerate(zip(frame, slotz, memory, attentd)):
        # if ii != 21:
        #     continue
        b, t, _, h, w = fi.shape
        # (b,t,c,h,w) (b,t,s,c) (b*t,s,c) (b,t,s,h,w)
        mi = rearrange(mi, "(b t) s c -> b t s c", b=b)
        fi, si, mi, ai = [_[0] for _ in [fi, si, mi, ai]]
        # fi = rearrange(fi, "t c h w -> t h w c")
        # ai = rearrange(ai, "t s h w -> t h w s")
        # (t,h,w,c) (t,s,c) (t,s,c) (t,h,w,s)

        _, s, _ = si.shape
        color = generate_spectrum_colors(s)
        savef = ttraj_file.parent / ttraj_file.name[:-4]
        savef.mkdir(parents=True, exist_ok=True)
        for ij, (fij, aij) in enumerate(zip(fi, ai)):
            visualiz_frame_attent(fij, aij, savef / f"{ij:04d}.png", color)

        ### reduce separately to the final dim
        si3 = func_pca(rearrange(si, "t s c -> (t s) c"), n_components=2)
        mi3 = func_pca(rearrange(mi, "t s c -> (t s) c"), n_components=2)
        si3, mi3 = [rearrange(_, "(t s) c -> t s c", t=t) for _ in [si3, mi3]]
        ###
        m13, m23 = Pipeline(
            [
                ("scaler", BlockVarianceScaler(k=288)),
                ("pca", JointSplitPCA(k=288, n_components=2)),
            ]
        ).fit_transform(rearrange(mi, "t s c -> (t s) c"))
        m13, m23 = [rearrange(_, "(t s) c -> t s c", t=t) for _ in [m13, m23]]

        plt.figure(figsize=(5, 5))
        for _i in range(s):
            color_i = np.array(color[_i]) / 255.0
            # (line,) = plt.plot(
            #     si3[:, _i, 0], si3[:, _i, 1], marker="o", lw=0.5, label=f"slots#{_i}"
            # )
            # plt.plot(
            #     mi3[:, _i, 0],
            #     mi3[:, _i, 1],
            #     marker="o",
            #     lw=0.5,
            #     label=f"memory#{_i}",
            #     color=color,
            # )
            (line,) = plt.plot(
                m13[:, _i, 0],
                m13[:, _i, 1],
                marker=".",
                lw=0.5,
                label=f"static#{_i}",
                color=color_i,
            )
            plt.plot(
                m23[:, _i, 0],
                m23[:, _i, 1],
                marker="*",
                lw=0.5,
                label=f"dynamic#{_i}",
                color=color_i,
            )
            # break
        plt.legend(framealpha=1, ncol=2)
        plt.title("Temporal Trajectories of a Set of Slots\n(with Joint PCA)")
        plt.tight_layout()
        plt.savefig(f"{ttraj_file.as_posix()[:-3]}png")
        plt.savefig(f"{ttraj_file.as_posix()[:-3]}pdf")
        plt.savefig(f"{ttraj_file.as_posix()[:-3]}svg")
        plt.show()
        break

    return
