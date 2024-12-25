import numpy as np
# import tensorflow as tf

from .pygkdtree import PyGKDTFilter


class BilateralStep:
    def __init__(self, height, width, n_refchannels, n_valchannels) -> None:
        self.h, self.w = height, width
        self.n = height * width
        self.pd = n_refchannels + 2
        self.vd = n_valchannels + 1

        self.gkdt_filter = PyGKDTFilter(pd=self.pd, vd=self.vd, n=self.n)

    def init(self, reference: np.ndarray, theta_alpha: np.float32, theta_beta: np.float32):
        spat_pos = np.mgrid[0:self.h, 0:self.w][::-1].astype(np.float32).transpose((1, 2, 0)) / theta_alpha
        colr_pos = reference / theta_beta
        pos = np.concatenate([spat_pos, colr_pos], axis=-1)
        pos = pos.reshape((-1, ))

        self.gkdt_filter.init(pos)

    def compute(self, value: np.ndarray):
        val1 = np.concatenate([value, np.ones((self.h, self.w, 1), dtype=np.float32)], axis=-1)
        val1 = val1.reshape((-1, ))

        self.gkdt_filter.compute(val1, val1)

        val1 = val1.reshape((self.h, self.w, self.vd))
        out = val1[..., :-1] / val1[..., -1:]

        return out

