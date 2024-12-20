import numpy as np
# import tensorflow as tf

from .pygkdtree_v9 import PyGKDTFilter


class SpatialStep:
    def __init__(self, height, width, n_valchannels) -> None:
        self.h, self.w = height, width
        self.n = height * width
        self.pd = 2
        self.vd = n_valchannels + 1

        self.gkdt_filter = PyGKDTFilter(pd=self.pd, vd=self.vd, n=self.n)

    def init(self, theta_gamma: np.float32):
        spat_pos = np.mgrid[0:self.h, 0:self.w][::-1].astype(np.float32).transpose((1, 2, 0)) / theta_gamma
        pos = spat_pos.reshape((-1, ))

        self.gkdt_filter.init(pos)

    def filter(self, value: np.ndarray):
        val1 = np.concatenate([value, np.ones((self.h, self.w, 1), dtype=np.float32)], axis=-1)
        val1 = val1.reshape((-1, ))

        self.gkdt_filter.filter(val1, val1)

        val1 = val1.reshape((self.h, self.w, self.vd))
        out = val1[..., :-1] / val1[..., -1:]

        return out

