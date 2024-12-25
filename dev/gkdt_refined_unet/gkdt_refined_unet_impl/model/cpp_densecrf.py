import numpy as np

from .gkdt_pydensecrf import PyDenseCRF

from ..config.global_config import GlobalConfig

from .dcrf_base import DenseCRFBase

class DenseCRF(DenseCRFBase):
    def __init__(self, config:GlobalConfig):
        self.config = config
        self.dcrf = PyDenseCRF(
            H=config.tile_height, W=config.tile_width, n_classes=config.num_classes, 
            d_bifeats=config.n_channels + 2, d_spfeats=2, 
            theta_alpha=config.theta_alpha, theta_beta=config.theta_beta, 
            theta_gamma=config.theta_gamma, 
            bilateral_compat=config.bilateral_compat, spatial_compat=config.spatial_compat, 
            n_iterations=config.num_iterations, 
        )

    def mean_field_approximation(self, unary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        norm = np.ones(
            (self.config.tile_height, self.config.tile_width, 1), 
            dtype=np.float32
        )
        unary1 = np.concatenate([unary, norm], axis=-1)
        unary1 = unary1.reshape((-1, ))
        reference = reference.reshape((-1, ))
        Q1 = unary1
        
        self.dcrf.inference(unary1, reference, Q1)

        Q1 = Q1.reshape((self.config.tile_height, self.config.tile_width, self.config.num_classes + 1))
        Q = Q1[..., :-1] / (Q1[..., -1:] + 1e-8)
        MAP = Q.argmax(axis=-1)

        return MAP