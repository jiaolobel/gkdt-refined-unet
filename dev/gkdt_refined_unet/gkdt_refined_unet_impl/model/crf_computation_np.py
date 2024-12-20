import numpy as np

# from .bilateral_step_factory_v3 import BilateralStep
# from .spatial_step_factory_v3 import SpatialStep

from .bilateral_step_factory_v9 import BilateralStep
from .spatial_step_factory_v9 import SpatialStep

# from ..config.inference_config import InferenceConfig

# from .gkdt_refined_unet_config import GKDTRefinedUNetConfig

class CRFComputation:
    def __init__(self, config):
        self.config = config
    
    def mean_field_approximation(
        self,
        unary: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Work in NumPy
        """
        def softmax(x, axis=None):
            x_max = np.max(x, axis=axis, keepdims=True)
            exp_x_shifted = np.exp(x - x_max)
            return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

        # -> Initialize bilateral step and spatial step
        bilateral_step = BilateralStep(
            height=self.config.tile_height, 
            width=self.config.tile_width, 
            n_refchannels=self.config.n_channels, 
            n_valchannels=self.config.num_classes
        )
        spatial_step = SpatialStep(
            height=self.config.tile_height, 
            width=self.config.tile_width, 
            n_valchannels=self.config.num_classes
        )
        bilateral_step.init(
            reference, 
            theta_alpha=self.config.theta_alpha, 
            theta_beta=self.config.theta_beta
        )
        spatial_step.init(
            theta_gamma=self.config.theta_gamma
        )

        # - Initialize Q
        Q = softmax(-unary, axis=-1)  # [H, W, N]

        for i in range(self.config.num_iterations):
            tmp1 = -unary  # [H, W, N]

            # - Symmetric normalization and bilateral message passing
            bilateral_out = bilateral_step.filter(Q)

            # - Symmetric normalization and spatial message passing
            spatial_out = spatial_step.filter(Q)

            # - Message passing
            message_passing = (
                self.config.bilateral_compat * bilateral_out + self.config.spatial_compat * spatial_out
            )  # [H, W, C]

            # - Compatibility transform
            pairwise = self.config.compatibility * message_passing  # [N, C]

            # - Local update
            tmp1 -= pairwise  # [N, C]

            # - Normalize
            Q = softmax(tmp1, axis=-1)  # [N, C]

        # - Maximum posterior estimation
        MAP = Q.argmax(axis=-1)  # [H, W]

        return MAP
    
