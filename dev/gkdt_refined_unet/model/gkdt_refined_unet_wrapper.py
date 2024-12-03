"""
Wrapper of Refined UNet. 

unet(tile) + dcrf(full)
"""
import sys

import numpy as np
import tensorflow as tf

from .crf_computation import CRFComputation
from .gkdt_refined_unet_config import GKDTRefinedUNetConfig
from .linear2 import Linear2

sys.path.append("..")

from ..util.tile_utils import reconstruct_full

class GKDTRefinedUNet(tf.Module):
    def __init__(
        self,
        config: GKDTRefinedUNetConfig = None,
        ugenerator_path: str = None,
        name: str = None,
    ):
        super().__init__(name)

        self.config = config
        self.crf_computation = CRFComputation(self.config)
        self.unary_generator = tf.saved_model.load(ugenerator_path)

        self.linear2 = Linear2()

        print("Activate modules...")
        
        self.unary_generator(
            tf.ones(
                shape=[1, self.config.height, self.config.width, self.config.num_bands, ], 
                dtype=tf.float32,
            )
        )
        self.linear2(
            tf.ones(
                shape=[self.config.height, self.config.width, self.config.n_channels], 
                dtype=tf.float32
            )
        )
        self.crf_computation.mean_field_approximation(
            tf.ones(
                shape=[self.config.height, self.config.width, self.config.num_classes],
                dtype=tf.float32,
            ),
            tf.ones(
                shape=[self.config.height, self.config.width, self.config.n_channels], 
                dtype=tf.float32
            ),
        )

        print("Modules activated.")

        # # - For code test
        # mf_unary = tf.ones(
        #     shape=[self.config.height, self.config.width, self.config.num_classes],
        #     dtype=tf.float32,
        # )
        # mf_ref = tf.ones(
        #     [self.config.height, self.config.width, self.config.n_channels], 
        #     dtype=tf.float32, 
        # )
        # print("mf_unary.shape: ", mf_unary.shape)
        # print("mf_ref.shape: ", mf_ref.shape)
        # self.crf_computation.mean_field_approximation(
        #     mf_unary, 
        #     mf_ref, 
        # )
        

    def inference(self, test_set: tf.Tensor) -> tf.Tensor:
        """
        Predict and refine.

        Args:
            image: 7-band input, full-resolution, [h, w, c], float32.

        Returns:
            rfn: [h, w],
        """

        utiles = []
        imtiles = []

        # start = time.time()
        i = 0
        for record in test_set.take(-1):
            print("Patch {}...".format(i))
            x = record["x_train"]
            utiles += [self.unary_generator(x).numpy()[0]]
            imtiles += [x.numpy()[0]]
            i += 1

        utiles = np.stack(utiles, axis=0)
        imtiles = np.stack(imtiles, axis=0)
        unary = reconstruct_full(utiles, crop_height=self.config.height, crop_width=self.config.width)
        image = reconstruct_full(imtiles, crop_height=self.config.height, crop_width=self.config.width)

        # unary = self.unary_generator(image[tf.newaxis, ...])[0]  # [H, W, N]
        ref = self.linear2(
            image[..., self.config.channel_start:self.config.channel_end:-1]
        ) # [H, W, C]

        # rfn = self.crf_computation.mean_field_approximation(
        #     unary,
        #     ref,
        # )

        with tf.device("CPU"):
            print("CRF...")
            rfn = self.crf_computation.mean_field_approximation(unary, ref)

        return rfn
