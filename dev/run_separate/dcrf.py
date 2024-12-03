import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util.bilateral_step_factory_v2 import BilateralStep
from util.spatial_step_factory_v2 import SpatialStep

utiles = np.load("output/unary_generator/LC08_L1TP_113026_20160412_20170326_01_T1_utiles.npz")["arr_0"]
reftiles = np.load("output/unary_generator/LC08_L1TP_113026_20160412_20170326_01_T1_reftiles.npz")["arr_0"]
num_iterations = 10
theta_alpha = 80.0
theta_beta = 0.03125
theta_gamma = 3.0
bilateral_compat: float = 10.0
spatial_compat: float = 3.0
compatibility: float = -1.0

# plt.imshow(unary.max(axis=-1))
# plt.show()
# plt.imshow(ref) # data error!!!
# plt.show()

# print(unary.shape, ref.shape)

idxs = [20, 37, 52, 172]

def apply_mask(image, mask, mask_value, color, alpha=.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == mask_value, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])

    return image

cloud, cloud_color = 3, [115, 223, 255]
shadow, shadow_color = 2, [38, 115, 0]

for idx in idxs:
    utile, reftile = utiles[idx], reftiles[idx]

    with tf.device("CPU"):

        bilateral_step = BilateralStep()
        spatial_step = SpatialStep()

        bilateral_step.init(reftile, theta_alpha, theta_beta)
        spatial_step.init(reftile, theta_gamma)

        Q = tf.nn.softmax(-utile, axis=-1).numpy()

        for i in range(num_iterations):
            print("iteration: ", i + 1)
            tmp1 = -utile

            bilateral_out = bilateral_step.filter(Q)
            spatial_out = spatial_step.filter(Q)

            message_passing = bilateral_compat * bilateral_out + spatial_compat * spatial_out

            pairwise = compatibility * message_passing

            tmp1 -= pairwise

            Q = tf.nn.softmax(tmp1).numpy()

        MAP = tf.math.argmax(Q, axis=-1).numpy()

    reftile = np.uint8(reftile * 255)

    labeled_image = apply_mask(reftile, MAP, cloud, cloud_color)
    labeled_image = apply_mask(labeled_image, MAP, shadow, shadow_color)

    plt.imshow(labeled_image)
    plt.show()