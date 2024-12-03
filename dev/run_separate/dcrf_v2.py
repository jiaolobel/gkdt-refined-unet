"""
gkdt filter v2 included.
full-resolution input included. 
"""

import numpy as np
import tensorflow as tf

from util.bilateral_step_factory_v2 import BilateralStep
from util.spatial_step_factory_v2 import SpatialStep

import matplotlib.pyplot as plt

unary = np.load("input/gkdt_dcrf/LC08_L1TP_113026_20160412_20170326_01_T1_unary.npz")["arr_0"]
# im = np.load("input/gkdt_dcrf/LC08_L1TP_113026_20160412_20170326_01_T1_im.npz")["arr_0"]
ref = np.load("input/gkdt_dcrf/LC08_L1TP_113026_20160412_20170326_01_T1_ref.npz")["arr_0"]

print(unary.shape, ref.shape)

# resize
sampled = 2
unary = unary[::sampled, ::sampled]
ref = ref[::sampled, ::sampled]

print(unary.shape, ref.shape)

plt.imshow(ref)
plt.show()

num_iterations = 10
theta_alpha = 80.0
theta_beta = 0.03125
theta_gamma = 3.0
bilateral_compat: float = 10.0
spatial_compat: float = 3.0
compatibility: float = -1.0

# plt.imshow()

def softmax(x, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

# def softmax(x, axis=None):
#     return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True) + 1e-5)

def apply_mask(image, mask, mask_value, color, alpha=.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == mask_value, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])

    return image

cloud, cloud_color = 3, [115, 223, 255]
shadow, shadow_color = 2, [38, 115, 0]



bilateral_step = BilateralStep()
spatial_step = SpatialStep()
print("initializing bilateral_step")
bilateral_step.init(ref, theta_alpha=theta_alpha, theta_beta=theta_beta)
print("bilateral_step initialized")
spatial_step.init(ref, theta_gamma=theta_gamma)

Q = softmax(-unary, axis=-1)

for i in range(num_iterations):
    print("iteration: ", i + 1)
    tmp1 = -unary

    bilateral_out = bilateral_step.filter(Q)
    spatial_out = spatial_step.filter(Q)

    message_passing = bilateral_compat * bilateral_out + spatial_compat * spatial_out

    pairwise = compatibility * message_passing

    tmp1 -= pairwise

    Q = softmax(tmp1, axis=-1)

MAP = Q.argmax(axis=-1)

ref_uint8 = np.uint8(ref * 255)
labeled_image = apply_mask(ref_uint8, MAP, cloud, cloud_color)
labeled_image = apply_mask(labeled_image, MAP, shadow, shadow_color)

plt.imshow(labeled_image)
plt.show()

# plt.imsave("output/gkdt_dcrf/LC08_L1TP_113026_20160412_20170326_01_T1_rfn.png", labeled_image)