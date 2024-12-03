import numpy as np
from PIL import Image

import tensorflow as tf

import cv2

from bilateral_filter_v2_tfpyfunc import bilateral_filter

im = Image.open("../../data/lenna.png").resize((1024, 1024))
im = np.array(im, dtype=np.float32) / 255.

out = tf.numpy_function(bilateral_filter, inp=[im, im, 5., .25], Tout=tf.float32)
# out = bilateral_filter(im, im, 5., .25)
out = out.numpy()

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", out[..., ::-1])
cv2.waitKey()
