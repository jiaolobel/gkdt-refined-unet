import numpy as np
from PIL import Image

import tensorflow as tf

import cv2

# from pygkdtree_v3 import PyGKDTFilter


from bilateral_filter_v9 import bilateral_filter

# from pympler import tracker

im = Image.open("../../data/lenna.png").resize((512, 512))
im = np.array(im, dtype=np.float32) / 255.

# out = tf.numpy_function(bilateral_filter, inp=[im, im, 5., .25], Tout=tf.float32)
# out = out.numpy()

out = bilateral_filter(im, im, 5., .25)

# tr = tracker.SummaryTracker()
# tr.print_diff()

# out = np.zeros_like(im, dtype=np.float32)
# bilateral_filter(im, im, 5., .25, out)

# tr.print_diff()

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", out[..., ::-1])
cv2.waitKey()
