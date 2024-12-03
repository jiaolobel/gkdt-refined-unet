import numpy as np
from PIL import Image

import cv2

from pygkdtree import pygkdtree_filter

im = Image.open("../data/lenna.png").resize((512, 512))
im = np.array(im, dtype=np.float32) / 255.

h, w, n_imchannels = im.shape
_, _, n_valchannels = im.shape

invSpatialStddev = 1. / 5.
invColorStddev = 1. / .25

feature = np.zeros((h, w, n_imchannels + 2), dtype=np.float32)
spatial_feat = np.mgrid[0:h, 0:w][::-1].transpose((1, 2, 0)) * invSpatialStddev
color_feat = im * invColorStddev
feature[..., :2] = spatial_feat
feature[..., 2:] = color_feat
feature = feature.reshape((-1, ))

# im = im.reshape((-1, ))
value = np.ones((h, w, n_valchannels + 1), dtype=np.float32)
value[..., :n_valchannels] = im
value = value.reshape((-1, ))

# normalization
# ones = np.ones((h * w, ), dtype=np.float32)
# norms = np.ones_like(ones, dtype=np.float32)
# py_gkdtree_filter(features, 5, ones, 1, h * w, norms)
# norms = norms.reshape((h, w, 1))

out = np.zeros_like(value, dtype=np.float32)
pygkdtree_filter(feature, n_imchannels + 2, value, n_valchannels + 1, h * w, out)
# out = out.reshape((h, w, 4))
out = out.reshape((h, w, n_valchannels + 1))

dst = out[..., :-1] / out[..., -1:]
# dst = out[..., :3]
# im = im.reshape((h, w, 3))
# dst = out / norms

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", dst[..., ::-1])
cv2.waitKey()
