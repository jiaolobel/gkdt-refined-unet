import numpy as np
from PIL import Image

import cv2

from gkdtree import gkdtree_filter

im = Image.open("../data/lenna.png").resize((128, 128))
im = np.array(im) / 255.

h, w, n_channels = im.shape

invSpatialStddev = 1. / 5.
invColorStddev = 1. / .25

features = np.zeros((h, w, 5), dtype=np.float32)
spatial_feat = np.mgrid[0:h, 0:w][::-1].transpose((1, 2, 0)) * invSpatialStddev
color_feat = im * invColorStddev
features[..., :2] = spatial_feat
features[..., 2:] = color_feat
features = features.reshape((-1, ))

values = np.ones((h, w, 4), dtype=np.float32)
values[..., :3] = im
values = values.reshape((-1, ))

out = gkdtree_filter(features, 5, values, 4, h * w)
out = out.reshape((h, w, 4))

dst = out[..., :3] / out[..., 3:]
# dst = out[..., :3]

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", dst[..., ::-1])
cv2.waitKey()
