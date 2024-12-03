import numpy as np
from PIL import Image

import cv2

from spatial_filter_factory_v2 import spatial_filter

im = Image.open("../../data/lenna.png").resize((512, 512))
im = np.array(im, dtype=np.float32) / 255.

out = spatial_filter(im, 5.)

cv2.imshow("im", im[..., ::-1])
cv2.imshow("dst", out[..., ::-1])
cv2.waitKey()
