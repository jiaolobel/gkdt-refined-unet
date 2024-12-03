import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util.tfrecordloader_l8 import load_testset
from util.tile_utils import reconstruct_full
from util.linear2 import Linear2

test_name = "../../data/l8/testcase/LC08_L1TP_113026_20160412_20170326_01_T1_train.tfrecords"
unary_generator = tf.saved_model.load("parameter/saved_model/unary_generator")
channel_start, channel_end = 4, 1 # channel_start included, channel_end excluded, [4, 3, 2] is commonly used as false-color image. 
crop_height, crop_width = 512, 512


test_set = load_testset([test_name], batch_size=1)

utiles, imtiles = [], []

for record in test_set.take(-1):
    x = record["x_train"]
    utiles += [unary_generator(x)[0].numpy()]
    imtiles += [x[0, ..., channel_start:channel_end:-1].numpy()]

utiles = np.stack(utiles, axis=0)
imtiles = np.stack(imtiles, axis=0)
# unary = reconstruct_full(utiles, crop_height=crop_height, crop_width=crop_width)
# image = reconstruct_full(imtiles, crop_height=crop_height, crop_width=crop_width)

reftiles = np.zeros_like(imtiles, dtype=imtiles.dtype)
linear2 = Linear2()

for i, image in enumerate(imtiles):
    reftiles[i] = linear2(image)

# for ref in reftiles:
#     plt.imshow(ref)
#     plt.show()

np.savez(test_name.replace("train.tfrecords", "utiles.npz"), utiles)
np.savez(test_name.replace("train.tfrecords", "reftiles.npz"), reftiles)
# plt.imsave(test_name.replace("train.tfrecords", "unary.png"), unary)
# plt.imsave(test_name.replace("train.tfrecords", "ref.png"), ref)