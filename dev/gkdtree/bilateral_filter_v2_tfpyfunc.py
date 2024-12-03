"""
tf.py_function decorating function
"""


import numpy as np
import tensorflow as tf

# from pygkdtree import pygkdtree_filter
from pygkdtree_v2 import PyGKDTFilter

# @tf.numpy_function(Tout=tf.float32)
def bilateral_filter(value, image, theta_alpha, theta_beta):
    h, w, n_imchannels = image.shape
    _, _, n_valchannels = value.shape

    pos = np.zeros((h, w, n_imchannels + 2), dtype=np.float32)
    spat_pos = np.mgrid[0:h, 0:w][::-1].transpose((1, 2, 0)) / theta_alpha
    color_pos = image / theta_beta
    pos[..., :2] = spat_pos
    pos[..., 2:] = color_pos
    pos = pos.reshape((-1, ))

    val = np.ones((h, w, n_valchannels + 1), dtype=np.float32)
    val[..., :n_valchannels] = value
    val = val.reshape((-1, ))

    out = np.zeros_like(val, dtype=np.float32)

    # splat_indices = np.zeros((h * w * 64, ), dtype=np.int32)
    # splat_weights = np.zeros((h * w * 64, ), dtype=np.float32)
    # splat_results = np.zeros((h * w, ), dtype=np.int32)
    # slice_indices = np.zeros((h * w * 64, ), dtype=np.int32)
    # slice_weights = np.zeros((h * w * 64, ), dtype=np.float32)
    # slice_results = np.zeros((h * w, ), dtype=np.int32)

    # nleaves = pygkdtree_v2.pyinit(pos, n_imchannels + 2, h * w, splat_indices, splat_weights, splat_results, slice_indices, slice_weights, slice_results)

    # pygkdtree_v2.pyfilter(val, n_valchannels + 1, h * w, splat_indices, splat_weights, splat_results, slice_indices, slice_weights, slice_results, nleaves, out)

    # pygkdtree_filter(pos, n_imchannels + 2, val, n_valchannels + 1, h * w, out)

    gkdtfilter = PyGKDTFilter(pos, n_imchannels + 2, h * w)
    gkdtfilter.filter(val, n_valchannels + 1, h * w, out)

    out = out.reshape((h, w, n_valchannels + 1))
    out = out[..., :-1] / out[..., -1:]

    return out
