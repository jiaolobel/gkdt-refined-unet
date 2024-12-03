import numpy as np
import tensorflow as tf

# from .pygkdtree import pygkdtree_filter
from .pygkdtree_v2 import PyGKDTFilter

# @tf.numpy_function(Tout=np.float32)
def bilateral_filter(value: tf.Tensor, image: tf.Tensor, theta_alpha, theta_beta):
    # value = value.numpy()
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

    pygkdtree_filter(pos, n_imchannels + 2, val, n_valchannels + 1, h * w, out)

    out = out.reshape((h, w, n_valchannels + 1))
    out = out[..., :-1] / out[..., -1:]

    # out = tf.convert_to_tensor(out, dtype=tf.float32)

    return out
