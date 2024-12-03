import numpy as np
import tensorflow as tf

from pygkdtree_v2 import PyGKDTFilter

# @tf.numpy_function(Tout=tf.float32)
def spatial_filter(value, theta_gamma):
    h, w, n_valchannels = value.shape

    pos = np.zeros((h, w, 2), dtype=np.float32)
    spat_pos = np.mgrid[0:h, 0:w][::-1].transpose((1, 2, 0)) / theta_gamma
    pos[..., :2] = spat_pos
    pos = pos.reshape((-1, ))

    val = np.ones((h, w, n_valchannels + 1), dtype=np.float32)
    val[..., :n_valchannels] = value
    val = val.reshape((-1, ))

    out = np.zeros_like(val, dtype=np.float32)

    # pygkdtree_filter(pos, 2, val, n_valchannels + 1, h * w, out)
    gkdt_filter = PyGKDTFilter(pos, 2, h * w)
    gkdt_filter.filter(val, n_valchannels + 1, h * w, out)

    out = out.reshape((h, w, n_valchannels + 1))
    out = out[..., :-1] / out[..., -1:]

    return out
