"""
tf.py_function decorating function
"""


import numpy as np
import tensorflow as tf

# from pygkdtree import pygkdtree_filter
# from pygkdtree_v3 import PyGKDTFilter
from pygkdtree import PyMTGKDTFilter

# from pympler import tracker

# @tf.numpy_function(Tout=tf.float32)
def bilateral_filter(value, image, theta_alpha, theta_beta, nthreads):
    h, w, n_imchannels = image.shape
    _, _, n_valchannels = value.shape

    # tr = tracker.SummaryTracker()
    # tr.print_diff()

    spat_pos = np.mgrid[0:h, 0:w][::-1].astype(np.float32).transpose((1, 2, 0)) / theta_alpha
    colr_pos = image / theta_beta
    pos = np.concatenate([spat_pos, colr_pos], axis=-1)
    pos = pos.reshape((-1, ))

    val1 = np.concatenate([value, np.ones((h, w, 1), dtype=np.float32)], axis=-1)
    val1 = val1.reshape((-1, ))

    # tr.print_diff()

    gkdtfilter = PyMTGKDTFilter(n_imchannels + 2, n_valchannels + 1, h * w, nthreads)
    gkdtfilter.seqinit(pos)
    gkdtfilter.mtcompute(val1, pos, val1)

    val1 = val1.reshape((h, w, n_valchannels + 1))
    
    out = val1[..., :-1] / val1[..., -1:]

    # tr.print_diff()

    # pos, val, gkdtfilter = None, None, None

    # tr.print_diff()

    return out
