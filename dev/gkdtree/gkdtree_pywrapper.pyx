# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as cnp

cdef extern from "gkdtree.h":
    cdef cppclass GKDTree:
        @staticmethod
        void filter(const float *pos, int pd, const float *val, int vd, int n, float *out) except +


def py_gkdtree_filter(float [:] pos, int pd, float [:] val, int vd, int n, float [:] out):
    GKDTree.filter(&pos[0], pd, &val[0], vd, n, &out[0])