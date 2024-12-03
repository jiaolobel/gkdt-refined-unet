# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as cnp

cdef extern from "gkdtree_v2.h":
    cdef cppclass GKDTFilter:
        GKDTFilter(const float *pos, int pd, int n) except +

        void filter(const float *val, int vd, int n, float *out) except +


cdef class PyGKDTFilter:
    cdef GKDTFilter *gkdtfilter

    def __cinit__(self, float[:] pos, int pd, int n):
        self.gkdtfilter = new GKDTFilter(&pos[0], pd, n)

    def filter(self, float[:] val, int vd, int n, float[:] out):
        self.gkdtfilter.filter(&val[0], vd, n, &out[0])

    def __dealloc__(self):
        del self.gkdtfilter