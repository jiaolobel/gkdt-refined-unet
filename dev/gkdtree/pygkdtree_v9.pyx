# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as cnp

cdef extern from "gkdtree_v9.h":
    cdef cppclass GKDTFilter:
        GKDTFilter(int pd, int vd, int n) except +
        void init(const float *pos) except +
        void filter(const float *val, float *out) except +


cdef class PyGKDTFilter:
    cdef GKDTFilter *gkdtfilter

    def __cinit__(self, int pd, int vd, int n):
        self.gkdtfilter = new GKDTFilter(pd, vd, n)

    def init(self, float[:] pos):
        self.gkdtfilter.init(&pos[0])

    def filter(self, float[:] val, float[:] out):
        self.gkdtfilter.filter(&val[0], &out[0])

    def __dealloc__(self):
        del self.gkdtfilter
