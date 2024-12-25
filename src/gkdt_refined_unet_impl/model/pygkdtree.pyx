# distutils: language = c++
# cython: language_level = 3

cdef extern from "gkdtree.h":
    cdef cppclass GKDTFilter:
        GKDTFilter(int pd, int vd, int n) except +
        void init(const float *pos) except +
        void compute(const float *val, float *out) except +


cdef class PyGKDTFilter:
    cdef GKDTFilter *gkdtfilter

    def __cinit__(self, int pd, int vd, int n):
        self.gkdtfilter = new GKDTFilter(pd, vd, n)

    def init(self, float[:] pos):
        self.gkdtfilter.init(&pos[0])

    def compute(self, float[:] val, float[:] out):
        self.gkdtfilter.compute(&val[0], &out[0])

    def __dealloc__(self):
        del self.gkdtfilter
