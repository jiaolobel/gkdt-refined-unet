# distutils: language = c++
# cython: language_level = 3
# distutils: sources = gkdtree.cpp

cdef extern from "gkdtree.h":
    cdef cppclass MTGKDTFilter:
        MTGKDTFilter(int pd, int vd, int n, int nthreads) except +
        void seqinit(const float *pos) except +
        void mtcompute(const float *val, const float *pos, float *out) except +


cdef class PyMTGKDTFilter:
    cdef MTGKDTFilter *filter

    def __cinit__(self, int pd, int vd, int n, int nthreads):
        self.filter = new MTGKDTFilter(pd, vd, n, nthreads)

    def seqinit(self, float[:] pos):
        self.filter.seqinit(&pos[0])

    def mtcompute(self, float[:] val, float[:] pos, float[:] out):
        self.filter.mtcompute(&val[0], &pos[0], &out[0])

    def __dealloc__(self):
        del self.filter