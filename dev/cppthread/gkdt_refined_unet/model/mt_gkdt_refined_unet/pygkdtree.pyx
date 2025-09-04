# distutils: language = c++
# cython: language_level = 3
# distutils: sources = gkdtree.cpp

cdef extern from "gkdtree.h":
    cdef cppclass GKDTFilter:
        GKDTFilter(int pd, int n, int nthreads) except +
        void seqinit(const float *pos) except +
        void seqcompute(const float *val, int vd, const float *pos, float *out) except +
        void mtcompute(const float *val, int vd, const float *pos, float *out) except +


cdef class PyGKDTFilter:
    cdef GKDTFilter *filter

    def __cinit__(self, int pd, int n, int nthreads):
        self.filter = new GKDTFilter(pd, n, nthreads)

    def seqinit(self, float[:] pos):
        self.filter.seqinit(&pos[0])

    def seqcompute(self, float[:] val, int vd, float[:] pos, float[:] out):
        self.filter.mtcompute(&val[0], vd, &pos[0], &out[0])

    def mtcompute(self, float[:] val, int vd, float[:] pos, float[:] out):
        self.filter.mtcompute(&val[0], vd, &pos[0], &out[0])

    def __dealloc__(self):
        del self.filter