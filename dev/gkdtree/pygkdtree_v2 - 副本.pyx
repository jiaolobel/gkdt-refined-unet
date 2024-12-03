# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as cnp

cdef extern from "gkdtree_v2.h":
    void init(const float *pos, int pd, int n, int *splat_indices, float *splat_weights, int *splat_results, int *slice_indices, float *slice_weights, int *slice_results, int *nleaves) except +

    void filter(const float *val, int vd, int n, const int *splat_indices, const float *splat_weights, const int *splat_results, const int *slice_indices, const float *slice_weights, const int *slice_results, int nleaves, float *out) except +


def pyinit(float [:] pos, int pd, int n, int [:] splat_indices, float [:] splat_weights, int [:] splat_results, int [:] slice_indices, float [:] slice_weights, int [:] slice_results):
    cdef int nleaves
    
    init(&pos[0], pd, n, &splat_indices[0], &splat_weights[0], &splat_results[0], &slice_indices[0], &slice_weights[0], &slice_results[0], &nleaves)
    
    return nleaves


def pyfilter(float [:] val, int vd, int n, int [:] splat_indices, float [:] splat_weights, int [:] splat_results, int [:] slice_indices, float [:] slice_weights, int [:] slice_results, int nleaves, float [:] out):
    filter(&val[0], vd, n, &splat_indices[0], &splat_weights[0], &splat_results[0], &slice_indices[0], &slice_weights[0], &slice_results[0], nleaves, &out[0])