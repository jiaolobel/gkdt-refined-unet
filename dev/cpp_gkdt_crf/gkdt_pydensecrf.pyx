# distutils: language = c++
# cython: language_level = 3

cdef extern from "gkdt_densecrf.h":
    cdef cppclass DenseCRF:
        DenseCRF(
            int H, int W, int n_classes,
            int d_bifeats, int d_spfeats,
            float theta_alpha, float theta_beta,
            float theta_gamma,
            float bilateral_compat, float spatial_compat,
            int n_iterations    
        ) except +

        void inference(
            const float *unary1, const float *ref,
            float *out1
        ) except +


cdef class PyDenseCRF:
    cdef DenseCRF *dcrf

    def __cinit__(self, 
        int H, int W, int n_classes,
        int d_bifeats, int d_spfeats,
        float theta_alpha, float theta_beta,
        float theta_gamma,
        float bilateral_compat, float spatial_compat,
        int n_iterations    
    ):
        self.dcrf = new DenseCRF(
            H, W, n_classes,
            d_bifeats, d_spfeats, 
            theta_alpha, theta_beta, 
            theta_gamma, 
            bilateral_compat, spatial_compat, 
            n_iterations
        )

    def inference(self, float[:] unary1, float[:] ref, float[:] out1):
        self.dcrf.inference(&unary1[0], &ref[0], &out1[0])

    def __dealloc__(self):
        del self.dcrf