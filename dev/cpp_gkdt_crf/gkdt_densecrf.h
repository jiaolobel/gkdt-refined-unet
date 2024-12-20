#include <iostream>
#include "Eigen/Dense"

#include "gkdtree_v10.h"

using namespace std;
using namespace Eigen;

class DenseCRF
{
protected:
    int H_, W_, N_, n_classes_, d_bifeats_, d_spfeats_, d_val_;

    float theta_alpha_, theta_beta_, theta_gamma_;

    float bilateral_compat_, spatial_compat_;

    int n_iterations_;

    GKDTFilter *bilateral_step_, *spatial_step_;

    // MatrixXf compatibility_matrix_;
    const int compatibility_ = -1;

public:
    DenseCRF(
        int H,
        int W,
        int n_classes,
        int d_bifeats,
        int d_spfeats,
        float theta_alpha,
        float theta_beta,
        float theta_gamma,
        float bilateral_compat,
        float spatial_compat,
        int n_iterations
    )
    {
        H_ = H;
        W_ = W;
        N_ = H_ * W_;
        n_classes_ = n_classes;
        d_bifeats_ = d_bifeats;
        d_spfeats_ = d_spfeats;
        d_val_ = n_classes_ + 1;

        theta_alpha_ = theta_alpha;
        theta_beta_ = theta_beta;
        theta_gamma_ = theta_gamma;

        bilateral_compat_ = bilateral_compat;
        spatial_compat_ = spatial_compat;

        n_iterations_ = n_iterations;

        bilateral_step_ = new GKDTFilter(d_bifeats, d_val_, N_);
        spatial_step_ = new GKDTFilter(d_spfeats, d_val_, N_);

        // potts_compatibility(compatibility_matrix_); // Potts model
    }

    ~DenseCRF()
    {
        delete bilateral_step_;
        delete spatial_step_;
    }

    // void potts_compatibility(MatrixXf &labelcompatibility)
    // {
    //     labelcompatibility.resize(n_classes_, n_classes_);
    //     labelcompatibility.setIdentity();
    //     labelcompatibility *= -1;
    // }

    void softmax(const MatrixXf &in, MatrixXf &out)
    {
        // in and out share the shape of [d, N], channel-first
        out = (in.rowwise() - in.colwise().maxCoeff()).array().exp();
        RowVectorXf sm = out.colwise().sum();
        out.array().rowwise() /= sm.array();
    }

    void inference(
        const float *unary1, // n_channels + 1 or n_classes + 1
        const float *ref,
        float *out1
    )
    {
        // Create bilateral and spatial features
        float *bilateral_feats = new float[N_ * d_bifeats_];
        float *spatial_feats = new float[N_ * d_spfeats_];

        for (int y = 0; y < H_; y++)
        {
            for (int x = 0; x < W_; x++)
            {
                bilateral_feats[y * W_ * d_bifeats_ + x * d_bifeats_ + 0] = (float)x / theta_alpha_;
                bilateral_feats[y * W_ * d_bifeats_ + x * d_bifeats_ + 1] = (float)y / theta_alpha_;
                for (int d = d_spfeats_; d < d_bifeats_; d++)
                {
                    bilateral_feats[y * W_ * d_bifeats_ + x * d_bifeats_ + d] = ref[y * W_ * (d_bifeats_ - d_spfeats_) + x * (d_bifeats_ - d_spfeats_) + (d - d_spfeats_)] / theta_beta_;
                }

                spatial_feats[y * W_ * d_spfeats_ + x * d_spfeats_ + 0] = (float)x / theta_gamma_;
                spatial_feats[y * W_ * d_spfeats_ + x * d_spfeats_ + 1] = (float)y / theta_gamma_;
            }
        }

        // Initialize bilateral and spatial filters
        bilateral_step_->init(bilateral_feats);
        spatial_step_->init(spatial_feats);
        printf("Filters initialized.\n");

        // Free features
        delete[] bilateral_feats;
        delete[] spatial_feats;

        // Compute symmetric normalizations
        // MatrixXf all_ones(1, N_), tmp(1, N_), bilateral_norm_vals(1, N_), spatial_norm_vals(1, N_);
        // all_ones.setOnes();

        // tmp.setZero();
        // bilateral_filter_->compute(all_ones, false, tmp);
        // bilateral_norm_vals = (tmp.array().pow(.5f) + 1e-20).inverse();

        // tmp.setZero();
        // spatial_filter_->compute(all_ones, false, tmp);
        // spatial_norm_vals = (tmp.array().pow(.5f) + 1e-20).inverse();

        // Transpose unary
        MatrixXf unary1_mat = Map<const MatrixXf>(unary1, d_val_, N_); // [vd, N]

        // Initialize Q
        MatrixXf Q(d_val_, N_);
        softmax(-unary1_mat, Q);

        MatrixXf tmp1(d_val_, N_);
        MatrixXf bilateral_out(d_val_, N_), spatial_out(d_val_, N_);
        MatrixXf message_passing(d_val_, N_);
        MatrixXf pairwise(d_val_, N_);

        for (int i = 0; i < n_iterations_; i++)
        {
            printf("Iteration %d / %d...\n", i + 1, n_iterations_);

            tmp1 = -unary1_mat; // [n_classes, N]

            // Bilateral message passing and symmetric normalization
            // bilateral_step_->compute(Q.array().rowwise(), bilateral_out); // [n_classes, N] why?
            bilateral_step_->compute(Q, bilateral_out);

            // Spatial message passing and symmetric normalization
            // spatial_step_->compute(Q.array().rowwise(), spatial_out); // [n_classes, N] why?
            spatial_step_->compute(Q, spatial_out);

            // Message passing
            message_passing.noalias() = bilateral_compat_ * bilateral_out + spatial_compat_ * spatial_out; // [n_classes, N]

            // Compatibility transformation
            pairwise.noalias() = compatibility_ * message_passing; // [n_classes, N]

            // Local update
            tmp1 -= pairwise; // [n_classes, N]

            // Normalize
            softmax(tmp1, Q); // [n_classes, N]
        }

        // Return
        Map<MatrixXf> out_mat(out1, d_val_, N_);
        out_mat = Q;
    }
};

