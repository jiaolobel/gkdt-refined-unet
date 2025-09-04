#include "gkdtree.h"

const float INF = std::numeric_limits<float>::infinity();

// Random floating-point number between zero and one
float randFloat() { return rand() / (RAND_MAX + 1.0f); }

// A quartic approximation to the integral of a Gaussian of variance 1/2
inline float gCDF(float x) {
    x *= 0.81649658092772592f;
    if (x < -2) {
        return 0;
    } else if (x < -1) {
        x += 2;
        x *= x;
        x *= x;
        return x;
    } else if (x < 0) {
        return 12 + x * (16 - x * x * (8 + 3 * x));
    } else if (x < 1) {
        return 12 + x * (16 - x * x * (8 - 3 * x));
    } else if (x < 2) {
        x -= 2;
        x *= x;
        x *= x;
        return 24 - x;
    } else {
        return 24;
    }
}

// GKDTree constructor
// kd: the dimensionality of the position vectors
// pos: an array of pointers to the position vectors
// n: the number of position vectors
// sBound: the maximum allowable side length of a leaf node
GKDTree::GKDTree(int kd, const float **pos, int n, float sBound)
    : dimensions(kd), sizeBound(sBound), leaves(0) {

    // Recursively build the tree
    root = build(pos, n);

    // Recursively compute the bounds of each node
    vector<float> kdtreeMins(dimensions, -INF);
    vector<float> kdtreeMaxs(dimensions, +INF);
    root->computeBounds(&kdtreeMins[0], &kdtreeMaxs[0]);
}

// Destructor. Recursively deletes the tree.
GKDTree::~GKDTree() { delete root; }

int GKDTree::nLeaves() { return leaves; }

// Query the kdtree. Returns the number of leaf nodes found.
// query: the position around which to search
// ids: the ids of the leaf nodes found
// weights: the weight for each leaf node found
// nSamples: how many query samples to use
int GKDTree::lookup(const float *query, int *ids, float *weights,
                    int nSamples) {
    return root->lookup(query, ids, weights, nSamples, 1);
}

GKDTree::Split::~Split() {
    delete left;
    delete right;
}

// For a Gaussian centered at the given value, truncated to
// within this leaf, what is the fraction of the Gaussian on
// the left of this cut value. This gives the probability of
// splitting left at this node.
float GKDTree::Split::pLeft(float value) {
    float val = gCDF(cutVal - value);
    float minBound = gCDF(minVal - value);
    float maxBound = gCDF(maxVal - value);
    return (val - minBound) / (maxBound - minBound);
}

int GKDTree::Split::lookup(const float *query, int *ids, float *weights,
                           int nSamples, float p) {
    // Compute the probability of a sample splitting left
    float val = pLeft(query[cutDim]);

    // Common-case optimization for a single sample
    if (nSamples == 1) {
        if (randFloat() < val) {
            // printf("go deeper in the left");
            return left->lookup(query, ids, weights, 1, p * val);
        } else {
            // printf("go deeper in the right");
            return right->lookup(query, ids, weights, 1, p * (1 - val));
        }
    }

    // Send some samples to the left of the split
    int leftSamples = (int)(floorf(val * nSamples));

    // Send some samples to the right of the split
    int rightSamples = (int)(floorf((1 - val) * nSamples));

    // There's probably one sample left over by the
    // rounding. Probabilistically assign it to the left or
    // right.
    if (leftSamples + rightSamples != nSamples) {
        float fval = val * nSamples - leftSamples;
        if (randFloat() < fval)
            leftSamples++;
        else
            rightSamples++;
    }

    int samplesFound = 0;

    // Descend the left subtree.
    if (leftSamples > 0) {
        // printf("go deeper in the left");
        samplesFound += left->lookup(query, ids, weights, leftSamples, p * val);
    }

    // Descend the right subtree.
    if (rightSamples > 0) {
        // printf("go deeper in the right");
        samplesFound +=
            right->lookup(query, ids + samplesFound, weights + samplesFound,
                          rightSamples, p * (1 - val));
    }

    return samplesFound;
}

// Recursively compute the bounds of each cell in the
// dimension that that cell cuts along.
void GKDTree::Split::computeBounds(float *mins, float *maxs) {
    minVal = mins[cutDim];
    maxVal = maxs[cutDim];

    maxs[cutDim] = cutVal;
    left->computeBounds(mins, maxs);
    maxs[cutDim] = maxVal;

    mins[cutDim] = cutVal;
    right->computeBounds(mins, maxs);
    mins[cutDim] = minVal;
}

GKDTree::Leaf::~Leaf() {}

int GKDTree::Leaf::lookup(const float *query, int *ids, float *weights,
                          int nSamples, float p) {
    // p is the probability with which one sample arrived
    // here. Calculate the correct probability, q, by
    // evaluating the Gaussian.

    // printf("reach a leaf\n");

    float q = 0;
    for (size_t i = 0; i < position.size(); i++) {
        float delta = query[i] - position[i];
        q += delta * delta;
    }
    q = expf(-q);

    // Weight each sample by the ratio of the correct
    // probability to the actual probability.
    *weights = nSamples * q / p;
    *ids = id;

    return 1;
}

// Construct a kd-tree node from an array of position vectors
GKDTree::Node *GKDTree::build(const float **pos, int n) {
    // Compute a bounding box
    vector<float> mins(dimensions, +INF), maxs(dimensions, -INF);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dimensions; j++) {
            if (pos[i][j] < mins[j])
                mins[j] = pos[i][j];
            if (pos[i][j] > maxs[j])
                maxs[j] = pos[i][j];
        }
    }

    // Find the longest dimension
    int longest = 0;
    for (int i = 1; i < dimensions; i++) {
        if (maxs[i] - mins[i] > maxs[longest] - mins[longest]) {
            longest = i;
        }
    }

    if (maxs[longest] - mins[longest] > sizeBound) {
        // If it's large enough, cut in that dimension and make a
        // split node
        Split *node = new Split;
        node->cutDim = longest;
        node->cutVal = (maxs[longest] + mins[longest]) / 2;

        // resort the input over the split
        int pivot = 0;
        for (int i = 0; i < n; i++) {
            // The next value is larger than the pivot
            if (pos[i][longest] >= node->cutVal)
                continue;

            // We haven't seen anything larger that the pivot yet
            if (i == pivot) {
                pivot++;
                continue;
            }

            // The current value is smaller that the pivot
            swap(pos[i], pos[pivot]);
            pivot++;
        }

        // Build the two subtrees
        node->left = build(pos, pivot);
        node->right = build(pos + pivot, n - pivot);
        return node;
    } else {
        // Make a leaf node with a sample in the center of the
        // bounding box
        Leaf *node = new Leaf;
        node->id = leaves++;
        node->position.resize(dimensions);
        for (int i = 0; i < dimensions; i++) {
            node->position[i] = (mins[i] + maxs[i]) / 2;
        }
        return node;
    }
}

GKDTFilter::GKDTFilter(int pd, int vd, int n) {
    this->pd = pd;
    this->vd = vd;
    this->n = n;

    splat_indices = new int[n * 4];
    splat_weights = new float[n * 4];
    splat_results = new int[n];

    slice_indices = new int[n * 64];
    slice_weights = new float[n * 64];
    slice_results = new int[n];
}
GKDTFilter::~GKDTFilter() {
    delete[] splat_indices;
    delete[] splat_weights;
    delete[] splat_results;
    delete[] slice_indices;
    delete[] slice_weights;
    delete[] slice_results;
}

void GKDTFilter::init(const float *pos) {
    vector<const float *> points(n);
    for (int i = 0; i < n; i++) {
        points[i] = pos + i * pd;
    }

    GKDTree tree(pd, &points[0], points.size(), sqrt(2.0f));

    for (int i = 0; i < n; i++) {
        *(splat_results + i) = tree.lookup(pos + i * pd, splat_indices + i * 4,
                                           splat_weights + i * 4, 4);

        if (*(splat_results + i) > 4) {
            printf(" GT 4!\n");
        }

        *(slice_results + i) = tree.lookup(pos + i * pd, slice_indices + i * 64,
                                           slice_weights + i * 64, 64);
    }

    nleaves = tree.nLeaves();
}

void GKDTFilter::compute(const float *val, float *out) {
    vector<float> leafValues(nleaves * vd, 0.0f);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < splat_results[i]; j++) {
            for (int k = 0; k < vd; k++) {
                leafValues[splat_indices[i * 4 + j] * vd + k] +=
                    val[i * vd + k] * splat_weights[i * 4 + j];
            }
        }
    }

    memset(out, 0, sizeof(float) * n * vd);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < slice_results[i]; j++) {
            for (int k = 0; k < vd; k++) {
                out[i * vd + k] +=
                    leafValues[slice_indices[i * 64 + j] * vd + k] *
                    slice_weights[i * 64 + j];
            }
        }
    }
}

MTGKDTFilter::MTGKDTFilter(int pd, int vd, int n, int nthreads) {
    pd_ = pd;
    vd_ = vd;
    n_ = n;

    nthreads_ = nthreads;
    if (n_ % nthreads_ != 0) {
        printf("Wrong number of threads, will compute sequentially.\n");
        nthreads_ = 1;
    }

    tn_ = n_ / nthreads_;
}

MTGKDTFilter::~MTGKDTFilter() {
    if (tree_ != NULL) {
        delete tree_;
    }
}

void MTGKDTFilter::seqinit(const float *pos) {
    // Make an array of pointer to each position vector. We'll
    // reshuffle this array while building the tree.
    vector<const float *> points(n_);
    for (int i = 0; i < n_; i++) {
        points[i] = pos + i * pd_;
    }

    // Build a tree. The last argument is the maximum side length
    // of a cell. We set it to twice the standard deviation of
    // splatting and slicing.
    tree_ = new GKDTree(pd_, &points[0], points.size(), sqrt(2.0f));
}

void MTGKDTFilter::mtcompute(const float *val, const float *pos, float *out) {
    int inleaf = 4, outleaf = 64;

    // Arrays to use while splatting and slicing
    int *indices = new int[nthreads_ * outleaf];
    float *weights = new float[nthreads_ * outleaf];
    // The values stored at the leaves
    float *leafValues = new float[tree_->nLeaves() * vd_];
    float *mtleafValues = new float[nthreads_ * tree_->nLeaves() * vd_];
    memset(leafValues, 0, sizeof(float) * tree_->nLeaves() * vd_);
    memset(mtleafValues, 0, sizeof(float) * nthreads_ * tree_->nLeaves() * vd_);

    auto tsplat = [&](int threadi) {
        int *tindices = indices + threadi * inleaf;
        float *tweights = weights + threadi * inleaf;
        float *tleafValues = mtleafValues + threadi * tree_->nLeaves() * vd_;
        const float *tpos = pos + threadi * tn_ * pd_;
        const float *tval = val + threadi * tn_ * vd_;

        // Splatting: For each position vector ...
        for (int i = 0; i < tn_; i++) {
            // find up to 4 leaves nearby ...
            int results =
                tree_->lookup(tpos + i * pd_, tindices, tweights, inleaf);
            // and scatter to them.
            for (int j = 0; j < results; j++) {
                for (int k = 0; k < vd_; k++) {
                    tleafValues[tindices[j] * vd_ + k] +=
                        tval[i * vd_ + k] * tweights[j];
                }
            }
        }
    };

    std::vector<std::thread> splatThreads;
    for (int i = 0; i < nthreads_; i++) {
        splatThreads.emplace_back(tsplat, i);
    }
    for (auto &t : splatThreads) {
        t.join();
    }

    // Reduce leaf values
    for (int i = 0; i < nthreads_; i++) {
        float *tleafValues = mtleafValues + i * tree_->nLeaves() * vd_;
        for (int j = 0; j < tree_->nLeaves(); j++) {
            for (int v = 0; v < vd_; v++) {
                leafValues[j * vd_ + v] += tleafValues[j * vd_ + v];
            }
        }
    }
    delete[] mtleafValues;

    // Clear the output array
    memset(out, 0, sizeof(float) * n_ * vd_);

    // Slicing: For each position vector ...
    auto tslice = [&](int threadi) {
        int *tindices = indices + threadi * outleaf;
        float *tweights = weights + threadi * outleaf;
        const float *tpos = pos + threadi * tn_ * pd_;
        float *tout = out + threadi * tn_ * vd_;
        for (int i = 0; i < tn_; i++) {
            // find up to 64 leaves nearby ...
            int results =
                tree_->lookup(tpos + i * pd_, tindices, tweights, outleaf);
            // and gather from them.
            for (int j = 0; j < results; j++) {
                for (int k = 0; k < vd_; k++) {
                    tout[i * vd_ + k] +=
                        leafValues[tindices[j] * vd_ + k] * tweights[j];
                }
            }
        }
    };

    std::vector<std::thread> sliceThreads;
    for (int i = 0; i < nthreads_; i++) {
        sliceThreads.emplace_back(tslice, i);
    }
    for (auto &t : sliceThreads) {
        t.join();
    }

    delete[] indices;
    delete[] weights;
    delete[] leafValues;
}