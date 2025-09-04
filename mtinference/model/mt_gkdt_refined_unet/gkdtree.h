#ifndef GKDTREE_H_
#define GKDTREE_H_

#include <algorithm>
#include <limits>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <thread>

using namespace std;

using std::swap;
using std::thread;
using std::vector;

class GKDTree {
  private:
    /* data */

    // The interface for nodes
    virtual class Node {
      public:
        ~Node() {};
        // Query the kdtree. Same interface as above, but also tracks
        // the probability of reaching this node using the last
        // argument.
        virtual int lookup(const float *query, int *ids, float *weights,
                           int nSamples, float p) = 0;

        // Compute the bounds of the node along the cut dimension
        virtual void computeBounds(float *mins, float *maxs) {}
    };

    // An internal split node.
    class Split : public Node {
      public:
        // The dimension along which this cell cuts
        int cutDim;
        // The cut value and bounds in that dimension
        float cutVal, minVal, maxVal;
        // The children of this node
        Node *left, *right;

        virtual ~Split();
        inline float pLeft(float value);
        int lookup(const float *query, int *ids, float *weights, int nSamples,
                   float p);
        void computeBounds(float *mins, float *maxs);
    };

    // A leaf node. Has an id and a position.
    class Leaf : public Node {
      public:
        int id;
        vector<float> position;

        ~Leaf();
        int lookup(const float *query, int *ids, float *weights, int nSamples,
                   float p);
    };

    Node *build(const float **pos, int n);

    Node *root;
    int dimensions;
    float sizeBound;
    int leaves;

  public:
    GKDTree(int kd, const float **pos, int n, float sBound);
    ~GKDTree();
    int nLeaves();
    int lookup(const float *query, int *ids, float *weights, int nSamples);
};

class GKDTFilter {
  private:
    int pd_ = -1, n_ = -1;
    int nthreads_ = 1;
    int tn_ = -1;
    GKDTree *tree_ = NULL;

  public:
    GKDTFilter(int pd, int n, int nthreads);
    ~GKDTFilter();
    void seqinit(const float *pos);
    void seqcompute(const float *val, int vd, const float *pos, float *out);
    void mtcompute(const float *val, int vd, const float *pos, float *out);
};

#endif