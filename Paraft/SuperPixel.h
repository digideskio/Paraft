#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include "Utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Cluster {
    Vector3i center;
    int num_pixel;
};

class SuperPixel {
public:
    explicit SuperPixel();
    virtual ~SuperPixel();

    void InitWith(const string& image_path);

    // segment by specific 1. number of segments or 2. segment size
    bool SegmentNumber(const int& expectedClusterNum, const float& compactness);
    bool SegmentSize(const int& expectedClusterSize, const float& compactness);
    void DrawContours(const CvScalar& drawing_color, const string& save_path);

    // accessors
    int GetNumCluster()                 const { return numCluster_; }
    const int* GetSegmentMap()          const { return pClusters_; }
    const vector<Cluster> GetClusters() const { return clusters_; }

private:
    IplImage *pImage_;  // input image
    int numCluster_;    // number of segments after segmentation

    vector<Cluster> clusters_;  // segment result

    int *pClusters_;
    int *pClustersTmp_; // intermedia

    Vector3i dim_;      // data dimension
    int kNumElements_;   // number of pixels / voxels in the data

    float *pLs;
    float *pAs;
    float *pBs;
    float *pGradients_;

    vector<float> pCLs;
    vector<float> pCAs;
    vector<float> pCBs;
    vector<int> pCXs;
    vector<int> pCYs;

    void bgr2lab();  // opencv uses BGR instead of RGB
    void detectGradients(); // detects gradient map to perturb seeds
    void getInitialCenters(int expectedClusterSize);

    // Super pixel clustering. Need post-processing for enforcing connectivity.
    void clusteringIteration(int expectedClusterSize, float compactness, int* pClustersTmp_);

    // Find next connected components(pixel) which belongs to the same cluster.
    // Function is called recursively to get the size of connected area cluster.
    void findNext(const int* pClustersTmp_, int x, int y, int clusterIndex, int* x_pos, int* y_pos, int* num_count);

    void enforceConnectivity(const int* pClustersTmp_, int expectedClusterSize);
};

#endif // SUPERPIXEL_H
