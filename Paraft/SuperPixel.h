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

    vector<float> k_centers_l_;
    vector<float> k_centers_a_;
    vector<float> k_centers_b_;
    vector<int> k_centers_x_;
    vector<int> k_centers_y_;

    void bgr2lab();  // opencv uses BGR instead of RGB
    void detectGradients(); // detects gradient map to perturb seeds
    void getInitialCenters(const int& expectedClusterSize);

    // Super pixel clustering. Need post-processing for enforcing connectivity.
    void clusteringIteration(const int& expectedClusterSize, const float& weight_m,
                             int* temp_segmentation_map);

    // Find next connected components(pixel) which belongs to the same cluster.
    // Function is called recursively to get the size of connected area cluster.
    void findNext(const int* temp_segmentation_map, const int& row_index, const int& col_index,
                  const int& segment_index, int* x_pos, int* y_pos, int* num_count);

    void enforceConnectivity(const int* temp_segmentation_map, const int& expected_seg_size);
};

#endif // SUPERPIXEL_H
