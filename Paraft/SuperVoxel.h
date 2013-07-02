#ifndef SUPERVOXEL_H
#define SUPERVOXEL_H

#include "Utils.h"
#include "Metadata.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class SuperVoxel {
public:
    explicit SuperVoxel(const Vector3i dim);
    explicit SuperVoxel(const Metadata& meta);
    virtual ~SuperVoxel();

//    void InitWith(const string& image_path);
//    void InitWith(const Metadata& meta);

    // segment by specific 1. number of segments or 2. segment size
    void SegmentByNumber(const int expectedClusterNum, const float compactness);
    void SegmentBySize(const int expectedClusterSize, const float compactness);
    void DrawContours(const CvScalar& drawing_color, const string& save_path);

    // accessors
    int GetNumCluster()                 const { return numCluster_; }
    const int* GetSegmentMap()          const { return pClusters_; }
    const vector<Cluster> GetClusters() const { return clusters_; }

    void SetDataPtr(float *pData) { pData_ = pData; }

private:
    float *pData_;
    float *pMask_;

    Vector3i dim_;

    int GetVoxelIndex(const Vector3i &v) const { return dim_.x*dim_.y*v.z+dim_.x*v.y+v.x; }
    Vector3i GetVoxelPosition(const int idx) const {
        int z = idx / (dim_.x * dim_.y);
        int y = (idx - z * dim_.x * dim_.y) / dim_.x;
        int x = idx % dim_.x;
        return Vector3i(x,y,z);
    }

    //-----

    IplImage *pImage_;  // input image
    int numCluster_;    // number of segments after segmentation

    vector<Cluster> clusters_;  // segment result

    int *pClusters_;
    int *pClustersTmp_; // intermedia


    int kNumElements_;   // number of pixels / voxels in the data

    float *pLs;
    float *pAs;
    float *pBs;
    float *pGradients_;

    vector<Vector3i> centroids_;
    vector<float>    centroidsValues_;

    vector<float> pCLs;
    vector<float> pCAs;
    vector<float> pCBs;
    vector<int> pCXs;
    vector<int> pCYs;

    void bgr2lab();  // opencv uses BGR instead of RGB
    void detectGradients(); // detects gradient map to perturb seeds
    void getInitialCenters(int expectedClusterSize);
    void getInitialCentroids(int expectedClusterSize);  // for voxel, 3d

    // Super pixel clustering. Need post-processing for enforcing connectivity.
    void clusteringIteration(int expectedClusterSize, float compactness, int* pClustersTmp_);

    // Find next connected components(pixel) which belongs to the same cluster.
    // Function is called recursively to get the size of connected area cluster.
    void findNext(const int* pClustersTmp_, int x, int y, int clusterIndex, int* x_pos, int* y_pos, int* num_count);

    void enforceConnectivity(const int* pClustersTmp_, int expectedClusterSize);
};

#endif // SUPERVOXEL_H
