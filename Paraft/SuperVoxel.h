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

    // segment by specific 1. number of segments or 2. segment size
    void ClusterByNumber(const int numClusters, const float compactness);
    void ClusterBySize(const int segLength, const float compactness);

    void SegmentByNumber(const int expectedClusterNum, const float compactness);
    void SegmentBySize(const int expectedClusterSize, const float compactness);
    void DrawContours(const CvScalar& drawing_color, const string& save_path);

    // accessors
    int GetNumCluster()                 const { return numClusters_; }
//    const int* GetSegmentMap()          const { return pClusters_; }
//    const vector<int>& GetMasks()       const { return &masks_; }
    const vector<Cluster> GetClusters() const { return clusterInfo_; }

    void SetDataPtr(float *pData) { pData_ = pData; }

private:
    std::vector<Vector3i> kNeighbors_;

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

    void calculateGradientsForEachVoxel();
    void dispatchInitialSeeds(int segLength);
    void perturbSeedsToLocalMinGradient();
    void clustering(int segLength, float compactness);
    void enforceConnectivity(int segLength);

    //-----

    IplImage *pImage_;  // input image

    int numVoxels_;   // number of pixels / voxels in the data
    int numClusters_;    // number of segments after segmentation

    vector<Cluster> clusterInfo_;  // segment result

    vector<int> masks_;
    vector<int> masksTmp_;
    vector<int> xcenters_;
    vector<int> ycenters_;

    vector<float> ls_;
    vector<float> as_;
    vector<float> bs_;
    vector<float> lcenters_;
    vector<float> acenters_;
    vector<float> bcenters_;
    vector<float> gradients_;

    vector<Vector3i> centroidCoords_;
    vector<float>    centroidValues_;

    void bgr2lab();  // opencv uses BGR instead of RGB
    void detectGradients(); // detects gradient map to perturb seeds
    void getInitialCenters(int expectedClusterSize);
    void getInitialCentroids(int expectedClusterSize);  // for voxel, 3d

    // Super pixel clustering. Need post-processing for enforcing connectivity.
    void clusteringIteration(int expectedClusterSize, float compactness);

    // Find next connected components(pixel) which belongs to the same cluster.
    // Function is called recursively to get the size of connected area cluster.
    void findNext(int x, int y, int clusterIndex, int* x_pos, int* y_pos, int* num_count);

    void enforceConnect(int expectedClusterSize);
    void growRegion(const Vector3i& seed, const int clusterIndex, std::vector<Vector3i> &pos, int& count);
};

#endif // SUPERVOXEL_H
