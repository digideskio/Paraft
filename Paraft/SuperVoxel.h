#ifndef SUPERVOXEL_H
#define SUPERVOXEL_H

#include "Utils.h"
#include "Metadata.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class SuperVoxel {
public:
    explicit SuperVoxel(const vector3i dim);
//    explicit SuperVoxel(const Metadata& meta);
    virtual ~SuperVoxel();

    // segment by specific 1. number of segments or 2. segment size
    void ClusterByNumber(const int numClusters, const float compactness);
    void ClusterBySize(const int segLength, const float compactness);

    void SegmentByNumber(const int expectedClusterNum, const float compactness);
    void SegmentBySize(const int expectedClusterSize, const float compactness);
    void DrawContours(const CvScalar& drawing_color, const string& save_path);

    // accessors
    int GetNumCluster()                 const { return numClusters_; }
    const vector<Cluster> GetClusters() const { return clusterInfo_; }
    void SetDataPtr(float *pData)             { pData_ = pData; }

private:
    const std::vector<vector3i> kNeighbors_ = {
        vector3i(-1, 0, 0),    // left
        vector3i( 1, 0, 0),    // right
        vector3i( 0, 1, 0),    // top
        vector3i( 0,-1, 0),    // bottom
        vector3i( 0, 0,-1),    // front
        vector3i( 0, 0, 1)     // back
    };

    float* pData_;

//    std::vector<float> data_;
    std::vector<int>   mask_;
    std::vector<int>   maskTmp_;
    std::vector<float> gradients_;
    std::vector<float> centroidValues_;
    std::vector<vector3i> centroidCoords_;
    std::vector<Cluster> clusterInfo_;  // segment result

    vector3i      dim_;
    int           numVoxels_;   // number of voxels in the volume data
    int           numClusters_; // number of segments after segmentation

    int GetVoxelIndex(const vector3i &v) const { return dim_.x*dim_.y*v.z+dim_.x*v.y+v.x; }
    vector3i GetVoxelPosition(const int idx) const {
        int z = idx / (dim_.x * dim_.y);
        int y = (idx - z * dim_.x * dim_.y) / dim_.x;
        int x = idx % dim_.x;
        return vector3i(x,y,z);
    }

    void calculateGradientsForEachVoxel();
    void dispatchInitialSeeds(int segLength);
    void perturbSeedsToLocalMinimumGradient();
    void clustering(int segLength, float compactness);
    void enforceConnectivity(int segLength);

    //-----

    IplImage *pImage_;  // input image

    vector<int> xcenters_;
    vector<int> ycenters_;

    vector<float> ls_;
    vector<float> as_;
    vector<float> bs_;
    vector<float> lcenters_;
    vector<float> acenters_;
    vector<float> bcenters_;


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
    void growRegion(const vector3i& seed, const int clusterIndex, std::vector<vector3i> &pos, int& count);
    void grow(const vector3i& seed, const int clusterIndex, std::vector<vector3i>& pos, int* count);
};

#endif // SUPERVOXEL_H
