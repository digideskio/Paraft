#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include <hash_map.h>
#include <math.h>
#include <vector>
#include <list>
#include <map>
#include "Consts.h"

using namespace std;

class FeatureTracker {

public:
    FeatureTracker(int xsize, int ysize, int zsize);
    ~FeatureTracker() ;

    void Reset();

    // Set seed at current time step. SetSeed will do three things :
    // 1. Do region growing at the current time step
    // 2. Adding a center point into the center point list
    // 3. Adding edge points into the edge list
    void FindNewFeature(int x, int y, int z, float lowerValue, float upperValue);

    // Track forward one time step based on the center points of the features at the last time step
    void TrackFeature(float* pDataSet, float lowerValue, float upperValue, int direction, int mode);

    // Get mask matrix of current time step
    float *GetMaskMatrixPointer() { return pMaskMatrixCurrent; }

    // accessor
    void SetVolumeDataPointer(float* pData) { pVolumeData = pData; }
    void SetTFColorMap(float* map) { pTFColorMap = map; }
    void SetTFResolution(int res) { tfResolution = res; }
    int GetPointIndex(DataPoint point) { return xs*ys*point.z+xs*point.y+point.x; }
    float* GetTFColorMap() { return pTFColorMap; }
    int GetTFResolution() { return tfResolution; }
    hash_map<int, float> GetDiffPoints() { return diffPoints; }


    // Get all features information of current time step
    vector<Feature>* GetCurrentFeatureInfo() { return &currentFeaturesHolder; }
    void SetCurrentFeatureInfo(vector<Feature>* pFeature);
    void ClearCurrentFeatureInfo() { currentFeaturesHolder.clear(); }

private:
    void predictRegion(int index, int direction, int mode); // predict region t based on direction
    void fillRegion(float maskValue);                       // scanline algorithm - fills everything inside edge
    void expandRegion(float segmentValue);                  // grows edge where possible
    void shrinkRegion(float maskValue);                     // shrinks edge where nescessary
    bool expandEdge(DataPoint point, float segmentValue);   // sub-func inside expandRegion
    void shrinkEdge(DataPoint point, float maskValue);      // sub-func inside shrinkRegion
    void backupFeatureInfo(int direction);                  // Update the feature vectors information after tracking

    float getOpacity(float value);

    void resetFeatureBoundaryInfo();
    void updateDiffPointList(int index, float valule);
    void updateFeatureMinMax(DataPoint point);
    void updateBoundaryMinMax(DataPoint point, int surface);
    void updateTouchedSurfaces();

    float* pMaskMatrixCurrent;   // Mask matrix, same size with a time step data
    float* pMaskMatrixPrevious;  // Mask matrix, for backward time step when tracking forward & backward
    float* pVolumeData;         // Volume intensity value
    float* pTFColorMap;
    float lowerThreshold, upperThreshold;
    int tfResolution;
    int numVoxelinFeature;      // Used to calculate the number of voxel in the feature
    int xs,ys,zs;               // Dimension of the dataset

    float maskValue;
    int volumeSize;
    int timestepsAvailableForward;
    int timestepsAvailableBackward;

    hash_map<int, float> diffPoints;

    // FeatureInfo
    list<DataPoint> dataPointList;  // Active queue for some kind of points ?!
    list<DataPoint> surfacePoints;  // For edge points saving
    list<DataPoint> innerPoints;    // Hold the temp buffer of the voxel position

    Vector3i    centroid;       // center point of a single feature
    Vector3i    featureMin;    // min value of x,y,z of a single feature
    Vector3i    featureMax;    // max value of x,y,z of a single feature
    Vector3i    sumCoordinateValue;    // Sum of the voxel values of the feature
    Vector3f    delta;

    Vector3i    sumBoundaryXYZValue[6];  // Sum of the voxel values on boundary surface
    int         numVoxelonBoundary[6];

    // 6 possible ghost area
    Vector3i    boundaryCentroid[6];   // center point of the ghost area of a single feature
    Vector3i    boundaryMin[6];    // min value of (x,y)|(x,z)|(y,z) of the boundary surface
    Vector3i    boundaryMax[6];    // max value of (x,y)|(x,z)|(y,z) of the boundary surface
    vector<int> touchedSurfaces; // Which boundary the feature touches

    vector<Feature> currentFeaturesHolder; // Hold all the features information in current time step
    vector<Feature> backup1FeaturesHolder; // Hold all the features information in the first backup time step
    vector<Feature> backup2FeaturesHolder; // Hold all the features information in the second backup time step
    vector<Feature> backup3FeaturesHolder; // Hold all the features information in the third backup time step
};

#endif // FEATURETRACKER_H
