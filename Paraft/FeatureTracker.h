#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include <math.h>
#include <vector>
#include <list>
#include "Consts.h"

using namespace std;

class FeatureTracker {

public:
    FeatureTracker(Vector3 size);
    ~FeatureTracker() ;

    void Reset();

    // Set seed at current time step. FindNewFeature will do three things :
    // 1. Do region growing at the current time step
    // 2. Adding a center point into the center point list
    // 3. Adding edge points into the edge list
//    void FindNewFeature(int x, int y, int z, float lowerValue, float upperValue);
    void FindNewFeature(DataPoint point, float lowerValue, float upperValue);

    // Track forward one time step based on the center points of the features at the last time step
    void TrackFeature(float* pDataSet, float lowerValue, float upperValue, int direction, int mode);

    // Get mask matrix of current time step
    float *GetMaskVolumePointer() { return pMaskVolumeCurrent; }

    void SaveExtractedFeatures(int index);

    // accessor
    void SetVolumeDataPointer(float* pData) { pVolumeData = pData; }
    void SetTFOpacityMap(float* map) { pTFOpacityMap = map; }
    void SetTFResolution(int res) { tfResolution = res; }
    int GetPointIndex(DataPoint point) { return xs*ys*point.z+xs*point.y+point.x; }
    int GetTFResolution() { return tfResolution; }
    float* GetTFOpacityMap() { return pTFOpacityMap; }
    IndexValueMap GetDiffPoints() { return diffPoints; }

    // Get all features information of current time step
    vector<Feature>* GetFeatureVectorPointer(int index) { return &featureSequence[index]; }
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

    float* pMaskVolumeCurrent;   // Mask volume, same size with a time step data
    float* pMaskVolumePrevious;  // Mask volume, for backward time step when tracking forward & backward
    float* pVolumeData;         // Volume intensity value
    float* pTFOpacityMap;
    float lowerThreshold, upperThreshold;
    int tfResolution;
    int numVoxelinFeature;      // Used to calculate the number of voxel in the feature
    int xs,ys,zs;               // Dimension of the dataset

    float maskValue;
    int volumeSize;
    int timestepsAvailableForward;
    int timestepsAvailableBackward;

    IndexValueMap diffPoints;

    // FeatureInfo
    list<DataPoint> dataPointList;  // Active queue for some kind of points ?!
    list<DataPoint> surfacePoints;  // For edge points saving
    list<DataPoint> innerPoints;    // Hold the temp buffer of the voxel position

    Vector3    centroid;       // center point of a single feature
    Vector3    featureMin;    // min value of x,y,z of a single feature
    Vector3    featureMax;    // max value of x,y,z of a single feature
    Vector3    sumCoordinateValue;    // Sum of the voxel values of the feature
    Vector3    delta;

    Vector3    sumBoundaryCoordinate[6];  // Sum of the voxel values on boundary surface
    int         numVoxelonBoundary[6];

    // 6 possible ghost area
    Vector3    boundaryCentroid[6];// center point of the ghost area of a single feature
    Vector3    boundaryMin[6];     // min value of (x,y)|(x,z)|(y,z) of the boundary surface
    Vector3    boundaryMax[6];     // max value of (x,y)|(x,z)|(y,z) of the boundary surface
    vector<int> touchedSurfaces;    // Which boundary the feature touches

    vector<Feature> currentFeaturesHolder; // Features info in current time step
    vector<Feature> backup1FeaturesHolder; // ... in the 1st backup time step
    vector<Feature> backup2FeaturesHolder; // ... in the 2nd backup time step
    vector<Feature> backup3FeaturesHolder; // ... in the 3rd backup time step

    FeatureVectorSequence featureSequence;
};

#endif // FEATURETRACKER_H
