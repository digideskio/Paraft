#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "hash_map.h"
#include "Consts.h"
#include "DataManager.h"
#include "FeatureTracker.h"

class BlockController {

public:
    BlockController();
    ~BlockController();

//    void InitData(int globalID, Vector3i partition, Vector3i blockCoord, DataSet dataset);
    void InitData(Vector3i partition, Vector3i blockCoord, DataSet ds);
    void TrackForward();
    void ExtractAllFeatures();
    void UpdateLocalGraph(int blockID, Vector3i blockCoord);

    // Accessor - DataManager
    float* GetMaskMatrixPointer() { return pDataManager->GetMaskMatrixPointer(); }
    float* GetVolumeDataPointer(int index) { return pDataManager->GetVolumeDataPointer(index); }
    int GetVolumeSize() { return pDataManager->GetVolumeSize(); }
    int GetFeatureVectorLength() { return pDataManager->GetFeatureVectorLength(); }
    vector<int> GetHighlightedFeatures() { return highlightedFeatures; }
    vector<Feature> *GetFeatureVector(int index) { return pDataManager->GetFeatureVector(index); }
    IndexValueMap GetDifferentPoints() { return pFeatureTracker->GetDiffPoints(); }

    // Feature Connectivity Graph
    IntMap GetAdjacentBlocks() { return adjacentBlocks; }
    vector<Edge> GetLocalEdges() { return localGraph; }

    // Accessor - FeatureTracker
    void SetVolumeDataPointerByIndex(int index) { pFeatureTracker->SetVolumeDataPointer(pDataManager->GetVolumeDataPointer(index));}
    void SetTFResolution(int res) { pFeatureTracker->SetTFResolution(res); }
    void SetTFColorMap(float* map) { pFeatureTracker->SetTFColorMap(map); }
    void SetCurrentTimestep(int index) { currentTimestep = index; }
    void ClearHighlightedFeatureList() { highlightedFeatures.clear(); }
    void AddHighlightedFeature(int index);
    void ResetMaskMatrixValue(float value);
    int  GetPointIndex(DataPoint p) { return pFeatureTracker->GetPointIndex(p); }

private:
    IntMap          adjacentBlocks;
    DataManager     *pDataManager;
    FeatureTracker  *pFeatureTracker;
    vector<Edge>    localGraph;
    vector<int>     highlightedFeatures;
    Vector3i        blockSize;
    int             currentTimestep;
    int             xs, ys, zs;

    void saveExtractedFeatures(vector<Feature>* f);
    void initAdjacentBlocks(Vector3i partition, Vector3i blockCoord);
};

#endif // DATABLOCKCONTROLLER_H
