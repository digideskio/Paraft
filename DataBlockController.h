#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "hash_map.h"
#include "DataManager.h"
#include "FeatureTracker.h"
#include "Consts.h"

class DataBlockController {

public:
    DataBlockController();
    ~DataBlockController();

    void InitData(int globalID, Vector3i workerNumProcXYZ,
                  Vector3i workerIDXYZ, DataSet dataset);

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
    hash_map<int, float> GetDifferentPoints() { return pFeatureTracker->GetDiffPoints(); }

    // Feature Connectivity Graph
    hash_map<int, int> GetAdjacentBlocks() { return adjacentBlocks; }

    vector<Edge> GetLocalEdges() { return localGraph; }

    // Accessor - FeatureTracker
    void SetVolumeDataPointerByIndex(int index) { pFeatureTracker->SetVolumeDataPointer(pDataManager->GetVolumeDataPointer(index));}
    void SetTFResolution(int res) { pFeatureTracker->SetTFResolution(res); }
    void SetTFColorMap(float* map) { pFeatureTracker->SetTFColorMap(map); }
    void SetCurrentTimestep(int index) { currentTimestep = index; }
    void ClearHighlightedFeatureList() { highlightedFeatures.clear(); }
    void AddHighlightedFeature(int index);
    void ResetMaskMatrixValue(float value);
    int GetPointIndex(DataPoint p) { return pFeatureTracker->GetPointIndex(p); }

private:
    hash_map<int, int>  adjacentBlocks;
    DataManager     *pDataManager;
    FeatureTracker  *pFeatureTracker;
    vector<Edge>    localGraph;
    vector<int>     highlightedFeatures;
    Vector3i        dataDim;
    int             currentTimestep;
    int             xs, ys, zs;

    void saveExtractedFeatures(vector<Feature>* f);
    void initAdjacentBlocks(Vector3i blockPartition, Vector3i blockCoord);
};

#endif // DATABLOCKCONTROLLER_H
