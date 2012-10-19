#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "Consts.h"
#include "DataManager.h"
#include "FeatureTracker.h"

class BlockController {

public:
    BlockController();
    ~BlockController();

    void InitData(Vector3i partition, Vector3i blockCoord, DataSet ds);
    void TrackForward();
    void ExtractAllFeatures();
    void UpdateLocalGraph(int blockID, Vector3i blockCoord);

    // Feature Connectivity Graph
    vector<int> GetAdjacentBlocks();
    vector<Edge> GetLocalGraph() { return localGraph; }
    void SetLocalGraph(vector<Edge> graph) { localGraph = graph; }

    // Accessor - FeatureTracker
    void SetVolumeDataPointerByIndex(int index) { pFeatureTracker->SetVolumeDataPointer(pDataManager->GetVolumeDataPointer(index));}
    void SetCurrentTimestep(int index) { timestep = index; }

private:
    DataManager     *pDataManager;
    FeatureTracker  *pFeatureTracker;

    IntMap          adjacentBlocks;
    vector<Edge>    localGraph;
    Vector3i        blockSize;
    int             timestep;

    void initAdjacentBlocks(Vector3i partition, Vector3i blockCoord);
};

#endif // DATABLOCKCONTROLLER_H
