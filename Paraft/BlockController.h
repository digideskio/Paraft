#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "Consts.h"
#include "DataManager.h"
#include "FeatureTracker.h"

class BlockController {

public:
    BlockController();
    ~BlockController();

    void InitParameters(Vector3i gridDim, Vector3i blockIdx, Metadata meta);
    void TrackForward(Vector3i gridDim, Vector3i blockIdx, Metadata meta);
    void ExtractAllFeatures();
    void UpdateLocalGraph(int blockID, Vector3i blockIdx);
    void SetCurrentTimestep(int timestep) { t = timestep; }

    // Feature Connectivity Graph
    vector<int> GetAdjacentBlocks();
    vector<Edge> GetLocalGraph() { return localGraph; }
    void SetLocalGraph(vector<Edge> graph) { localGraph = graph; }

private:
    DataManager     *pDataManager;
    FeatureTracker  *pFeatureTracker;

    IntMap          adjacentBlocks;
    vector<Edge>    localGraph;
    Vector3i        blockDim;
    int             t;  // current timestep

    void initAdjacentBlocks(Vector3i gridDim, Vector3i blockIdx);
};

#endif // DATABLOCKCONTROLLER_H
