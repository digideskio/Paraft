#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "Consts.h"
#include "DataManager.h"
#include "FeatureTracker.h"

class BlockController {

public:
    BlockController();
    ~BlockController();

    void InitParameters(Vector3 gridDim, Vector3 blockIdx, Metadata meta);
    void TrackForward(Vector3 gridDim, Vector3 blockIdx, Metadata meta);
    void ExtractAllFeatures();
    void UpdateLocalGraph(int blockID, Vector3 blockIdx);
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
    Vector3         blockDim;
    int             t;  // current timestep

    void initAdjacentBlocks(Vector3 gridDim, Vector3 blockCoord);
};

#endif // DATABLOCKCONTROLLER_H
