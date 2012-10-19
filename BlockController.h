#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "Consts.h"
#include "DataManager.h"
#include "FeatureTracker.h"

class BlockController {

public:
    BlockController();
    ~BlockController();

    void InitParameters(Vector3i partition, Vector3i blockCoord, DataSet ds);
    void TrackForward(Vector3i partition, Vector3i blockCoord, DataSet ds);
    void ExtractAllFeatures();
    void UpdateLocalGraph(int blockID, Vector3i blockCoord);
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
    Vector3i        blockSize;
    int             t;  // current timestep

    void initAdjacentBlocks(Vector3i partition, Vector3i blockCoord);
};

#endif // DATABLOCKCONTROLLER_H
