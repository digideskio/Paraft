#ifndef MULTICORECONTROLLER_H
#define MULTICORECONTROLLER_H

#include <fstream>
#include <numeric>
#include "BlockController.h"
#include "mpi.h"

class MpiController {

public:
    MpiController();
    ~MpiController();

    void Init(int argc, char** argv);
    void Start();
    void TrackForward();

private:
    BlockController *pBlockController;

    MPI_Comm worker_comm;
    MPI_Comm adjacent_comm;
    MPI_Status status;

    int globalID;
    int blockID;
    int blockGID;
    int timestep;
    int globalNumProc;
    int blockCount;

    Vector3i partition;
    Vector3i blockCoord;

    // for global graph
    int globalEdgeCount;
    vector<Edge> updateGlobalGraph(vector<Edge> localEdgesVector);

    // for feature graph
    int adjacentBlockCount;
    int adjacentEdgeCount;
    Edge *pFeatureGraph;
    vector<Edge> updateFeatureGraph(vector<Edge> localEdgesVector);

    CSVWriter   csv;
    DataSet     dataset;

    void initBlockController();
    void initLocalCommGroup();
    void syncTFParameters();
    void precalculateT0();
    void waitingForOrders();
    void trackForward_worker();

    void mergeCorrespondentEdges(vector<Edge> &edgeVector);

    void debug(string msg);
};

#endif // MULTICORECONTROLLER_H