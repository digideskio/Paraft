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

    MPI_Datatype MPI_TYPE_EDGE;
    MPI_Request request;
    MPI_Status status;

    int my_rank;
    int num_proc;

    CSVWriter csv;
    DataSet ds;

    Vector3i partition;
    Vector3i blockCoord;
    int timestep;

    // for global graph
    int globalEdgeCount;
    vector<Edge> updateGlobalGraph(vector<Edge> localEdgeVector);

    // for feature graph
    vector<int> adjacentBlocks;
    vector<Edge> adjacentGraph;

    // global feature info
    FeatureTable featureTable;
    vector<FeatureTable> featureTableVector;    // for time varying data

    void initBlockController();
    void initTFParameters();
    void precalculateT0();

    void syncFeatureGraph();
    void updateFeatureTable(Edge edge);
    bool featureTableUpdated;

    void mergeCorrespondentEdges();

    void debug(string msg);
};

#endif // MULTICORECONTROLLER_H
