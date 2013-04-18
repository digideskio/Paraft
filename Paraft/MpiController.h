#ifndef MULTICORECONTROLLER_H
#define MULTICORECONTROLLER_H

#include <fstream>
#include <mpi.h>
#include "BlockController.h"

class MpiController {

public:
    MpiController();
    ~MpiController();

    void InitWith(int argc, char** argv);
    void Start();
    void TrackForward();

private:
    BlockController *pBlockController;

    MPI_Datatype MPI_TYPE_EDGE;
    MPI_Comm MPI_COMM_LOCAL;
    MPI_Request request;
    MPI_Status status;

    int myRank;
    int numProc;
    int t;  // current timestep

    CSVWriter csv;
    Metadata meta;

    Vector3 gridDim;     // #processes in each dimension (xyz)
    Vector3 blockIdx;    // xyz coordinate of current processor

    // for global graph
    int globalEdgeCount;
    void gatherGlobalGraph();

    // for feature graph
    vector<int> adjacentBlocks;
    void syncFeatureGraph();
    void updateFeatureTable(Edge edge);
    bool need_to_send;
    bool need_to_recv;
    bool any_send, any_recv;

    // global feature info
    FeatureTable featureTable;
    hash_map<int, FeatureTable> featureTableVector;    // for time varying data

    void initBlockController();
    void mergeCorrespondentEdges(vector<Edge> edges);
};

#endif // MULTICORECONTROLLER_H
