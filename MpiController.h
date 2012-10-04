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

    int my_rank;
    int num_proc;
    int timestep;

    CSVWriter csv;
    DataSet ds;

    Vector3i partition;     // #processes in each dimension (xyz)
    Vector3i blockCoord;    // xyz coordinate of current processor

    // for global graph
    int globalEdgeCount;
    void gatherGlobalGraph();

    // for feature graph
    vector<int> adjacentBlocks;
    void syncFeatureGraph();
    void updateFeatureTable(Edge edge);
    int need_to_send;
    int need_to_recv;
    int any_send, any_recv;

    // for adjacent graph
    void gatherNeighboringGraph();

    // global feature info
    FeatureTable featureTable;
    hash_map<int, FeatureTable> featureTableVector;    // for time varying data

    void initBlockController();
    void initParameters();
    void precalculateTimestep1();

    void mergeCorrespondentEdges(vector<Edge> edges);

    void debug(string msg);
};

#endif // MULTICORECONTROLLER_H
