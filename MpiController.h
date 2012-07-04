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

    MPI_Comm local_comm;
    MPI_Status status;
    int my_rank;
    int num_proc;

    CSVWriter csv;
    DataSet ds;

    Vector3i partition;
    Vector3i blockCoord;
    int timestep;

    // for global graph
    int globalEdgeSize;
    vector<Edge> updateGlobalGraph(vector<Edge> localEdgesVector);

    // for feature graph
    int adjacentBlockCount;
    int adjacentEdgeCount;
    Edge *pFeatureGraph;
    vector<Edge> updateFeatureGraph(vector<Edge> localEdgesVector);

    void initBlockController();
    void initLocalCommGroup();
    void initTFParameters();
    void precalculateT0();

    void mergeCorrespondentEdges(vector<Edge> &edgeVector);

    void debug(string msg);
};

#endif // MULTICORECONTROLLER_H
