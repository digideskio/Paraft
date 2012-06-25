#ifndef MULTICORECONTROLLER_H
#define MULTICORECONTROLLER_H

#include <fstream>
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

    MPI_Comm workerCommunicator;
    MPI_Comm adjacentCommunicator;
//    MPI_Group adjacentGroup;
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
    int globalGraphSize;
    Edge *pGlobalGraph;
    Edge* updateGlobalGraph(vector<Edge> localEdgesVector);

    // for feature graph
    Edge* updateFeatureGraph(vector<Edge> localEdgesVector);

    CSVWriter   csv;
    DataSet     dataset;

    void initBlockController();
    void initLocalCommunicator();
    void syncTFParameters();
    void precalculateT0();
    void waitingForOrders();
    void trackForward_worker();

    void debug(string msg);
};

#endif // MULTICORECONTROLLER_H
