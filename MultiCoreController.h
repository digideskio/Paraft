#ifndef MULTICORECONTROLLER_H
#define MULTICORECONTROLLER_H

#include <fstream>
#include "DataBlockController.h"
#include "mpi.h"

class MultiCoreController {

public:
    MultiCoreController();
    ~MultiCoreController();

    void Init(int argc, char** argv);
    void Start();
    void TrackForward();

private:
    DataBlockController *pDataBlockController;
    MPI_Comm MY_COMM_WORKER;
    MPI_Status status;

    int globalID;
    int blockID;
    int blockGID;
    int timestep;
    int globalNumProc;
    int blockCount;

    Vector3i partition;
    Vector3i wCoord;

    // for global graph
    int globalGraphSize;
    Edge *pGlobalGraph;
    Edge* updateGlobalGraph(vector<Edge> localEdgesVector);

    // for feature graph
    Edge* updateFeatureGraph(vector<Edge> localEdgesVector);

    CSVWriter   csv;
    DataSet     ds;

    void initDataBlockController();
    void syncTFParameters();
    void precalculateT0();
    void waitingForOrders();
    void trackForward_worker();

    void debug(string msg);
};

#endif // MULTICORECONTROLLER_H
