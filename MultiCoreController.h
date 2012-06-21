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

    int gID;
    int wID;
    int wGID;
    int timestep;
    int globalNumProcesses;
    int workerNumProcesses;

    Vector3i wSegXYZ;
    Vector3i wXYZ;

    Edge *pGlobalGraph;
    int globalGraphSize;

    CSVWriter   csv;
    DataSet     ds;

    void initDataBlockController();
    void syncTFParameters();
    void precalculateT0();
    void waitingForOrders();
    void trackForward_worker();

    Edge* updateGlobalConnectivityGraph(vector<Edge> localEdgesVector);
    void debug(string msg);
};

#endif // MULTICORECONTROLLER_H
