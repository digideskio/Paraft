#ifndef MULTICORECONTROLLER_H
#define MULTICORECONTROLLER_H

#define MPI_TAG_NULL -1
#define MPI_TAG_TIMESTEP_0 0
#define MPI_TAG_TF_RESOLUTION 2
#define MPI_TAG_TF_COLOR_MAP 3
#define MPI_TAG_SEGMENT_MATRIX 4
#define MPI_TAG_ROUTER 5
#define MPI_TAG_HIGHLIGHT_FEATURE 6
#define MPI_TAG_SELECTED_FEATURE_INFO 7
#define MPI_TAG_SELECTED_FEATURE_INFO_SIZE 8
#define MPI_TAG_SYNC_TIMESTEP 9
#define MPI_TAG_TRACK_FORWARD 10
#define MPI_TAG_GET_FEATURE_ID 11
#define MPI_TAG_SET_FEATURE_ID 12

#define INT_SIZE 1
#define FLOAT_SIZE sizeof(float)

#include <fstream>
#include <QObject>
#include <QDebug>
#include "DataBlockController.h"
#include "mpi.h"

class MultiCoreController : public QObject {
    Q_OBJECT
public:
    explicit MultiCoreController(QObject *parent = 0);
    ~MultiCoreController();

    void Init(int argc, char** argv);
    void Start();
signals:

public slots:
    void TrackForward_host();

private:
    DataBlockController *pDataBlockController;
    MPI_Comm MY_COMM_WORKER;
    MPI_Status status;

    int xs, ys, zs;
    int gID;
    int wID;
    int wGID;
    int timestep;
    int globalNumProcesses;
    int workerNumProcesses;

    Vector3d wSegXYZ;
    Vector3d wXYZ;

    Edge *pGlobalGraph;
    int globalGraphSize;

    CSVWriter   csv;
    DataSet     ds;

    void initVolumeData_both();
    void synchronizeParameters_both();
    void precalculateFirstStep_both();
    void waitForUIEvent();
    void trackForward_worker();

    Edge* updateGlobalConnectivityGraph(vector<Edge> localEdgesVector);
};

#endif // MULTICORECONTROLLER_H
