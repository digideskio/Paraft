#include "MultiCoreController.h"

MultiCoreController::MultiCoreController() {}
MultiCoreController::~MultiCoreController() {
    MPI_Finalize();
    pDataBlockController->~DataBlockController();
}

void MultiCoreController::Init(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gID);
    MPI_Comm_size(MPI_COMM_WORLD, &globalNumProcesses);

    int color = gID == HOST_NODE ? 0 : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, gID, &MY_COMM_WORKER);
    MPI_Comm_rank(MY_COMM_WORKER, &wID);
    MPI_Comm_size(MY_COMM_WORKER, &workerNumProcesses);

    wSegXYZ.x = atoi(argv[1]);
    wSegXYZ.y = atoi(argv[2]);
    wSegXYZ.z = atoi(argv[3]);

    int datasetID = atoi(argv[4]);
    if (datasetID == 0) {
        ds.index_start = 0;
        ds.index_end   = 10;
        ds.prefix      = "vorts";
        ds.surfix      = "data";
        ds.data_path   = "../../Data/vorts";
    } else if (datasetID == 1) {
        ds.index_start = 0;
        ds.index_end   = 7;
        ds.prefix      = "large_vorts_";
        ds.surfix      = "dat";
        ds.data_path   = "../../Data/vorts1";
    }

    wXYZ.z = wID / (wSegXYZ.x * wSegXYZ.y);
    wXYZ.y = (wID - wXYZ.z * wSegXYZ.x * wSegXYZ.y) / wSegXYZ.x;
    wXYZ.x = wID % wSegXYZ.x;

    csv.num_Seg = wSegXYZ;
    csv.num_worker = workerNumProcesses;
    csv.num_feature = 0;
    csv.time_1 = 0;
    csv.time_2 = 0;
    csv.time_3 = 0;
}

void MultiCoreController::Start() {
    initVolumeData();
    syncTFParameters();
    precalculateT0();

    if (gID == HOST_NODE) {
        for (int i = 0; i < NUM_TRACK_STEPS; i++ ){
            TrackForward();
        }
    } else { // slave nodes
        waitingForOrders();
    }
}

void MultiCoreController::TrackForward() {  // called by processor 0
    debug("TrackForward() start");

    timestep++;
    if (timestep > ds.index_end) {
        timestep = ds.index_end;
        debug("Already last timestep");
        return;
    }

    cout << "["<<wID<<"/"<<gID<<"]" << " |-- Time: " << timestep << endl;

    int router = MPI_TAG_SYNC_TIMESTEP;
    for (wGID = 1; wGID < globalNumProcesses; wGID++) {
        MPI_Ssend(&router, INT_SIZE, MPI_INT, wGID, MPI_TAG_ROUTER, MPI_COMM_WORLD);
        MPI_Ssend(&timestep, INT_SIZE, MPI_INT, wGID, MPI_TAG_SYNC_TIMESTEP, MPI_COMM_WORLD);
    }

    router = MPI_TAG_TRACK_FORWARD;
    for (wGID = 1; wGID < globalNumProcesses; wGID++) {
        MPI_Ssend(&router, INT_SIZE, MPI_INT, wGID, MPI_TAG_ROUTER, MPI_COMM_WORLD);
    }

    debug("TrackForward() end");
}

//// Member Function /////////////////////////////////////////////////////
void MultiCoreController::initVolumeData() {
    pDataBlockController = new DataBlockController();
    pDataBlockController->InitData(gID, wSegXYZ, wXYZ, ds);
    xs = pDataBlockController->GetVolumeDimX();
    ys = pDataBlockController->GetVolumeDimY();
    zs = pDataBlockController->GetVolumeDimZ();
    debug("Load volume data: " + ds.prefix + " ready");
}

void MultiCoreController::syncTFParameters() {
    int tfSize = TF_RESOLUTION * 4;         // float*rgba
    int bufSize = tfSize * FLOAT_SIZE;      // file size

    float* pTFColorMap = new float[tfSize];
    timestep = ds.index_start;

    string configFile = "tf_config.dat";
    if (gID == HOST_NODE) {
        ifstream inf(configFile.c_str(), ios::binary);
        if (!inf) { debug("Cannot read config file: " + configFile); }
        inf.read(reinterpret_cast<char *>(pTFColorMap), bufSize);
        inf.close();
    }

    MPI_Bcast(pTFColorMap, tfSize, MPI_FLOAT, HOST_NODE, MPI_COMM_WORLD);

    pDataBlockController->SetVolumeDataPointerByIndex(timestep);
    pDataBlockController->SetCurrentTimestep(timestep);
    pDataBlockController->SetTFResolution(TF_RESOLUTION);
    pDataBlockController->SetTFColorMap(pTFColorMap);
    debug("DataBlockController ready");
}

void MultiCoreController::precalculateT0() {
    timestep++;
    pDataBlockController->ExtractAllFeatures();
    pDataBlockController->SetCurrentTimestep(timestep);
    pDataBlockController->TrackForward();
    debug("Pre-calculate timestep 1 ready");
}

void MultiCoreController::waitingForOrders() {
    int router = MPI_TAG_NULL;
    while (true) {
        MPI_Recv(&router, INT_SIZE, MPI_INT, HOST_NODE, MPI_TAG_ROUTER,
                 MPI_COMM_WORLD, &status);
        switch (router) {
            case MPI_TAG_SYNC_TIMESTEP:
                MPI_Recv(&timestep, INT_SIZE, MPI_INT, HOST_NODE,
                         MPI_TAG_SYNC_TIMESTEP, MPI_COMM_WORLD, &status);
                break;
            case MPI_TAG_TRACK_FORWARD:
                trackForward_worker();
                break;
        default:
            debug("Internal Error: No matched tag found");
        }
    }
}

void MultiCoreController::trackForward_worker() {
    pDataBlockController->SetCurrentTimestep(timestep);
    MPI_Barrier(MY_COMM_WORKER);
    double t0 = MPI_Wtime();

    pDataBlockController->TrackForward();
    MPI_Barrier(MY_COMM_WORKER);
    double t1 = MPI_Wtime();

    pDataBlockController->UpdateLocalGraph(wID, wXYZ);
    MPI_Barrier(MY_COMM_WORKER);
    double t2 = MPI_Wtime();

    vector<Edge> localEdges = pDataBlockController->GetLocalEdges();
    updateGlobalConnectivityGraph(localEdges);
    MPI_Barrier(MY_COMM_WORKER);
    double t3 = MPI_Wtime();

    ///////////////////////////////////////////
    csv.time_1 = t1 - t0;
    csv.time_2 = t2 - t1;
    csv.time_3 = t3 - t2;
    csv.num_feature = globalGraphSize / 2;

    char result[1024];
    sprintf(result, "./Data/result.csv");

    ofstream outf(result, ios::out | ios::app);
    outf << csv.num_worker << "," << csv.num_feature << ","
         << csv.num_Seg.x << "," << csv.num_Seg.y << "," << csv.num_Seg.z << ","
         << csv.time_1 << "," << csv.time_2 << "," << csv.time_3 << endl;
    outf.close();

    debug("Done ----------------------");

    //// Test Graph ////////////////////////////////////////////////
    if (wID == 0) {
        for (int i = 0; i < globalGraphSize; i++) {
            cout << pGlobalGraph[i].id << pGlobalGraph[i].start << "->" << pGlobalGraph[i].end
                 << pGlobalGraph[i].centroid.x << pGlobalGraph[i].centroid.y << pGlobalGraph[i].centroid.z
                 << endl;
        }
    }
    ////////////////////////////////////////////////////////////////
}

Edge* MultiCoreController::updateGlobalConnectivityGraph(vector<Edge> localEdgesVector) {
    int localEdgeSize = localEdgesVector.size();
    Edge *localEdges = new Edge[localEdgeSize];

    for (int i = 0; i < localEdgeSize; i++) {
        localEdges[i] = localEdgesVector.at(i);
    }

    int globalGraphSizeSeg[workerNumProcesses];  // allgather result container
    MPI_Allgather(&localEdgeSize, 1, MPI_INT, globalGraphSizeSeg, 1, MPI_INT, MY_COMM_WORKER);

    globalGraphSize = 0;
    for (int i = 0; i < workerNumProcesses; i++) {
        globalGraphSize += globalGraphSizeSeg[i];
    }

    pGlobalGraph = new Edge[globalGraphSize];

    int displs[workerNumProcesses];
    displs[0] = 0;
    for (int i = 1; i < workerNumProcesses; i++) {
        displs[i] = globalGraphSizeSeg[i-1] + displs[i-1];
    }

    MPI_Datatype MPI_TYPE_EDGE;
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    MPI_Allgatherv(localEdges, localEdgeSize, MPI_TYPE_EDGE, pGlobalGraph,
                   globalGraphSizeSeg, displs, MPI_TYPE_EDGE, MY_COMM_WORKER);

    Edge ei, ej;
    for (int i = 0; i < globalGraphSize; i++) {
        ei = pGlobalGraph[i];
        if (ei.start > ei.end) {
            ei.start = pGlobalGraph[i].end;
            ei.end = pGlobalGraph[i].start;
        }
        for (int j = i+1; j < globalGraphSize; j++) {
            ej = pGlobalGraph[j];
            if (ej.start > ej.end) {
                ej.start = pGlobalGraph[j].end;
                ej.end = pGlobalGraph[j].start;
            }
            if (ei.start == ej.start && ei.end == ej.end) {
                if ((ei.centroid.x - ej.centroid.x) * (ei.centroid.x - ej.centroid.x) +
                    (ei.centroid.y - ej.centroid.y) * (ei.centroid.y - ej.centroid.y) +
                    (ei.centroid.z - ej.centroid.z) * (ei.centroid.z - ej.centroid.z) <= 4) {
                    if (ei.id < ej.id) {
                        pGlobalGraph[j].id = pGlobalGraph[i].id;
                    } else {
                        pGlobalGraph[i].id = pGlobalGraph[j].id;
                    }
                }
            }
        }
    }

    delete [] localEdges;
    return pGlobalGraph;
}

void MultiCoreController::debug(string msg) {
    cout << "[" << wID << "/" << gID << "] ";
    cout << msg << endl;
}

//// Member Function /////////////////////////////////////////////////////
