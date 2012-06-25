#include "MpiController.h"

MpiController::MpiController() {}
MpiController::~MpiController() {
    MPI_Finalize();
    pBlockController->~BlockController();
}

void MpiController::Init(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &globalID);
    MPI_Comm_size(MPI_COMM_WORLD, &globalNumProc);

    int color = globalID == HOST_NODE ? 0 : 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, globalID, &workerCommunicator);
    MPI_Comm_rank(workerCommunicator, &blockID);
    MPI_Comm_size(workerCommunicator, &blockCount);

    partition.x = atoi(argv[1]);
    partition.y = atoi(argv[2]);
    partition.z = atoi(argv[3]);

    int datasetID = atoi(argv[4]);
    if (datasetID == 0) {
        dataset.index_start = 0;
        dataset.index_end   = 10;
        dataset.prefix      = "vorts";
        dataset.surfix      = "data";
        dataset.data_path   = "../Data/vorts";
    } else if (datasetID == 1) {
        dataset.index_start = 0;
        dataset.index_end   = 7;
        dataset.prefix      = "large_vorts_";
        dataset.surfix      = "dat";
        dataset.data_path   = "../Data/vorts1";
    }

    blockCoord.z = blockID / (partition.x*partition.y);
    blockCoord.y = (blockID - blockCoord.z*partition.x*partition.y) / partition.x;
    blockCoord.x = blockID % partition.x;

    csv.partition = partition;
    csv.num_worker = blockCount;
    csv.num_feature = 0;
    csv.time_1 = 0;
    csv.time_2 = 0;
    csv.time_3 = 0;
}

void MpiController::Start() {
    initBlockController();
    initLocalCommunicator();
    syncTFParameters();
    precalculateT0();

    if (globalID == HOST_NODE) {
        for (int i = 0; i < NUM_TRACK_STEPS; ++i) {
            TrackForward();
        }
    } else { // slave nodes
        waitingForOrders();
    }
}

//// Member Function /////////////////////////////////////////////////////
void MpiController::initBlockController() {
    pBlockController = new BlockController();
    pBlockController->InitData(globalID, partition, blockCoord, dataset);
    debug("Load volume data: " + dataset.prefix + " ready");
}

////
void MpiController::initLocalCommunicator() {
    IntMap adjacentBlocks = pBlockController->GetAdjacentBlocks();

    vector<int> neighbors;
    for (int i = 0; i < adjacentBlocks.size(); ++i) {
        if (adjacentBlocks[i] != -1) {  // -1: no neighbor in this direction
            neighbors.push_back(adjacentBlocks[i]);
        }
    }
    neighbors.push_back(blockID);   // add self to adjacentCommunicator;

    MPI_Group workerGroup, adjacentGroup;
    MPI_Comm_group(workerCommunicator, &workerGroup);
    MPI_Group_incl(workerGroup, neighbors.size(), &neighbors[0], &adjacentGroup);
    MPI_Comm_create(workerCommunicator, adjacentGroup, &adjacentCommunicator);
    MPI_Group_free(&workerGroup);
    MPI_Group_free(&adjacentGroup);
}
////

void MpiController::syncTFParameters() {
    int tfSize = TF_RESOLUTION * 4;         // float*rgba
    int bufSize = tfSize * FLOAT_SIZE;      // file size

    float* pTFColorMap = new float[tfSize];
    timestep = dataset.index_start;

    string configFile = "tf_config.dat";
    if (globalID == HOST_NODE) { // host read in, then broadcast to others
        ifstream inf(configFile.c_str(), ios::binary);
        if (!inf) { debug("Cannot read config file: " + configFile); }
        inf.read(reinterpret_cast<char *>(pTFColorMap), bufSize);
        inf.close();
    }

    MPI_Bcast(pTFColorMap, tfSize, MPI_FLOAT, HOST_NODE, MPI_COMM_WORLD);

    pBlockController->SetVolumeDataPointerByIndex(timestep);
    pBlockController->SetCurrentTimestep(timestep);
    pBlockController->SetTFResolution(TF_RESOLUTION);
    pBlockController->SetTFColorMap(pTFColorMap);
    debug("DataBlockController ready");
}

void MpiController::precalculateT0() {
    timestep++; // all nodes
    pBlockController->ExtractAllFeatures();
    pBlockController->SetCurrentTimestep(timestep);
    pBlockController->TrackForward();
    debug("Pre-calculate timestep 1 ready");
}

void MpiController::TrackForward() {  // triggered by host
    debug("TrackForward() triggered");

    timestep++;
    if (timestep > dataset.index_end) {
        timestep = dataset.index_end;
        debug("Already last timestep");
        return;
    }

    cout << "|-- Current timestep: " << timestep << endl;

    int router = MPI_TAG_SYNC_TIMESTEP;
    for (blockGID = 1; blockGID < globalNumProc; ++blockGID) {
        MPI_Ssend(&router, INT_SIZE, MPI_INT, blockGID, MPI_TAG_ROUTER, MPI_COMM_WORLD);
        MPI_Ssend(&timestep, INT_SIZE, MPI_INT, blockGID, MPI_TAG_SYNC_TIMESTEP, MPI_COMM_WORLD);
    }

    router = MPI_TAG_TRACK_FORWARD;
    for (blockGID = 1; blockGID < globalNumProc; ++blockGID) {
        MPI_Ssend(&router, INT_SIZE, MPI_INT, blockGID, MPI_TAG_ROUTER, MPI_COMM_WORLD);
    }

    debug("TrackForward() done");
}

void MpiController::waitingForOrders() {
    int router = MPI_TAG_NULL;
    while (true) {
        MPI_Recv(&router, INT_SIZE, MPI_INT, HOST_NODE, MPI_TAG_ROUTER, MPI_COMM_WORLD, &status);
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

void MpiController::trackForward_worker() {
    pBlockController->SetCurrentTimestep(timestep);
    MPI_Barrier(workerCommunicator);
    double t0 = MPI_Wtime();

    pBlockController->TrackForward();
    MPI_Barrier(workerCommunicator);
    double t1 = MPI_Wtime();

    pBlockController->UpdateLocalGraph(blockID, blockCoord);
    MPI_Barrier(workerCommunicator);
    double t2 = MPI_Wtime();

    vector<Edge> localEdges = pBlockController->GetLocalEdges();

    // option1: all gather and create a global graph
    updateGlobalGraph(localEdges);

    // option2: gather adjacent to create feaure graph
    updateFeatureGraph(localEdges);

    MPI_Barrier(workerCommunicator);
    double t3 = MPI_Wtime();

    ///////////////////////////////////////////
    csv.time_1 = t1 - t0;
    csv.time_2 = t2 - t1;
    csv.time_3 = t3 - t2;
    csv.num_feature = globalGraphSize / 2;

    string result = "result.csv";
    ofstream outf(result.c_str(), ios::out | ios::app);
    outf << csv.num_worker << "," << csv.num_feature << ","
         << csv.partition.x << "," << csv.partition.y << "," << csv.partition.z << ","
         << csv.time_1 << "," << csv.time_2 << "," << csv.time_3 << endl;
    outf.close();

    debug("Done ----------------------");

    //// Test Graph ////////////////////////////////////////////////
    if (blockID == 0) {
        cerr << "globalGraphSize: " << globalGraphSize << endl;
        for (int i = 0; i < globalGraphSize; ++i) {
            cout << pGlobalGraph[i].id << pGlobalGraph[i].start << "->" << pGlobalGraph[i].end
                 << pGlobalGraph[i].centroid.x << pGlobalGraph[i].centroid.y << pGlobalGraph[i].centroid.z
                 << endl;
        }
    }
    ////////////////////////////////////////////////////////////////
}

Edge* MpiController::updateFeatureGraph(vector<Edge> localEdgesVector) {

}

Edge* MpiController::updateGlobalGraph(vector<Edge> localEdgesVector) {
    int localEdgeSize = localEdgesVector.size();
    Edge *localEdges = new Edge[localEdgeSize];

    for (int i = 0; i < localEdgeSize; ++i) {
        localEdges[i] = localEdgesVector.at(i);
    }

    int globalGraphSizeSeg[blockCount];  // allgather result container
    MPI_Allgather(&localEdgeSize, 1, MPI_INT, globalGraphSizeSeg, 1, MPI_INT, workerCommunicator);

    globalGraphSize = 0;
    for (int i = 0; i < blockCount; ++i) {
        globalGraphSize += globalGraphSizeSeg[i];
    }

    pGlobalGraph = new Edge[globalGraphSize];

    int displs[blockCount];
    displs[0] = 0;
    for (int i = 1; i < blockCount; ++i) {
        displs[i] = globalGraphSizeSeg[i-1] + displs[i-1];
    }

    MPI_Datatype MPI_TYPE_EDGE;
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    MPI_Allgatherv(localEdges, localEdgeSize, MPI_TYPE_EDGE, pGlobalGraph,
                   globalGraphSizeSeg, displs, MPI_TYPE_EDGE, workerCommunicator);

    Edge ei, ej;
    for (int i = 0; i < globalGraphSize; ++i) {
        ei = pGlobalGraph[i];
        if (ei.start > ei.end) {
            ei.start = pGlobalGraph[i].end;
            ei.end = pGlobalGraph[i].start;
        }
        for (int j = i+1; j < globalGraphSize; ++j) {
            ej = pGlobalGraph[j];
            if (ej.start > ej.end) {    // always in ascending order
                ej.start = pGlobalGraph[j].end;
                ej.end = pGlobalGraph[j].start;
            }
            if (ei.start == ej.start && ei.end == ej.end) {
                if (ei.centroid.distanceFrom(ej.centroid) <= 4) {
                    if (ei.id < ej.id) {    // use the smaller id
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

void MpiController::debug(string msg) {
    cout << "[" << blockID << "/" << globalID << "] ";
    cout << msg << endl;
}

//// Member Function /////////////////////////////////////////////////////
