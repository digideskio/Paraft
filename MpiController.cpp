#include "MpiController.h"

MpiController::MpiController() {}
MpiController::~MpiController() {
    MPI_Finalize();
    pBlockController->~BlockController();
}

void MpiController::Init(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

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

    blockCoord.z = my_rank / (partition.x*partition.y);
    blockCoord.y = (my_rank - blockCoord.z*partition.x*partition.y) / partition.x;
    blockCoord.x = my_rank % partition.x;

    csv.partition = partition;
    csv.num_proc = num_proc;
    csv.num_feature = 0;
    csv.time_1 = 0;
    csv.time_2 = 0;
    csv.time_3 = 0;
}

void MpiController::Start() {
    initBlockController();
    initLocalCommGroup();
    initTFParameters();
    precalculateT0();

    for (int i = 0; i < NUM_TRACK_STEPS; ++i) {
        TrackForward();
    }
}

//// Member Function /////////////////////////////////////////////////////
void MpiController::initBlockController() {
    pBlockController = new BlockController();
    pBlockController->InitData(partition, blockCoord, dataset);
    debug("Load volume data: " + dataset.prefix + " ready");
}

void MpiController::initLocalCommGroup() {
    int ndims = 3;
    int *dims = partition.toArray();
    int *periods = Vector3i(1,1,1).toArray();
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &local_comm);

    debug("initLocalCommGroup ready");
}

void MpiController::initTFParameters() {
    int tfSize = TF_RESOLUTION * 4;         // float*rgba
    int bufSize = tfSize * FLOAT_SIZE;      // file size

    float* pTFColorMap = new float[tfSize];
    timestep = dataset.index_start;

    string configFile = "tf_config.dat";
    ifstream inf(configFile.c_str(), ios::binary);
    if (!inf) { debug("Cannot read config file: " + configFile); }
    inf.read(reinterpret_cast<char *>(pTFColorMap), bufSize);
    inf.close();

    pBlockController->SetVolumeDataPointerByIndex(timestep);
    pBlockController->SetCurrentTimestep(timestep);
    pBlockController->SetTFResolution(TF_RESOLUTION);
    pBlockController->SetTFColorMap(pTFColorMap);
    debug("initTFParameters ready");
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

    pBlockController->SetCurrentTimestep(timestep);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    pBlockController->TrackForward();
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    pBlockController->UpdateLocalGraph(my_rank, blockCoord);
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    vector<Edge> localEdges = pBlockController->GetLocalEdges();

    // option1: all gather and create a global graph
    vector<Edge> globalEdges = updateGlobalGraph(localEdges);

    // option2: gather adjacent to create feaure graph
//    updateFeatureGraph(localEdges);

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    ///////////////////////////////////////////
    csv.time_1 = t1 - t0;
    csv.time_2 = t2 - t1;
    csv.time_3 = t3 - t2;
    csv.num_feature = globalEdgeCount / 2;

    string result = "result.csv";
    ofstream outf(result.c_str(), ios::out | ios::app);
    outf << csv.num_proc << "," << csv.num_feature << ","
         << csv.partition.x << "," << csv.partition.y << "," << csv.partition.z << ","
         << csv.time_1 << "," << csv.time_2 << "," << csv.time_3 << endl;
    outf.close();

    debug("Done ----------------------");

    //// Test Graph ////////////////////////////////////////////////
    if (my_rank == 0) {
        cerr << "globalEdgeCount: " << globalEdgeCount << endl;
        for (int i = 0; i < globalEdgeCount; ++i) {
            cout << globalEdges[i].id << globalEdges[i].start << "->" << globalEdges[i].end
                 << globalEdges[i].centroid.x << globalEdges[i].centroid.y << globalEdges[i].centroid.z
                 << endl;
        }
    }
    ////////////////////////////////////////////////////////////////

    debug("TrackForward() done");
}

vector<Edge> MpiController::updateFeatureGraph(vector<Edge> localEdgesVector) {
    int localEdgeSize = localEdgesVector.size();

    vector<int> adjacentEdgeSizeVector(adjacentBlockCount);

    MPI_Allgather(&localEdgeSize, 1, MPI_INT,               // send (gather from)
                  &adjacentEdgeSizeVector[0], 1, MPI_INT,   // recv (gather to)
                  local_comm);                           // within adjacent blocks

    adjacentEdgeCount = accumulate(adjacentEdgeSizeVector.begin(),
                                   adjacentEdgeSizeVector.end(),0);

    vector<Edge> adjacentEdgeVector(adjacentEdgeCount);

    int displs[num_proc];
    displs[0] = 0;
    for (int i = 1; i < num_proc; ++i) {
        displs[i] = adjacentEdgeSizeVector[i-1] + displs[i-1];
    }

    MPI_Datatype MPI_TYPE_EDGE;
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    MPI_Allgatherv(&localEdgesVector[0], localEdgeSize, MPI_TYPE_EDGE, // send
                   &adjacentEdgeVector[0], &adjacentEdgeSizeVector[0], // recv
                   displs, MPI_TYPE_EDGE, local_comm);

    mergeCorrespondentEdges(adjacentEdgeVector);

    return adjacentEdgeVector;
}

vector<Edge> MpiController::updateGlobalGraph(vector<Edge> localEdgesVector) {
    int localEdgeSize = localEdgesVector.size();

    vector<int> globalGraphSizeVector(num_proc);

    MPI_Allgather(&localEdgeSize, 1, MPI_INT,               // send (gather from)
                  &globalGraphSizeVector[0], 1, MPI_INT,    // recv (gather to)
                  MPI_COMM_WORLD);                          // within non-host

    globalEdgeCount = accumulate(globalGraphSizeVector.begin(),
                                 globalGraphSizeVector.end(),0);

    vector<Edge> globalEdgeVector(globalEdgeCount);

    int displs[num_proc];
    displs[0] = 0;
    for (int i = 1; i < num_proc; ++i) {
        displs[i] = globalGraphSizeVector[i-1] + displs[i-1];
    }

    MPI_Datatype MPI_TYPE_EDGE;
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    MPI_Allgatherv(&localEdgesVector[0], localEdgeSize, MPI_TYPE_EDGE,
                   &globalEdgeVector[0], &globalGraphSizeVector[0],
                   displs, MPI_TYPE_EDGE, MPI_COMM_WORLD);

    mergeCorrespondentEdges(globalEdgeVector);

    return globalEdgeVector;
}

void MpiController::mergeCorrespondentEdges(vector<Edge> &edgeVector) {
    Edge ei, ej;
    for (unsigned int i = 0; i < edgeVector.size(); ++i) {
        ei = edgeVector[i];
        if (ei.start > ei.end) {
            ei.start = edgeVector[i].end;
            ei.end = edgeVector[i].start;
        }
        for (unsigned int j = i+1; j < edgeVector.size(); ++j) {
            ej = edgeVector[j];
            if (ej.start > ej.end) {    // always in ascending order
                ej.start = edgeVector[j].end;
                ej.end = edgeVector[j].start;
            }
            if (ei.start == ej.start && ei.end == ej.end) {
                if (ei.centroid.distanceFrom(ej.centroid) <= 4) {
                    if (ei.id < ej.id) {    // use the smaller id
                        edgeVector[j].id = edgeVector[i].id;
                    } else {
                        edgeVector[i].id = edgeVector[j].id;
                    }
                }
            }
        }
    }
}

void MpiController::debug(string msg) {
    cout << "[" << my_rank << "] ";
    cout << msg << endl;
}

//// Member Function /////////////////////////////////////////////////////
