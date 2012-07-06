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

    // init MPI_TYPE_EDGE
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    partition.x = atoi(argv[1]);
    partition.y = atoi(argv[2]);
    partition.z = atoi(argv[3]);

    int datasetID = atoi(argv[4]);
    if (datasetID == 0) {
        ds.index_start = 0;
        ds.index_end   = 10;
        ds.prefix      = "vorts";
        ds.surfix      = "data";
        ds.data_path   = "../Data/vorts";
    } else if (datasetID == 1) {
        ds.index_start = 0;
        ds.index_end   = 7;
        ds.prefix      = "large_vorts_";
        ds.surfix      = "dat";
        ds.data_path   = "../Data/vorts1";
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
    pBlockController->InitData(partition, blockCoord, ds);
//    debug("Load volume data: " + ds.prefix + " ready");
}

void MpiController::initLocalCommGroup() {
//    int ndims = 3;
//    int *dims = partition.toArray();
//    int *periods = Vector3i(1,1,1).toArray();
//    int reorder = 0;
//    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &local_comm);
//    MPI_Comm_size(local_comm, &adjacentBlockCount);

    adjacentIndices = pBlockController->GetAdjacentBlocksIndices();
//    adjacentIndices.push_back(my_rank);

    adjacentBlockCount = adjacentIndices.size();

    MPI_Group world_group, local_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, adjacentBlockCount, &adjacentIndices[0], &local_group);

    MPI_Comm_create(MPI_COMM_WORLD, local_group, &local_comm);

//    int size = 0;
//    MPI_Comm_size(local_comm_1, &size);
////    MPI_Comm_size(local_comm, &size);

//    cerr << "adjacentBlockCount: " << adjacentBlockCount << " size: " << size << endl;

//    for (unsigned int i = 0; i < adjacentBlocks.size(); i++) {
//        cerr << my_rank << " adjacentBlocks[i]: " << adjacentBlocks[i] << endl;
//        if (my_rank == adjacentBlocks[i]) {
//            color = 0;
//            cerr << my_rank << "'s color is 1." << endl;
//            break;
//        }
//    }

//    MPI_Comm_split(MPI_COMM_WORLD, color, my_rank, &local_comm);

//    int size = 0;
//    MPI_Comm_size(local_comm, &size);
//    cerr << "[" << my_rank << "] local_comm.size: " << size << endl;


//    adjacentBlockCount = pBlockController->GetAdjacentBlocksIndices().size();
//    debug("initLocalCommGroup ready");
}

void MpiController::initTFParameters() {
    int tfSize = TF_RESOLUTION * 4;         // float*rgba
    int bufSize = tfSize * FLOAT_SIZE;      // file size

    float* pTFColorMap = new float[tfSize];
    timestep = ds.index_start;

    string configFile = "tf_config.dat";
    ifstream inf(configFile.c_str(), ios::binary);
    if (!inf) { debug("Cannot read config file: " + configFile); }
    inf.read(reinterpret_cast<char *>(pTFColorMap), bufSize);
    inf.close();

    pBlockController->SetVolumeDataPointerByIndex(timestep);
    pBlockController->SetCurrentTimestep(timestep);
    pBlockController->SetTFResolution(TF_RESOLUTION);
    pBlockController->SetTFColorMap(pTFColorMap);
//    debug("initTFParameters ready");
}

void MpiController::precalculateT0() {
    timestep++; // all nodes
    pBlockController->ExtractAllFeatures();
    pBlockController->SetCurrentTimestep(timestep);
    pBlockController->TrackForward();
//    debug("Pre-calculate timestep 1 ready");
}

void MpiController::TrackForward() {  // triggered by host
    debug("TrackForward() start");

    timestep++;
    if (timestep > ds.index_end) {
        timestep = ds.index_end;
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
    vector<Edge> edges = updateGlobalGraph(localEdges);

    // option2: gather adjacent to create feaure graph
//    vector<Edge> edges = updateFeatureGraph(localEdges);

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

    //// Test Graph ////////////////////////////////////////////////
    if (my_rank == 0) {
        cerr << "Edge count: " << edges.size() << endl;
        for (unsigned int i = 0; i < edges.size(); ++i) {
            cout << edges[i].id << " : " << edges[i].start << "->" << edges[i].end
                 << edges[i].centroid.x << edges[i].centroid.y << edges[i].centroid.z
                 << endl;
        }
    }
    ////////////////////////////////////////////////////////////////

    debug("Done ----------------------");
}

vector<Edge> MpiController::updateFeatureGraph(vector<Edge> localEdgeVector) {
    int localEdgeCount = localEdgeVector.size();

    vector<Edge> adjacentEdgeVector;    // resize later

    for (int i = 0; i < adjacentBlockCount; i++) {
        int dest = adjacentIndices[i];
        int edgeCount = 0;

        MPI_Irecv(&edgeCount, 1, MPI_INT, dest, 100, MPI_COMM_WORLD, &request);
        MPI_Send(&localEdgeCount, 1, MPI_INT, dest, 100, MPI_COMM_WORLD);
        MPI_Wait(&request, &status);

        if (edgeCount == 0) {
            continue;
        }

        Edge *adjacentEdges = new Edge[edgeCount];
        MPI_Irecv(adjacentEdges, edgeCount, MPI_TYPE_EDGE, dest, 101, MPI_COMM_WORLD, &request);
        MPI_Send(&localEdgeVector[0], localEdgeCount, MPI_TYPE_EDGE, dest, 101, MPI_COMM_WORLD);
        MPI_Wait(&request, &status);

        for (int i = 0; i < edgeCount; i++) {
            adjacentEdgeVector.push_back(adjacentEdges[i]);
        }
    }

    adjacentEdgeVector.insert(adjacentEdgeVector.end(), localEdgeVector.begin(),
                              localEdgeVector.end());

    mergeCorrespondentEdges(adjacentEdgeVector);

    return adjacentEdgeVector;
}

vector<Edge> MpiController::updateGlobalGraph(vector<Edge> localEdgeVector) {
    int localEdgeCount = localEdgeVector.size();

    vector<int> globalEdgeCountVector(num_proc);

    MPI_Allreduce(&localEdgeCount, &globalEdgeCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    vector<Edge> globalEdgeVector(globalEdgeCount);

    int displs[num_proc];
    displs[0] = 0;
    for (int i = 1; i < num_proc; ++i) {
        displs[i] = globalEdgeCountVector[i-1] + displs[i-1];
        cerr << "displs[" << i << "] = " << displs[i] << " = "
             << globalEdgeCountVector[i-1] << " + " << displs[i-1] << endl;
    }

    MPI_Allgatherv(&localEdgeVector[0], localEdgeCount, MPI_TYPE_EDGE,
                   &globalEdgeVector[0], &globalEdgeCountVector[0],
                   displs, MPI_TYPE_EDGE, MPI_COMM_WORLD);

    for (unsigned int i = 0; i < localEdgeVector.size(); i++) {
        cerr << localEdgeVector[i].id << endl;
    }

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
