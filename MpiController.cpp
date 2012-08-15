#include "MpiController.h"

MpiController::MpiController() {}

MpiController::~MpiController() {
    MPI_Finalize();
    pBlockController->~BlockController();
}

void MpiController::InitWith(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    // init MPI_TYPE_EDGE
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    partition.x = atoi(argv[1]);
    partition.y = atoi(argv[2]);
    partition.z = atoi(argv[3]);

    blockCoord.z = my_rank/(partition.x*partition.y);
    blockCoord.y = (my_rank-blockCoord.z*partition.x*partition.y)/partition.x;
    blockCoord.x = my_rank%partition.x;

    ds.index_start = 1;
    ds.index_end   = 10;
    ds.prefix      = "vorts";
    ds.surfix      = "data";
    ds.data_path   = "../Data/vorts";

    csv.partition = partition;
    csv.num_proc = num_proc;
    csv.num_feature = 0;
    csv.time_1 = 0;
    csv.time_2 = 0;
    csv.time_3 = 0;
}

void MpiController::Start() {
    initBlockController();
    initTFParameters();
    precalculateTimestep1();

    for (int i = 0; i < NUM_TRACK_STEPS; ++i) {
        TrackForward();
    }

    cout << my_rank << " done." << endl;
}

void MpiController::initBlockController() {
    pBlockController = new BlockController();
    pBlockController->InitData(partition, blockCoord, ds);
//    debug("Load volume data: " + ds.prefix + " ready");
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

void MpiController::precalculateTimestep1() {
    pBlockController->ExtractAllFeatures();
    pBlockController->SetCurrentTimestep(timestep);
    pBlockController->TrackForward();
//    debug("Pre-calculate timestep 1 ready");
}

void MpiController::TrackForward() {  // triggered by host
//    debug("TrackForward() start");

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

    featureTable.clear();

    // option1: all gather and create a global graph
//    gatherGlobalGraph();

    // option2: gather adjacent to create feaure graph;
    // initially sync with all adjacent blocks, when one finished syncing,
    // delete it from the adjacentBlocks list
    adjacentBlocks = pBlockController->GetAdjacentBlocks();
    need_to_send = true;
    need_to_recv = true;

    while (need_to_send || need_to_recv) {
        syncFeatureGraph();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    featureTableVector[timestep] = featureTable;

    csv.time_1 = t1 - t0;
    csv.time_2 = t2 - t1;
    csv.time_3 = t3 - t2;
    csv.num_feature = globalEdgeCount / 2;

    string result = "result.csv";
    ofstream outf(result.c_str(), ios::out | ios::app);
    outf << my_rank << "," << csv.num_proc << "," << csv.num_feature << ","
         << csv.partition.x << ","
         << csv.partition.y << ","
         << csv.partition.z << ","
         << csv.time_1 << "," << csv.time_2 << "," << csv.time_3 << endl;
    outf.close();
}

void MpiController::gatherGlobalGraph() {
    vector<Edge> localEdges = pBlockController->GetLocalGraph();
    int localEdgeCount = localEdges.size();

    vector<int> globalEdgeCountVector(num_proc);

    MPI_Allgather(&localEdgeCount, 1, MPI_INT,
                  &globalEdgeCountVector.front(), 1, MPI_INT, MPI_COMM_WORLD);

    globalEdgeCount = 0;
    for (uint i = 0; i < globalEdgeCountVector.size(); i++) {
        globalEdgeCount += globalEdgeCountVector[i];
    }

    vector<Edge> globalEdges(globalEdgeCount);

    int displs[num_proc];
    displs[0] = 0;
    for (int i = 1; i < num_proc; i++) {
        displs[i] = globalEdgeCountVector[i-1] + displs[i-1];
    }

    MPI_Allgatherv(&localEdges.front(), localEdgeCount, MPI_TYPE_EDGE,
                   &globalEdges.front(), &globalEdgeCountVector.front(),
                   displs, MPI_TYPE_EDGE, MPI_COMM_WORLD);

    mergeCorrespondentEdges(globalEdges);
}

void MpiController::syncFeatureGraph() {
    vector<Edge> localEdges = pBlockController->GetLocalGraph();
    int localEdgeCount = localEdges.size();

    vector<int> blocksToSync(num_proc);

    int blockToSync = need_to_recv ? my_rank : -1;
    MPI_Allgather(&blockToSync, 1, MPI_INT, &blocksToSync.front(), 1,
                  MPI_INT, MPI_COMM_WORLD);

    vector<int> adjacentBlocksToSync;

    sort(adjacentBlocks.begin(), adjacentBlocks.end());
    sort(blocksToSync.begin(), blocksToSync.end());

    set_intersection(adjacentBlocks.begin(), adjacentBlocks.end(),
                     blocksToSync.begin(), blocksToSync.end(),
                     back_inserter(adjacentBlocksToSync));

    if (my_rank == 0) {
        cout << "\n+ adjacentBlocks: ";
        for (uint i = 0; i < adjacentBlocks.size(); i++) {
            cout << adjacentBlocks[i] << " ";
        }
        cout << endl;

        cout << "+ blocksToSync: ";
        for (uint i = 0; i < blocksToSync.size(); i++) {
            cout << blocksToSync[i] << " ";
        }
        cout << endl;

        cout << "+ adjacentBlocksToSync: ";
        for (uint i = 0; i < adjacentBlocksToSync.size(); i++) {
            cout << adjacentBlocksToSync[i] << " ";
        }
        cout << endl;
    }


    vector<Edge> adjacentGraph;

    for (uint i = 0; i < adjacentBlocksToSync.size(); i++) {
        int dest = adjacentBlocksToSync[i];

        if (need_to_send) {
            // 1. send count
            MPI_Send(&localEdgeCount, 1, MPI_INT, dest, 100, MPI_COMM_WORLD);

            // 2. send content
            if (localEdgeCount != 0) {
                MPI_Send(&localEdges.front(), localEdgeCount, MPI_TYPE_EDGE,
                         dest, 101, MPI_COMM_WORLD);
            }
        }

        if (need_to_recv) {
            // 1. recv count
            int destEdgeCount = 0;
            MPI_Recv(&destEdgeCount, 1, MPI_INT, dest, 100, MPI_COMM_WORLD,
                     &status);

            // 2. recv content
            if (destEdgeCount != 0) {
                vector<Edge> destEdges(destEdgeCount);
                MPI_Recv(&destEdges.front(), destEdgeCount, MPI_TYPE_EDGE,
                         dest, 101, MPI_COMM_WORLD, &status);

                for (int i = 0; i < destEdgeCount; i++) {
                    bool isNew = true;
                    for (uint j = 0; j < adjacentGraph.size(); j++) {
                        if (destEdges[i] == adjacentGraph[j]) {
                            isNew = false; break;
                        }
                    }
                    if (isNew) {
                        adjacentGraph.push_back(destEdges[i]);
                    }
                }
            }
        }
    }

    adjacentBlocks = adjacentBlocksToSync;
    need_to_send = adjacentBlocks.size() > 0 ? true : false;

    // add local edges
    for (uint i = 0; i < localEdges.size(); i++) {
        bool isNew = true;
        for (uint j = 0; j < adjacentGraph.size(); j++) {
            if (localEdges[i] == adjacentGraph[j]) {
                isNew = false; break;
            }
        }
        if (isNew) {
            adjacentGraph.push_back(localEdges[i]);
        }
    }

    mergeCorrespondentEdges(adjacentGraph);
    pBlockController->SetLocalGraph(adjacentGraph);
}

void MpiController::mergeCorrespondentEdges(vector<Edge> edges) {
    need_to_recv = false;

    for (uint i = 0; i < edges.size(); i++) {
        Edge ei = edges[i];
        for (uint j = i+1; j < edges.size(); j++) {
            Edge ej = edges[j];

            // sync the id of feature if two matches
            if (ei.start == ej.end && ei.end == ej.start &&  // 0->1 | 1->0
                (ei.start == my_rank || ei.end == my_rank) &&
                ei.centroid.distanceFrom(ej.centroid) <= DIST_THRESHOLD) {
                if (ei.id < ej.id) {    // use the smaller id
                    edges[j].id = ei.id;
                } else {
                    edges[i].id = ej.id;
                }
                updateFeatureTable(ei);
            }
        }
    }

    // if either start or end equals to my_rank, add to featureTable
    for (uint i = 0; i < edges.size(); i++) {
        Edge edge = edges[i];
        if (edge.start == my_rank || edge.end == my_rank) {
            updateFeatureTable(edge);
        }
    }

    // if both start and end are not equal to my_rank,
    // but the id is already in the feature table, update featureTable
    for (uint i = 0; i < edges.size(); i++) {
        Edge edge = edges[i];
        if (edge.start != my_rank || edge.end != my_rank) {
            if (featureTable.find(edge.id) != featureTable.end()) {
                updateFeatureTable(edge);
            }
        }
    }

    if (my_rank == 0) {
        cerr << "-----------------------------------" << endl;
        FeatureTable::iterator it;
        for (it = featureTable.begin(); it != featureTable.end(); it++) {
            int id = it->first;
            cerr << id << ": ( ";
            vector<int> value = it->second;
            for (uint i = 0; i < value.size(); i++) {
                cerr << value[i] << " ";
            }
            cerr << ")" << endl;
        }
    }
}

void MpiController::updateFeatureTable(Edge edge) {
    if (featureTable.find(edge.id) == featureTable.end()) {
        vector<int> value;
        value.push_back(edge.start);
        value.push_back(edge.end);
        featureTable[edge.id] = value;
        need_to_recv = true;
    } else {
        vector<int> value = featureTable[edge.id];
        if (find(value.begin(), value.end(), edge.start) == value.end()) {
            value.push_back(edge.start);
            need_to_recv = true;
        }

        if (find(value.begin(), value.end(), edge.end) == value.end()) {
            value.push_back(edge.end);
            need_to_recv = true;
        }

        featureTable[edge.id] = value;
    }
}

void MpiController::debug(string msg) {
    cout << "[" << my_rank << "] ";
    cout << msg << endl;
}
