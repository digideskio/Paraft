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

    ds.start    = 99;
    ds.end      = 150;
    ds.prefix   = "vort_";
    ds.surfix   = ".raw";
    ds.path     = "/Users/Yang/Develop/Data/yubo_new/vorts/";
    ds.tf       = "config.tfe";
    ds.dim      = Vector3i(256, 256, 256);

    csv.partition = partition;
    csv.num_proc = num_proc;
    csv.num_feature = 0;
    csv.time_1 = 0;
    csv.time_2 = 0;
    csv.time_3 = 0;
}

void MpiController::Start() {
    initBlockController();

    for (int i = 1; i < ds.end-ds.start; i++) {
        TrackForward();
        cout << "["<<my_rank<<"] T" << ds.start+timestep+1 << " done." << endl;
    }

    cout << my_rank << " over." << endl;
}

void MpiController::initBlockController() {
    timestep = 0;   // in fact ds.start
    pBlockController = new BlockController();
    pBlockController->InitData(partition, blockCoord, ds);
    pBlockController->SetCurrentTimestep(timestep);
    pBlockController->SetVolumeDataPointerByIndex(timestep);
    pBlockController->ExtractAllFeatures();
}

void MpiController::TrackForward() {
    timestep++;
    if (timestep > ds.end) {
        timestep = ds.end;
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
    need_to_send = 1;
    need_to_recv = 1;
    any_send = 1;
    any_recv = 1;

    while (any_send || any_recv) {
        syncFeatureGraph();
        if (my_rank == 0) cout << "-----------" << endl;
    }

    // option3: gather adjacent to create feaure graph;
//    gatherNeighboringGraph();

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    featureTableVector[timestep] = featureTable;

    double delta = t1 - t0;
    MPI_Allreduce(&delta, &csv.time_1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delta = t2 - t1;
    MPI_Allreduce(&delta, &csv.time_2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delta = t3 - t2;
    MPI_Allreduce(&delta, &csv.time_3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    csv.time_1 /= (double)num_proc;
    csv.time_2 /= (double)num_proc;
    csv.time_3 /= (double)num_proc;

    csv.num_feature = featureTable.size();

    char np[21];
    sprintf(np, "%d", num_proc);
    string fpath = "";
    string prefix = "result_";
    string surfix = ".csv";
    string result = fpath.append(prefix).append(np).append(surfix);
    ofstream outf(result.c_str(), ios::out | ios::app);

    if (my_rank == 0) {
        outf << csv.num_proc << "," << csv.num_feature << "," << timestep+ds.start << ","
             << csv.time_1 << "," << csv.time_2 << "," << csv.time_3 << endl;
    }

    outf.close();
}

void MpiController::syncFeatureGraph() {
    vector<Edge> localEdges = pBlockController->GetLocalGraph();
    int localEdgeCount = localEdges.size();

    vector<int> blocksNeedRecv(num_proc);
    for (uint i = 0; i < num_proc; i++) {
        blocksNeedRecv.at(i) = -1;
    }

    int recv_id = need_to_recv ? my_rank : -1;

    MPI_Allgather(&recv_id, 1, MPI_INT, &blocksNeedRecv.front(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> adjacentBlocksNeedRecv;

    sort(adjacentBlocks.begin(), adjacentBlocks.end());
    sort(blocksNeedRecv.begin(), blocksNeedRecv.end());

    set_intersection(adjacentBlocks.begin(), adjacentBlocks.end(),
                     blocksNeedRecv.begin(), blocksNeedRecv.end(),
                     back_inserter(adjacentBlocksNeedRecv));

    // -- debug ---------
//    cout << "\n+[" << my_rank << "] adjacentBlocks: ";
//    for (uint i = 0; i < adjacentBlocks.size(); i++) {
//        cout << adjacentBlocks[i] << " ";
//    } cout << endl;

//    cout << "+[" << my_rank << "] blocksNeedRecv: ";
//    for (uint i = 0; i < blocksNeedRecv.size(); i++) {
//        cout << blocksNeedRecv[i] << " ";
//    } cout << endl;

//    cout << "+[" << my_rank << "] adjacentBlocksNeedRecv: ";
//    for (uint i = 0; i < adjacentBlocksNeedRecv.size(); i++) {
//        cout << adjacentBlocksNeedRecv[i] << " ";
//    } cout << endl;
    // -- debug ---------

    need_to_send = adjacentBlocksNeedRecv.size() > 0 ? 1 : 0;

    // -- debug ---------
//    if (need_to_send) {
//        cout << "[" << my_rank << "] need_to_send yes ";
//        for (uint i = 0; i < adjacentBlocksNeedRecv.size(); i++) {
//            cout << adjacentBlocksNeedRecv.at(i) << " ";
//        }
//        cout << endl;
//    } else {
//        debug("need_to_send no");
//    }

//    if (need_to_recv) {
//        cout << "[" << my_rank << "] need_to_recv yes " << endl;
//    } else {
//        debug("need_to_recv no");
//    }
    // -- debug ---------

    vector<Edge> adjacentGraph;

    if (need_to_send) {
        for (uint i = 0; i < adjacentBlocksNeedRecv.size(); i++) {
            int dest = adjacentBlocksNeedRecv.at(i);
            // 1. send count
            MPI_Send(&localEdgeCount, 1, MPI_INT, dest, 100, MPI_COMM_WORLD);
            // 2. send content
            if (localEdgeCount != 0) {
                MPI_Send(&localEdges.front(), localEdgeCount, MPI_TYPE_EDGE, dest, 101, MPI_COMM_WORLD);
            }
        }
    }

    if (need_to_recv) {
        for (uint i = 0; i < adjacentBlocks.size(); i++) {
            int src = adjacentBlocks.at(i);
            int srcEdgeCount = 0;
            // 1. recv count
            MPI_Recv(&srcEdgeCount, 1, MPI_INT, src, 100, MPI_COMM_WORLD, &status);
            // 2. recv content
            if (srcEdgeCount != 0) {
                vector<Edge> srcEdges(srcEdgeCount);
                MPI_Recv(&srcEdges.front(), srcEdgeCount, MPI_TYPE_EDGE, src, 101, MPI_COMM_WORLD, &status);

                for (int i = 0; i < srcEdgeCount; i++) {
                    bool isNew = true;
                    for (uint j = 0; j < adjacentGraph.size(); j++) {
                        if (srcEdges[i] == adjacentGraph[j]) {
                            isNew = false; break;
                        }
                    }
                    if (isNew) {
                        adjacentGraph.push_back(srcEdges[i]);
                    }
                }
            }
        }
        need_to_recv = 0;
    }

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

    any_send = -1;
    any_recv = -1;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&need_to_send, &any_send, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&need_to_recv, &any_recv, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

//    if (need_to_send) {
//        cout << "[" << my_rank << "] need_to_send yes ";
//        for (uint i = 0; i < adjacentBlocksNeedRecv.size(); i++) {
//            cout << adjacentBlocksNeedRecv.at(i) << " ";
//        }
//        cout << endl;
//    } else {
//        debug("need_to_send no");
//    }

//    if (need_to_recv) {
//        cout << "[" << my_rank << "] need_to_recv yes " << endl;
//    } else {
//        debug("need_to_recv no");
//    }
}

void MpiController::gatherNeighboringGraph() {
    vector<Edge> localEdges = pBlockController->GetLocalGraph();
    int localEdgeCount = localEdges.size();
//    int neighborEdgeCount = 0;

//    cout << "MPI_Allgather start." << endl;
//    MPI_Allreduce(&localEdgeCount, &neighborEdgeCount, 1, MPI_INT, MPI_SUM, MPI_COMM_LOCAL);
//    cout << "neighboringEdgeCount: " << neighborEdgeCount << endl;
//    cout << "MPI_Allgather end." << endl;

    vector<int> neighboringEdgeCountVector(3);

    cerr << my_rank << ": MPI_Allgather start." << endl;
    MPI_Allgather(&localEdgeCount, 1, MPI_INT, &neighboringEdgeCountVector.front(),
                  1, MPI_INT, MPI_COMM_LOCAL);
    cerr << my_rank << ": MPI_Allgather over." << endl;

    int neighborEdgeCount = 0;
    for (uint i = 0; i < neighboringEdgeCountVector.size(); i++) {
        neighborEdgeCount += neighboringEdgeCountVector[i];
    }

    cout << "neighboringEdgeCount: " << neighborEdgeCount << endl;
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

void MpiController::mergeCorrespondentEdges(vector<Edge> edges) {
//    need_to_recv = false;

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

    if (my_rank == 0) {     // debug log
        FeatureTable::iterator it;
        for (it = featureTable.begin(); it != featureTable.end(); it++) {
            int id = it->first;
            cout << "[" << my_rank << "]" << id << ": ( ";
            vector<int> value = it->second;
            for (uint i = 0; i < value.size(); i++) {
                cout << value[i] << " ";
            }
            cout << ")" << endl;
        }
    }
}

void MpiController::updateFeatureTable(Edge edge) {
    if (featureTable.find(edge.id) == featureTable.end()) {
        vector<int> value;
        value.push_back(edge.start);
        value.push_back(edge.end);
        featureTable[edge.id] = value;
        need_to_recv = 1;
    } else {
        vector<int> value = featureTable[edge.id];
        if (find(value.begin(), value.end(), edge.start) == value.end()) {
            value.push_back(edge.start);
            need_to_recv = 1;
        }

        if (find(value.begin(), value.end(), edge.end) == value.end()) {
            value.push_back(edge.end);
            need_to_recv = 1;
        }

        featureTable[edge.id] = value;
    }
}

void MpiController::debug(string msg) {
    cout << "[" << my_rank << "] ";
    cout << msg << endl;
}
