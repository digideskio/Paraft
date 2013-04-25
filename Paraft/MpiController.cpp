#include "MpiController.h"

MpiController::MpiController() {}

MpiController::~MpiController() {
    MPI_Finalize();
    pBlockController->~BlockController();
}

void MpiController::InitWith(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    // init MPI_TYPE_EDGE
    MPI_Type_contiguous(sizeof(Edge), MPI_BYTE, &MPI_TYPE_EDGE);
    MPI_Type_commit(&MPI_TYPE_EDGE);

    gridDim = Vector3i(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));

    blockIdx.z = myRank / (gridDim.x * gridDim.y);
    blockIdx.y = (myRank - blockIdx.z * gridDim.x * gridDim.y) / gridDim.x;
    blockIdx.x = myRank % gridDim.x;

    meta->start      = 100;
    meta->end        = 102;
    meta->prefix     = "i2vgt_";
    meta->surfix     = "raw";
    meta->path       = "/Users/Yang/Develop/ffv/sandbox/raw/i2vgt";
    meta->tfPath     = "ffvc.tfe";
    meta->volumeDim  = Vector3i(400, 200, 200);

    csv.gridDim = gridDim;
    csv.numProc = numProc;
    csv.numFeature = 0;
    csv.t1 = 0.0;
    csv.t2 = 0.0;
    csv.t3 = 0.0;
    csv.t4 = 0.0;
}

void MpiController::Start() {
    initBlockController();

    while (currentTimestep < meta->end) {
        TrackForward();
        if (myRank == 0) cout << currentTimestep << " done." << endl;
        currentTimestep++;
    }

    if (myRank == 0) cout << myRank << " over." << endl;
}

void MpiController::initBlockController() {
    currentTimestep = meta->start;
    pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentTimestep);
    pBlockController->InitParameters(gridDim, blockIdx, meta);
    pBlockController->ExtractAllFeatures();
}

void MpiController::TrackForward() {
    pBlockController->SetCurrentTimestep(currentTimestep);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    pBlockController->TrackForward(gridDim, blockIdx, meta);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    pBlockController->UpdateLocalGraph(myRank, blockIdx);
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    featureTable.clear();

    adjacentBlocks = pBlockController->GetAdjacentBlocks();
    need_to_send = need_to_recv = true;
    any_send = any_recv = true;

    while (any_send || any_recv) {
        syncFeatureGraph();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    gatherGlobalGraph();

    MPI_Barrier(MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    featureTableVector[currentTimestep] = featureTable;

    double delta = t1 - t0;
    MPI_Allreduce(&delta, &csv.t1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delta = t2 - t1;
    MPI_Allreduce(&delta, &csv.t2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delta = t3 - t2;
    MPI_Allreduce(&delta, &csv.t3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delta = t4 - t3;
    MPI_Allreduce(&delta, &csv.t4, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    csv.t1 /= (double)numProc;
    csv.t2 /= (double)numProc;
    csv.t3 /= (double)numProc;
    csv.t4 /= (double)numProc;

    csv.numFeature = featureTable.size();

    char np[21];
    sprintf(np, "%d", numProc);
    string fpath = "";
    string prefix = "result_";
    string surfix = ".csv";
    string result = fpath.append(prefix).append(np).append(surfix);
    ofstream outf(result.c_str(), ios::out | ios::app);

    if (myRank == 0) {
        outf << csv.numProc << "," << csv.numFeature << "," << currentTimestep << ","
             << csv.t1 << "," << csv.t2 << "," << csv.t3 << "," << csv.t4 << endl;
    }

    outf.close();
}

void MpiController::syncFeatureGraph() {
    vector<Edge> localEdges = pBlockController->GetLocalGraph();
    int localEdgeCount = localEdges.size();

    vector<int> blocksNeedRecv(numProc);
    fill(blocksNeedRecv.begin(), blocksNeedRecv.end(), -1);

    int recv_id = need_to_recv ? myRank : -1;
    MPI_Allgather(&recv_id, 1, MPI_INT, &blocksNeedRecv.front(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> adjacentBlocksNeedRecv;

    sort(adjacentBlocks.begin(), adjacentBlocks.end());
    sort(blocksNeedRecv.begin(), blocksNeedRecv.end());

    set_intersection(adjacentBlocks.begin(), adjacentBlocks.end(), blocksNeedRecv.begin(), blocksNeedRecv.end(),
                     back_inserter(adjacentBlocksNeedRecv));

    need_to_send = adjacentBlocksNeedRecv.size() > 0 ? true : false;

    vector<Edge> adjacentGraph;

    if (need_to_recv) {
        for (size_t i = 0; i < adjacentBlocks.size(); i++) {
            int src = adjacentBlocks.at(i);
            int srcEdgeCount = 0;

            // 1. recv count
            MPI_Irecv(&srcEdgeCount, 1, MPI_INT, src, 100, MPI_COMM_WORLD, &request);
            // 2. recv content
            if (srcEdgeCount == 0) continue;

            vector<Edge> srcEdges(srcEdgeCount);
            MPI_Irecv(&srcEdges.front(), srcEdgeCount, MPI_TYPE_EDGE, src, 101, MPI_COMM_WORLD, &request);

            for (int i = 0; i < srcEdgeCount; i++) {
                bool isNew = true;
                for (size_t j = 0; j < adjacentGraph.size(); j++) {
                    if (srcEdges[i] == adjacentGraph[j]) {
                        isNew = false; break;
                    }
                }
                if (isNew) {
                    adjacentGraph.push_back(srcEdges[i]);
                }
            }
        }
        need_to_recv = false;
    }

    if (need_to_send) {
        for (size_t i = 0; i < adjacentBlocksNeedRecv.size(); i++) {
            int dest = adjacentBlocksNeedRecv.at(i);

            // 1. send count
            MPI_Send(&localEdgeCount, 1, MPI_INT, dest, 100, MPI_COMM_WORLD);
            // 2. send content
            if (localEdgeCount > 0) {
                MPI_Send(&localEdges.front(), localEdgeCount, MPI_TYPE_EDGE, dest, 101, MPI_COMM_WORLD);
            }
        }
    }

    // add local edges
    for (size_t i = 0; i < localEdges.size(); i++) {
        bool isNew = true;
        for (size_t j = 0; j < adjacentGraph.size(); j++) {
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

    MPI_Allreduce(&need_to_send, &any_send, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    MPI_Allreduce(&need_to_recv, &any_recv, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
}

void MpiController::gatherGlobalGraph() {
    vector<Edge> localEdges = pBlockController->GetLocalGraph();
    int localEdgeCount = localEdges.size();

    vector<int> globalEdgeCountVector(numProc);

    MPI_Allgather(&localEdgeCount, 1, MPI_INT, &globalEdgeCountVector.front(), 1, MPI_INT, MPI_COMM_WORLD);

    globalEdgeCount = 0;
    for (size_t i = 0; i < globalEdgeCountVector.size(); i++) {
        globalEdgeCount += globalEdgeCountVector[i];
    }

    vector<Edge> globalEdges(globalEdgeCount);

    int displs[numProc];
    displs[0] = 0;
    for (int i = 1; i < numProc; i++) {
        displs[i] = globalEdgeCountVector[i-1] + displs[i-1];
    }

    MPI_Allgatherv(&localEdges.front(), localEdgeCount, MPI_TYPE_EDGE, &globalEdges.front(),
                   &globalEdgeCountVector.front(), displs, MPI_TYPE_EDGE, MPI_COMM_WORLD);

    mergeCorrespondentEdges(globalEdges);
}

void MpiController::mergeCorrespondentEdges(vector<Edge> edges) {
    for (size_t i = 0; i < edges.size(); i++) {
        Edge ei = edges[i];
        for (size_t j = i+1; j < edges.size(); j++) {
            Edge ej = edges[j];

            // sync the id of feature if two matches
            if (ei.start == ej.end && ei.end == ej.start &&  // 0->1 | 1->0
                (ei.start == myRank || ei.end == myRank) &&
                ei.centroid.DistanceFrom(ej.centroid) <= DIST_THRESHOLD) {
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
    for (size_t i = 0; i < edges.size(); i++) {
        Edge edge = edges[i];
        if (edge.start == myRank || edge.end == myRank) {
            updateFeatureTable(edge);
        }
    }

    // if both start and end are not equal to my_rank,
    // but the id is already in the feature table, update featureTable
    for (size_t i = 0; i < edges.size(); i++) {
        Edge edge = edges[i];
        if (edge.start != myRank || edge.end != myRank) {
            if (featureTable.find(edge.id) != featureTable.end()) {
                updateFeatureTable(edge);
            }
        }
    }

    if (myRank == 0) {     // debug log
        FeatureTable::iterator it;
        for (it = featureTable.begin(); it != featureTable.end(); it++) {
            int id = it->first;
            cout << "[" << myRank << "]" << id << ": ( ";
            vector<int> value = it->second;
            for (size_t i = 0; i < value.size(); i++) {
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
