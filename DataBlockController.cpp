#include "DataBlockController.h"

DataBlockController::DataBlockController() {
    xs = 0; ys = 0; zs = 0;
    for (int surface = 0; surface < 6; surface++) {
        adjacentBlocks[surface] = SURFACE_NULL;
    }
}

DataBlockController::~DataBlockController() {
    highlightedFeatures.clear();
    localConnectivityGraph.clear();
    pDataManager->~DataManager();
    pFeatureTracker->~FeatureTracker();
}

void DataBlockController::InitData(int globalID, Vector3i workerNumProcessesXYZ,
                                   Vector3i workerIDXYZ, DataSet ds) {
    Vector3i dimXYZ; {
        dimXYZ.x = DATA_DIM_X;
        dimXYZ.y = DATA_DIM_Y;
        dimXYZ.z = DATA_DIM_Z;
    }

    if (globalID == HOST_NODE) {
        workerNumProcessesXYZ.x = 1;
        workerNumProcessesXYZ.y = 1;
        workerNumProcessesXYZ.z = 1;
    } else {
        initAdjacentBlocks(workerNumProcessesXYZ, workerIDXYZ);
    }

    dataset = ds;

    pDataManager = new DataManager();
    pDataManager->ReadDataSequence(dataset.data_path, dataset.prefix, dataset.surfix,
                                   dataset.index_start, dataset.index_end,
                                   dimXYZ, workerNumProcessesXYZ, workerIDXYZ);
    pDataManager->CreateNewMaskMatrix();

    xs = pDataManager->GetVolumeDimX();
    ys = pDataManager->GetVolumeDimY();
    zs = pDataManager->GetVolumeDimZ();
    pFeatureTracker = new FeatureTracker(xs, ys, zs);
}

void DataBlockController::TrackForward() {
    pFeatureTracker->TrackFeature(pDataManager->GetVolumeDataPointer(currentTimestep),
                                  LOW_THRESHOLD, HIGH_THRESHOLD,
                                  TRACKING_DIRECTION_FORWARD, TRACKING_MODE_DIRECT);
    ExtractAllFeatures();
}

void DataBlockController::ExtractAllFeatures() {
    float opacity;
    int index, tfIndex;

    for (int z = 0; z < zs; z++) {
        for (int y = 0; y < ys; y++ ) {
            for (int x = 0; x < xs; x++) {
                index = z * ys * xs + y * xs + x;
                if (pFeatureTracker->GetMaskMatrixPointer()[index] != 0) { continue; }
                tfIndex = (int)(pDataManager->GetVolumeDataPointer(currentTimestep)[index] * pFeatureTracker->GetTFResolution());
                opacity = pFeatureTracker->GetTFColorMap()[tfIndex*4+3];
                if (opacity >= LOW_THRESHOLD && opacity <= HIGH_THRESHOLD) {
                    pFeatureTracker->FindNewFeature(x, y, z, LOW_THRESHOLD, HIGH_THRESHOLD);
                }
            }
        }
    }
    saveExtractedFeatures(pFeatureTracker->GetCurrentFeatureInfo());
}

void DataBlockController::AddHighlightedFeature(int index) {
    vector<int>::iterator it = find(highlightedFeatures.begin(), highlightedFeatures.end(), index);
    if (it == highlightedFeatures.end()) {
        highlightedFeatures.push_back(index);
    }
}

void DataBlockController::ResetMaskMatrixValue(float value) {
    memset(pDataManager->GetMaskMatrixPointer(), value, sizeof(float)*pDataManager->GetVolumeSize());
}

void DataBlockController::saveExtractedFeatures(vector<Feature>* f) {
    vector<Feature> temp;
    for (int i = 0; i < f->size(); i++) {
        temp.push_back(f->at(i));
    }
    pDataManager->SaveExtractedFeatures(temp);
}

void DataBlockController::initAdjacentBlocks(Vector3i workerNumProcessesXYZ,
                                             Vector3i workerIDXYZ) {
    int npx = workerNumProcessesXYZ.x;
    int npy = workerNumProcessesXYZ.y;
    int npz = workerNumProcessesXYZ.z;
    int z = workerIDXYZ.z;
    int y = workerIDXYZ.y;
    int x = workerIDXYZ.x;
    adjacentBlocks[SURFACE_LEFT]   = x-1 >= 0  ? npx*npy*z + npx*y + x - 1 : -1;
    adjacentBlocks[SURFACE_RIGHT]  = x+1 <  npx? npx*npy*z + npx*y + x + 1 : -1;
    adjacentBlocks[SURFACE_BOTTOM] = y-1 >= 0  ? npx*npy*z + npx*(y-1) + x : -1;
    adjacentBlocks[SURFACE_TOP]    = y+1 <  npy? npx*npy*z + npx*(y+1) + x : -1;
    adjacentBlocks[SURFACE_FRONT]  = z-1 >= 0  ? npx*npy*(z-1) + npx*y + x : -1;
    adjacentBlocks[SURFACE_BACK]   = z+1 <  npz? npx*npy*(z+1) + npx*y + x : -1;
}

void DataBlockController::UpdateLocalGraph(int workerID, Vector3i workerIDXYZ) {
    localConnectivityGraph.clear();

    vector<Feature> *pCurrentFeatures;
    pCurrentFeatures = pDataManager->GetFeatureVector(currentTimestep);
    if (pCurrentFeatures->size() == 0) {
        return;
    }

    vector<int> touchedSurfaces;
    int surface = SURFACE_NULL;
    int adjacentBlock = -1;
    Feature feature;
    Vector3i centroid;
    Edge edge;
    for (int i = 0; i < pCurrentFeatures->size(); i++) {
        feature = pCurrentFeatures->at(i);
        touchedSurfaces = feature.TouchedSurfaces;

        for (int j = 0; j < touchedSurfaces.size(); j++) {
            surface = touchedSurfaces[j];
            adjacentBlock = adjacentBlocks[surface];
            if (adjacentBlock == -1) {
                continue;
            }

            centroid = feature.BoundaryCentroid[surface];

            edge.id         = feature.ID;
            edge.start      = workerID;
            edge.end        = adjacentBlock;
            edge.centroid = ConvertLocalCoord2GlobalCoord(centroid, workerIDXYZ);

            localConnectivityGraph.push_back(edge);
        }
    }
}

Vector3i DataBlockController::ConvertLocalCoord2GlobalCoord(Vector3i point, Vector3i workerIDXYZ) {
    Vector3i globalCoord;
    globalCoord.x = point.x + xs * workerIDXYZ.x;
    globalCoord.y = point.y + ys * workerIDXYZ.y;
    globalCoord.z = point.z + zs * workerIDXYZ.z;
    return globalCoord;
}
