#include "DataBlockController.h"

DataBlockController::DataBlockController() {
    dataDim.x = dataDim.y = dataDim.z = 0;
    for (int surface = 0; surface < 6; surface++) {
        adjacentBlocks[surface] = SURFACE_NULL;
    }
}

DataBlockController::~DataBlockController() {
    highlightedFeatures.clear();
    localGraph.clear();
    pDataManager->~DataManager();
    pFeatureTracker->~FeatureTracker();
}

void DataBlockController::InitData(int globalID, Vector3i workerNumProcXYZ,
                                   Vector3i workerIDXYZ, DataSet ds) {

    Vector3i dimXYZ(DATA_DIM_X, DATA_DIM_Y, DATA_DIM_Z);

    if (globalID == HOST_NODE) {
        workerNumProcXYZ.x = 1;
        workerNumProcXYZ.y = 1;
        workerNumProcXYZ.z = 1;
    } else {
        initAdjacentBlocks(workerNumProcXYZ, workerIDXYZ);
    }

    pDataManager = new DataManager();
    pDataManager->ReadDataSequence(ds, dimXYZ, workerNumProcXYZ, workerIDXYZ);
    pDataManager->CreateNewMaskMatrix();

    dataDim = pDataManager->GetVolumeDimension();
    pFeatureTracker = new FeatureTracker(dataDim.x, dataDim.y, dataDim.z);
}

void DataBlockController::TrackForward() {
    pFeatureTracker->TrackFeature(pDataManager->GetVolumeDataPointer(currentTimestep),
                                  LOW_THRESHOLD, HIGH_THRESHOLD,
                                  TRACKING_FORWARD, TRACKING_MODE_DIRECT);
    ExtractAllFeatures();
}

void DataBlockController::ExtractAllFeatures() {
    float opacity;
    int index, tfIndex, tfRes = pFeatureTracker->GetTFResolution();

    for (int z = 0; z < dataDim.z; z++) {
        for (int y = 0; y < dataDim.y; y++ ) {
            for (int x = 0; x < dataDim.x; x++) {
                index = z * dataDim.y * dataDim.x + y * dataDim.x + x;
                if (pFeatureTracker->GetMaskMatrixPointer()[index] != 0) { continue; }
                tfIndex = (int)(pDataManager->GetVolumeDataPointer(currentTimestep)[index] * tfRes);
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
    memset(pDataManager->GetMaskMatrixPointer(), value,
           sizeof(float)*pDataManager->GetVolumeSize());
}

void DataBlockController::saveExtractedFeatures(vector<Feature>* f) {
    vector<Feature> temp;
    for (int i = 0; i < f->size(); i++) {
        temp.push_back(f->at(i));
    }
    pDataManager->SaveExtractedFeatures(temp);
}

void DataBlockController::initAdjacentBlocks(Vector3i blockPartition,
                                             Vector3i blockCoord) {

    int px = blockPartition.x, py = blockPartition.y, pz = blockPartition.z;
    int  x = blockCoord.x,      y = blockCoord.y,      z = blockCoord.z;

    adjacentBlocks[SURFACE_LEFT]   = x-1 >= 0  ? px*py*z + px*y + x - 1 : -1;
    adjacentBlocks[SURFACE_RIGHT]  = x+1 <  px ? px*py*z + px*y + x + 1 : -1;
    adjacentBlocks[SURFACE_BOTTOM] = y-1 >= 0  ? px*py*z + px*(y-1) + x : -1;
    adjacentBlocks[SURFACE_TOP]    = y+1 <  py ? px*py*z + px*(y+1) + x : -1;
    adjacentBlocks[SURFACE_FRONT]  = z-1 >= 0  ? px*py*(z-1) + px*y + x : -1;
    adjacentBlocks[SURFACE_BACK]   = z+1 <  pz ? px*py*(z+1) + px*y + x : -1;
}

void DataBlockController::UpdateLocalGraph(int blockID, Vector3i blockCoord) {
    localGraph.clear();

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
            edge.start      = blockID;
            edge.end        = adjacentBlock;
            edge.centroid   = centroid + dataDim * blockCoord;

            localGraph.push_back(edge);
        }
    }
}
