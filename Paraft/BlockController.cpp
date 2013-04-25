#include "BlockController.h"

BlockController::BlockController()  {}
BlockController::~BlockController() {
    pDataManager->~DataManager();
    pFeatureTracker->~FeatureTracker();
}

void BlockController::InitParameters(Metadata *meta) {
    pDataManager = new DataManager();
    pDataManager->LoadDataSequence(meta, currentTimestep);
    pDataManager->CreateNewMaskVolume();
    pDataManager->InitTFSettings(meta->tfPath);

    blockDim = pDataManager->GetVolumeDimension();
    pFeatureTracker = new FeatureTracker(blockDim);
    pFeatureTracker->SetThresholds(LOW_THRESHOLD, HIGH_THRESHOLD);
    pFeatureTracker->SetTFResolution(pDataManager->GetTFResolution());
    pFeatureTracker->SetTFOpacityMap(pDataManager->GetTFOpacityMap());
    pFeatureTracker->SetDataPointer(pDataManager->GetDataPointer(currentTimestep));
}

void BlockController::TrackForward(Metadata *meta) {
    pDataManager->LoadDataSequence(meta, currentTimestep);
    pFeatureTracker->TrackFeature(pDataManager->GetDataPointer(currentTimestep),
                                  TRACKING_FORWARD, TRACKING_MODE_DIRECT);
    ExtractAllFeatures();
}

void BlockController::ExtractAllFeatures() {
    int tfRes = pFeatureTracker->GetTFResolution();
    float *pMask = pFeatureTracker->GetMaskVolumePointer();
    float *pData = pDataManager->GetDataPointer(currentTimestep);
    float *pTfMap = pFeatureTracker->GetTFOpacityMap();

    for (int z = 0; z < blockDim.z; z++) {
        for (int y = 0; y < blockDim.y; y++) {
            for (int x = 0; x < blockDim.x; x++) {
                int index = z * blockDim.y * blockDim.x + y * blockDim.x + x;
                if (pMask[index] != 0)  // point already within a feature
                    continue;           // most points stop here
                int tfIndex = (int)(pData[index] * (float)(tfRes-1));
                if (pTfMap[tfIndex] >= LOW_THRESHOLD && pTfMap[tfIndex] <= HIGH_THRESHOLD) {
                    pFeatureTracker->FindNewFeature(DataPoint(x,y,z));
                }
            }
        }
    }
    pFeatureTracker->SaveExtractedFeatures(currentTimestep);
}
