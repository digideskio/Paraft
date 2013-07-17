#include "BlockController.h"

BlockController::BlockController() {}
BlockController::~BlockController() {
    pDataManager_->~DataManager();
    pFeatureTracker_->~FeatureTracker();
}

void BlockController::InitParameters(const Metadata &meta) {
    pDataManager_ = new DataManager();
    pDataManager_->InitTF(meta);
    pDataManager_->LoadDataSequence(meta, currentTimestep_);

    pFeatureTracker_ = new FeatureTracker(pDataManager_->GetBlockDim());
    pFeatureTracker_->SetTFRes(pDataManager_->GetTFRes());
    pFeatureTracker_->SetTFMap(pDataManager_->GetTFMap(currentTimestep_));
    pFeatureTracker_->SetDataPtr(pDataManager_->GetDataPtr(currentTimestep_));
}

void BlockController::TrackForward(const Metadata &meta) {
    pDataManager_->LoadDataSequence(meta, currentTimestep_);
    pFeatureTracker_->SetTFMap(pDataManager_->GetTFMap(currentTimestep_));
    pFeatureTracker_->ExtractAllFeatures();
    pFeatureTracker_->TrackFeature(pDataManager_->GetDataPtr(currentTimestep_), FT_FORWARD, FT_DIRECT);
    pFeatureTracker_->SaveExtractedFeatures(currentTimestep_);
    pDataManager_->SaveMaskVolume(pFeatureTracker_->GetMaskPtr(), meta, currentTimestep_);
}

void BlockController::Segment2SuperVoxel(const Metadata &meta) {
    pDataManager_ = new DataManager();
    pDataManager_->LoadDataSequence(meta, currentTimestep_);

    pSuperVoxel_ = new SuperVoxel(meta.volumeDim());
    pSuperVoxel_->SetDataPtr(pDataManager_->GetDataPtr(currentTimestep_));
    pSuperVoxel_->ClusterByNumber(8, 20);
}
