#include "BlockController.h"

BlockController::BlockController()  {}
BlockController::~BlockController() {
    pDataManager_->~DataManager();
    pFeatureTracker_->~FeatureTracker();
}

void BlockController::InitParameters(const Metadata &meta) {
    pDataManager_ = new DataManager();
    pDataManager_->InitTF(meta);
    pDataManager_->LoadDataSequence(meta, t_);

    pFeatureTracker_ = new FeatureTracker(pDataManager_->GetBlockDim());
    pFeatureTracker_->SetTFRes(pDataManager_->GetTFRes());
    pFeatureTracker_->SetTFMap(pDataManager_->GetTFMap(t_));
    pFeatureTracker_->SetDataPtr(pDataManager_->GetDataPtr(t_));
}

void BlockController::TrackForward(const Metadata &meta) {
    pDataManager_->LoadDataSequence(meta, t_);
    pFeatureTracker_->SetTFMap(pDataManager_->GetTFMap(t_));
    pFeatureTracker_->ExtractAllFeatures();
    pFeatureTracker_->TrackFeature(pDataManager_->GetDataPtr(t_), FT_FORWARD, FT_DIRECT);
    pFeatureTracker_->SaveExtractedFeatures(t_);
    pDataManager_->SaveMaskVolume(pFeatureTracker_->GetMaskPtr(), meta, t_);
}
