#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "Utils.h"
#include "DataManager.h"
#include "FeatureTracker.h"
#include "SuperVoxel.h"

class BlockController {

public:
    BlockController();
   ~BlockController();

    void InitParameters(const Metadata& meta);
    void TrackForward(const Metadata& meta);
    void ExtractAllFeatures();
    void SetCurrentTimestep(int t) { currentTimestep_ = t; }

    void Segment2SuperVoxel(const Metadata& meta);

private:
    DataManager    *pDataManager_;
    FeatureTracker *pFeatureTracker_;
    SuperVoxel     *pSuperVoxel_;
    int             currentTimestep_;
};

#endif // DATABLOCKCONTROLLER_H
