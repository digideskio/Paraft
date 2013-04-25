#ifndef DATABLOCKCONTROLLER_H
#define DATABLOCKCONTROLLER_H

#include "Consts.h"
#include "DataManager.h"
#include "FeatureTracker.h"

class BlockController {

public:
    BlockController();
    ~BlockController();

    void InitParameters(Metadata *meta);
    void TrackForward(Metadata *meta);
    void ExtractAllFeatures();
    void SetCurrentTimestep(int t) { currentTimestep = t; }

private:
    DataManager    *pDataManager;
    FeatureTracker *pFeatureTracker;
    Vector3i        blockDim;
    int             currentTimestep;
};

#endif // DATABLOCKCONTROLLER_H
