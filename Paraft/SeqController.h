#ifndef NONMPICONTROLLER_H
#define NONMPICONTROLLER_H

#include "BlockController.h";

class SeqController {
public:
    SeqController();
    ~SeqController();

    void Init();
    void Start();
//    void TrackForward();

private:
    BlockController *pBlockController;
    FeatureTable *featureTable;
    Metadata *meta;
    int currentTimestep;  // current timestep
};

#endif // NONMPICONTROLLER_H
