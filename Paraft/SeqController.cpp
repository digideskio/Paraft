#include "SeqController.h"

SeqController::SeqController() { }

SeqController::~SeqController() {
    pBlockController->~BlockController();
}

void SeqController::Init() {
    meta = new Metadata(); {
        meta->start      = 100;
        meta->end        = 110;
        meta->prefix     = "i2vgt_";
        meta->surfix     = "raw";
        meta->path       = "/Users/Yang/Develop/ffv/sandbox/raw/i2vgt";
        meta->tfPath     = "ffvc.tfe";
        meta->volumeDim  = Vector3i(400, 200, 200);
    }
}

void SeqController::Start() {
    currentTimestep = meta->start;

    pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentTimestep);

    pBlockController->InitParameters(meta);
    cout << "1" << endl;
    pBlockController->ExtractAllFeatures();
    cout << "2" << endl;

    while (currentTimestep++ < meta->end) {
        pBlockController->SetCurrentTimestep(currentTimestep);
        pBlockController->TrackForward(meta);
        cout << currentTimestep << " done." << endl;
    }
}
