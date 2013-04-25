#include "BlockController.h"

int main (int argc, char** argv) {

    Metadata meta; {
        meta.start      = 100;
        meta.end        = 110;
        meta.prefix     = "vort_";
        meta.surfix     = "raw";
        meta.path       = "/Users/Yang/Develop/Data/yubo_new/vorts";
        meta.tfPath     = "vorts8x.tfe";
        meta.volumeDim  = Vector3i(256, 256, 256);
    }

    int currentTimestep = meta.start;

    BlockController *pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentTimestep);
    pBlockController->InitParameters(meta);
//    pBlockController->ExtractAllFeatures();

    while (currentTimestep++ < meta.end) {
        pBlockController->SetCurrentTimestep(currentTimestep);
        pBlockController->TrackForward(meta);
        cout << currentTimestep << " done." << endl;
    }

    delete pBlockController;
    return 0;
}
