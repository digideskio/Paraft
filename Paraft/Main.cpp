#include "Consts.h"
#include "BlockController.h"

int main (int argc, char** argv) {
    Metadata *meta = new Metadata(); {
        meta->start      = 50;
        meta->end        = 60;
        meta->prefix     = "vorts";
        meta->surfix     = "data";
        meta->path       = "/Users/Yang/Develop/Data/vorts8x";
        meta->tfPath     = "vorts8x.tfe";
        meta->volumeDim  = Vector3i(256, 256, 256);
    }

    int currentTimestep = meta->start;

    BlockController *pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentTimestep);
    pBlockController->InitParameters(meta);
    pBlockController->ExtractAllFeatures();

    while (currentTimestep++ < meta->end) {
        pBlockController->SetCurrentTimestep(currentTimestep);
        pBlockController->TrackForward(meta);
        cout << currentTimestep << " done." << endl;
    }

    delete meta;
    delete pBlockController;

    return 0;
}
