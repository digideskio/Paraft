#include "BlockController.h"

int main (int argc, char** argv) {
    Metadata meta; {
        meta.start      = 0;
        meta.end        = 5;
        meta.prefix     = "ball_64_";
        meta.surfix     = "raw";
        meta.path       = "/Users/Yang/Develop/Data/ball5";
        meta.tfPath     = "/Users/Yang/Develop/Data/ball5/ball.tfe";
        meta.timeFormat = "%d";
        meta.volumeDim  = Vector3i(64, 64, 64);
    }

    int currentTimestep = meta.start;

    BlockController *pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentTimestep);
    pBlockController->InitParameters(meta);

    while (currentTimestep++ < meta.end) {
        pBlockController->SetCurrentTimestep(currentTimestep);
        pBlockController->TrackForward(meta);
        cout << currentTimestep << " done." << endl;
    }

    delete pBlockController;
    return 0;
}
