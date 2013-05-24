#include "BlockController.h"

int main (int argc, char** argv) {
//    Metadata meta; {
//        meta.start      = 0;
//        meta.end        = 5;
//        meta.prefix     = "ball_64_";
//        meta.surfix     = "raw";
//        meta.path       = "/Users/Yang/Develop/Data/ball5";
//        meta.tfPath     = "/Users/Yang/Develop/Data/ball5/ball.tfe";
//        meta.timeFormat = "%d";
//        meta.volumeDim  = Vector3i(64, 64, 64);
//    }

    Metadata meta; {
        meta.start      = 50;
        meta.end        = 60;
        meta.prefix     = "vorts";
        meta.surfix     = "data";
        meta.path       = "/Users/Yang/Develop/Data/vorts";
        meta.tfPath     = "/Users/Yang/Develop/Data/vorts/vorts.tfe";
        meta.timeFormat = "%d";
        meta.volumeDim  = Vector3i(128, 128, 128);
    }

    int currentTimestep = meta.start;

    BlockController *pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentTimestep);
    pBlockController->InitParameters(meta);

    while (currentTimestep++ < meta.end) {
        pBlockController->SetCurrentTimestep(currentTimestep);
        cout << "-- " << currentTimestep << " --" << endl;
        pBlockController->TrackForward(meta);
        cout << "-- done --" << endl;
    }

    delete pBlockController;
    return 0;
}
