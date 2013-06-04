#include "BlockController.h"
#include "Metadata.h"

int main (int argc, char** argv) {
    Metadata *meta = new Metadata("");
//    Metadata meta; {
//        meta.start      = 0;
//        meta.end        = 23;
//        meta.prefix     = "";
//        meta.surfix     = "raw";
//        meta.path       = "/Users/Yang/Develop/ffv/sandbox/raw";
//        meta.tfPath     = "/Users/Yang/Develop/ffv/sandbox/raw/config.tfe";
//        meta.timeFormat = "%03d";
//        meta.volumeDim  = Vector3i(128, 64, 64);
//    }

//    Metadata meta; {
//        meta.start      = 60;
//        meta.end        = 65;
//        meta.prefix     = "vorts";
//        meta.surfix     = "data";
//        meta.path       = "/Users/Yang/Develop/Data/vorts";
//        meta.tfPath     = "/Users/Yang/Develop/Data/vorts/vorts.tfe";
//        meta.timeFormat = "%d";
//        meta.volumeDim  = Vector3i(128, 128, 128);
//    }

    int currentTimestep = meta->start();

    BlockController *pBlockController = new BlockController();
    pBlockController->SetCurrentTimestep(currentTimestep);
    pBlockController->InitParameters(*meta);

    while (currentTimestep++ < meta->end()) {
        pBlockController->SetCurrentTimestep(currentTimestep);
        pBlockController->TrackForward(*meta);
        cout << "-- " << currentTimestep << " done --" << endl;
    }

    delete pBlockController;
    return 0;
}
