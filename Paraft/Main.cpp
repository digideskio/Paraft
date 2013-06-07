#include "BlockController.h"
#include "Metadata.h"

int main (int argc, char** argv) {
    Metadata *meta = new Metadata("jet.config");
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
