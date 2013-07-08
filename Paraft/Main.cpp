#include "BlockController.h"
#include "Metadata.h"
#include "SuperVoxel.h"

using namespace std;

int main (int argc, char** argv) {
//    Metadata meta("/Users/Yang/Develop/Paraft/Paraft/vorts.config");
//    int currentTimestep = meta.start();

//    BlockController *pBlockController = new BlockController();
//    pBlockController->SetCurrentTimestep(currentTimestep);
//    pBlockController->InitParameters(meta);

//    while (currentTimestep++ < meta.end()) {
//        pBlockController->SetCurrentTimestep(currentTimestep);
//        pBlockController->TrackForward(meta);
//        cout << "-- " << currentTimestep << " done --" << endl;
//    }

//    delete pBlockController;
//    return EXIT_SUCCESS;


    Metadata meta("/Users/Yang/Develop/Paraft/Paraft/supervoxel.config");
    SuperVoxel sv(meta);
    sv.SegmentByNumber(100, 5);
    vector<Cluster> clusters = sv.GetClusters();

    int numClusters = sv.GetNumCluster();
    for (int i = 0; i < numClusters; ++i) {
        cout << "------------- " << i << " -----------------" << endl;
        cout << "center.x: " << clusters[i].center.x << endl;
        cout << "center.y: " << clusters[i].center.y << endl;
        cout << "num of pixels: " << clusters[i].numVoxels << endl;
    }

    CvScalar color = CV_RGB(255, 255, 255);
    string save_path = "/Users/Yang/Desktop/result.jpg";
    sv.DrawContours(color, save_path);

    return EXIT_SUCCESS;
}
