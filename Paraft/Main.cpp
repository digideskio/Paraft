#include "BlockController.h"
#include "Metadata.h"
#include "SuperVoxel.h"

using namespace std;

int main () {
//    Metadata meta("/Users/Yang/Develop/Paraft/Paraft/vorts.config");
//    int currentTimestep = meta.start();

//    BlockController blockController;
//    blockController.SetCurrentTimestep(currentTimestep);
//    blockController.InitParameters(meta);

//    while (currentTimestep++ < meta.end()) {
//        blockController.SetCurrentTimestep(currentTimestep);
//        blockController.TrackForward(meta);
//        cout << "-- " << currentTimestep << " done --" << endl;
//    }

//    return EXIT_SUCCESS;


//    Metadata meta("/Users/Yang/Develop/Paraft/Paraft/supervoxel.config");
//    SuperVoxel sv(meta);
//    sv.SegmentByNumber(1000, 20);
//    vector<Cluster> clusters = sv.GetClusters();

//    int numClusters = sv.GetNumCluster();
//    for (int i = 0; i < numClusters; ++i) {
//        cout << "------------- " << i << " -----------------" << endl;
//        cout << "center.x: " << clusters[i].center.x << endl;
//        cout << "center.y: " << clusters[i].center.y << endl;
//        cout << "num of pixels: " << clusters[i].numVoxels << endl;
//    }

//    CvScalar color = CV_RGB(255, 255, 255);
//    string save_path = "/Users/Yang/Desktop/result.jpg";
//    sv.DrawContours(color, save_path);

//    return EXIT_SUCCESS;

    Metadata meta("/Users/Yang/Develop/Paraft/Paraft/vorts.config");
    int currentTimestep = meta.start();

    BlockController bc;
    bc.SetCurrentTimestep(currentTimestep);
    bc.Segment2SuperVoxel(meta);

    return EXIT_SUCCESS;
}
