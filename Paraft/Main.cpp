#include "BlockController.h"
#include "Metadata.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SuperPixel.h"

using namespace std;

int main (int argc, char** argv) {
//    Metadata *meta = new Metadata("jet.config");
//    int currentTimestep = meta->start();

//    BlockController *pBlockController = new BlockController();
//    pBlockController->SetCurrentTimestep(currentTimestep);
//    pBlockController->InitParameters(*meta);

//    while (currentTimestep++ < meta->end()) {
//        pBlockController->SetCurrentTimestep(currentTimestep);
//        pBlockController->TrackForward(*meta);
//        cout << "-- " << currentTimestep << " done --" << endl;
//    }

//    delete pBlockController;
//    return 0;

    string filename = "/Users/Yang/Desktop/1.png";


    SuperPixel *sp = new SuperPixel();
    sp->InitWith(filename.c_str());
    sp->SegmentNumber(100, 5);
    vector<Segment> segs = sp->GetSegments();

    int num_segs = sp->GetNumSegments();
    for (int i = 0; i < num_segs; ++i) {
        cout << "------------- " << i << " -----------------" << endl;
        cout << "center.x: " << segs[i].center.x << endl;
        cout << "center.y: " << segs[i].center.y << endl;
        cout << "num of pixels: " << segs[i].num_pixel << endl;
    }

    CvScalar color = CV_RGB(255, 255, 255);
    string save_path = "/Users/Yang/Desktop/result.jpg";
    sp->DrawContours(color, save_path);

    delete sp;

    return EXIT_SUCCESS;
}
