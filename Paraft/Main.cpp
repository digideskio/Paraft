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

    string filename = "/Users/Yang/Pictures/QQ20130131-1.png";
//    string window = "Blah";

//    IplImage *img = cvLoadImage(filename.c_str(), CV_LOAD_IMAGE_COLOR);

    SuperPixel *sp = new SuperPixel(filename.c_str());
    sp->SegmentNumber(500, 10);
    vector<Segment> segs = sp->GetSegments();

    int num_segs = sp->GetNumSegments();
    for (int i = 0; i < num_segs; ++i) {
        cout << "------------- " << i << " -----------------" << endl;
        cout << "center.x: " << segs[i].center.x << endl;
        cout << "center.y: " << segs[i].center.y << endl;
        cout << "num of pixels: " << segs[i].num_pixel << endl;
    }

    CvScalar color = CV_RGB(255, 255, 0);
    string save_path = "./result.jpg";
    sp->DrawContours(color, save_path);

    delete sp;
//    cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
//    cv::namedWindow(window.c_str(), CV_WINDOW_AUTOSIZE);
//    cv::imshow(window.c_str(), image);

//    cv::waitKey(0);
    return EXIT_SUCCESS;
}
