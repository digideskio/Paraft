#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include "Utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Segment {
    Vector3i center;
    int num_pixel;
};

class SuperPixel {
public:
    explicit SuperPixel();
    virtual ~SuperPixel();

    void InitWith(const string& image_path);

    // segment by specific 1. number of segments or 2. segment size
    bool SegmentNumber(const int& expected_seg_num, const float& weight_m);
    bool SegmentSize(const int& expected_seg_size, const float& weight_m);

    void DrawContours(const CvScalar& drawing_color, const string& save_path);

    // accessors
    int GetNumSegments()                const { return num_segments_; }
    const int* GetSegmentMap()          const { return segmentation_map_; }
    const vector<Segment> GetSegments() const { return segments_; }

private:
    IplImage *pImage_;   // input image
    int num_segments_;   // number of segments after segmentation
    int width_;         // input image width
    int height_;        // input image height
    int *segmentation_map_;    // ?
    vector<Segment> segments_;  // segment result

    Vector3i dim_;     // data dimension

    float *l_values_;
    float *a_values_;
    float *b_values_;
    float *gradients_;

    vector<float> k_centers_l_;
    vector<float> k_centers_a_;
    vector<float> k_centers_b_;
    vector<int> k_centers_x_;
    vector<int> k_centers_y_;

    void bgr2lab();  // opencv uses BGR instead of RGB
    void detectGradients(); // detects gradient map to perturb seeds
    void getInitialCenters(const int& expected_seg_size);

    // Super pixel clustering. Need post-processing for enforcing connectivity.
    void clusteringIteration(const int& expected_seg_size, const float& weight_m, int* temp_segmentation_map);

    // Find next connected components(pixel) which belongs to the same cluster.
    // Function is called recursively to get the size of connected area cluster.
    void findNext(const int* temp_segmentation_map, const int& row_index, const int& col_index,
                  const int& segment_index, int* x_pos, int* y_pos, int* num_count);

    void enforceConnectivity(const int* temp_segmentation_map, const int& expected_seg_size);
};

#endif // SUPERPIXEL_H
