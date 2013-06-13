#include "SuperPixel.h"
//#include <float.h>
//#include <algorithm>

SuperPixel::SuperPixel(const string& image_path) {
    image_ = cvLoadImage(image_path.c_str(), CV_LOAD_IMAGE_COLOR);
    num_segments_ = -1;

    width_ = image_->width;     // will change to vector3i later
    height_ = image_->height;   // will change to vector3i later

    segmentation_map_ = nullptr;
    l_values_ = nullptr;
    a_values_ = nullptr;
    b_values_ = nullptr;
    gradients_ = nullptr;
}

SuperPixel::~SuperPixel() {
    if (image_)             { cvReleaseImage(&image_); image_ = nullptr; }
    if (segmentation_map_)  { delete [] segmentation_map_; segmentation_map_ = nullptr; }
    if (l_values_)          { delete [] l_values_; l_values_ = nullptr; }
    if (a_values_)          { delete [] a_values_; a_values_ = nullptr; }
    if (b_values_)          { delete [] b_values_; b_values_ = nullptr; }
    if (gradients_)         { delete [] gradients_; gradients_ = nullptr; }
}

bool SuperPixel::SegmentNumber(const int &expected_seg_num, const float &weight_m) {
    int num_pixels = width_ * height_;
    int expected_seg_size = cvRound(sqrt(static_cast<double>(num_pixels) / expected_seg_num));
    return SegmentSize(expected_seg_size, weight_m);
}

bool SuperPixel::SegmentSize(const int &expected_seg_size, const float &weight_m) {
    if (!image_ || image_->nChannels != 3 || image_->depth != IPL_DEPTH_8U) { return false; }
    if (expected_seg_size > 100) { return false; }

    // 0. init
    int num_pixels = width_ * height_;

    int *temp_segmentation_map = new int[num_pixels];
    std::fill_n(temp_segmentation_map, num_pixels, -1);

    segmentation_map_ = new int[num_pixels];
    std::fill_n(segmentation_map_, num_pixels, -1);

    l_values_ = new float[num_pixels];
    a_values_ = new float[num_pixels];
    b_values_ = new float[num_pixels];

    gradients_ = new float[num_pixels];
    std::fill_n(gradients_, num_pixels, 0.0);

    // 1. RGB to LAB
    bgr2lab();

    // 2. Spread seeds
    detectGradients();
    getInitialCenters(expected_seg_size);

    // 3. clustering
    clusteringIteration(expected_seg_size, weight_m, temp_segmentation_map);

    // 4. enforce connectivity
    enforceConnectivity(temp_segmentation_map, expected_seg_size);

    // 5. generate segmentation results
    for (int i = 0; i < num_segments_; i++) {
        Segment seg; {
            seg.center.x = 0;
            seg.center.y = 0;
            seg.num_pixel = 0;
        }
        segments_.push_back(seg);
    }
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            int pos = y * width_ + x;
            int index = segmentation_map_[pos];
            segments_[index].center.x += x;
            segments_[index].center.y += y;
            segments_[index].num_pixel++;
        }
    }
    for (int i = 0; i < num_segments_; i++) {
        segments_[i].center.x = cvRound(static_cast<float>(segments_[i].center.x) / segments_[i].num_pixel);
        segments_[i].center.y = cvRound(static_cast<float>(segments_[i].center.y) / segments_[i].num_pixel);
        segments_[i].center.x = segments_[i].center.x >= width_  ? width_  - 1 : segments_[i].center.x;
        segments_[i].center.y = segments_[i].center.y >= height_ ? height_ - 1 : segments_[i].center.y;
    }

    // 6. cleaning
    k_centers_l_.clear();
    k_centers_a_.clear();
    k_centers_b_.clear();
    k_centers_x_.clear();
    k_centers_y_.clear();

    delete [] temp_segmentation_map;
    temp_segmentation_map = nullptr;

    return true;
}

void SuperPixel::DrawContours(const CvScalar& drawing_color, const std::string& save_path) {
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
    IplImage* contour = cvCreateImage(cvSize(width_, height_), image_->depth, image_->nChannels);
    cvCopy(image_, contour);
    int step = contour->widthStep;
    uchar* data = reinterpret_cast<uchar*>(contour->imageData);
    int const kTotalPixelNum = width_ * height_;
    std::vector<bool> istaken(kTotalPixelNum, false);
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            int diff = 0;
            int pos_a = y * width_ + x;
            for (int i = 0; i < 8; ++i) {
                int x = x + dx8[i];
                int y = y + dy8[i];
                if ((x >= 0 && x < width_) && (y >= 0 && y < height_)) {
                    int pos_b = y * width_ + x;
                    if ((false == istaken[pos_a]) &&
                        (segmentation_map_[pos_a] != segmentation_map_[pos_b])) {
                        diff++;
                    }
                }
            }
            if (diff >= 2) {
                istaken[pos_a] = true;
                data[y * step + x * 3 + 0] = drawing_color.val[0];
                data[y * step + x * 3 + 1] = drawing_color.val[1];
                data[y * step + x * 3 + 2] = drawing_color.val[2];
            }
        }
    }
    cvSaveImage(save_path.c_str(), contour);
    cvReleaseImage(&contour);
}

void SuperPixel::bgr2lab() {
    int step = image_->widthStep;
    uchar* data = reinterpret_cast<uchar*>(image_->imageData);
    double epsilon = 0.008856;
    double kappa   = 903.3;
    double Xr = 0.950456;
    double Yr = 1.0;
    double Zr = 1.088754;
    for (int row = 0; row < height_; ++row) {
        for (int col = 0; col < width_; ++col) {
            // Access pixel values.
            double B = static_cast<double>(data[row * step + col * 3 + 0]) / 255.0;
            double G = static_cast<double>(data[row * step + col * 3 + 1]) / 255.0;
            double R = static_cast<double>(data[row * step + col * 3 + 2]) / 255.0;

            // Step 1: RGB to XYZ conversion.
            double r = R <= 0.04045 ? R/12.92 : pow((R+0.055)/1.055, 2.4);
            double g = G <= 0.04045 ? G/12.92 : pow((G+0.055)/1.055, 2.4);
            double b = B <= 0.04045 ? B/12.92 : pow((B+0.055)/1.055, 2.4);

            double X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
            double Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
            double Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

            // Step 2: XYZ to LAB conversion.
            double xr = X/Xr;
            double yr = Y/Yr;
            double zr = Z/Zr;

            double fx = xr > epsilon ? pow(xr, 1.0/3.0) : (kappa*xr + 16.0)/116.0;
            double fy = yr > epsilon ? pow(yr, 1.0/3.0) : (kappa*yr + 16.0)/116.0;
            double fz = zr > epsilon ? pow(zr, 1.0/3.0) : (kappa*zr + 16.0)/116.0;

            // Add converted color to 1-D vectors.
            int pos = row * width_ + col;
            l_values_[pos] = 116.0 * fy - 16.0;
            a_values_[pos] = 500.0 * (fx - fy);
            b_values_[pos] = 200.0 * (fy - fz);
        }
    }
}

void SuperPixel::detectGradients() {
    for (int row = 1; row < height_ - 1; ++row) {
        for (int col = 1; col < width_ - 1; ++col) {
            int i = row * width_ + col;
            double dx = (l_values_[i-1]-l_values_[i+1]) * (l_values_[i-1]-l_values_[i+1]) +
                        (a_values_[i-1]-a_values_[i+1]) * (a_values_[i-1]-a_values_[i+1]) +
                        (b_values_[i-1]-b_values_[i+1]) * (b_values_[i-1]-b_values_[i+1]);
            double dy = (l_values_[i-width_]-l_values_[i+width_]) * (l_values_[i-width_]-l_values_[i+width_]) +
                        (a_values_[i-width_]-a_values_[i+width_]) * (a_values_[i-width_]-a_values_[i+width_]) +
                        (b_values_[i-width_]-b_values_[i+width_]) * (b_values_[i-width_]-b_values_[i+width_]);
            gradients_[i] = (dx + dy);
        }
    }
}

// ============================================================================
// Get the initial centers(seeds) based on given expected super pixel size.
// ============================================================================
void SuperPixel::getInitialCenters(const int& expected_seg_size) {
    // Step 1: evenly dispatch the initial seeds(centers).
    int x_strips = cvFloor(static_cast<double>(width_) / expected_seg_size);
    int y_strips = cvFloor(static_cast<double>(height_) / expected_seg_size);
    int x_err = width_ - expected_seg_size * x_strips;
    int y_err = height_ - expected_seg_size * y_strips;
    float x_err_per_strip = static_cast<float>(x_err) / x_strips;
    float y_err_per_strip = static_cast<float>(y_err) / y_strips;
    float x_offset = expected_seg_size / 2.0;
    float y_offset = expected_seg_size / 2.0;
    for (int y = 0; y < y_strips; ++y) {
        float y_err = y * y_err_per_strip;
        for (int x = 0; x < x_strips; ++x) {
            float x_err = x * x_err_per_strip;
            int x_pos = std::min(cvRound(x * expected_seg_size + x_offset + x_err), (width_ - 1));
            int y_pos = std::min(cvRound(y * expected_seg_size + y_offset + y_err), (height_ - 1));
            k_centers_x_.push_back(x_pos);
            k_centers_y_.push_back(y_pos);
            int position = y_pos * width_ + x_pos;
            k_centers_l_.push_back(l_values_[position]);
            k_centers_a_.push_back(a_values_[position]);
            k_centers_b_.push_back(b_values_[position]);
        }
    }

    // Step 2: Find local lowest gradient positions in 3x3 area and perturb seeds.
    const int kDx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int kDy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
    const int kSeedsNum = k_centers_l_.size();
    for (int n = 0; n < kSeedsNum; ++n) {
        int original_x = k_centers_x_[n];
        int original_y = k_centers_y_[n];
        int original_pos = original_y * width_ + original_x;
        int new_pos = original_pos;
        for (int i = 0; i < 8; ++i) {
            int temp_x = original_x + kDx8[i];
            int temp_y = original_y + kDy8[i];
            if (temp_x >= 0 && temp_x < width_ && temp_y >= 0 && temp_y < height_) {
                int temp_pos = temp_y * width_ + temp_x;
                if (gradients_[temp_pos] < gradients_[new_pos]) {
                    new_pos = temp_pos;
                }
            }
        }
        if (original_pos != new_pos) {
            k_centers_x_[n] = new_pos % width_;
            k_centers_y_[n] = new_pos / width_;
            k_centers_l_[n] = l_values_[new_pos];
            k_centers_a_[n] = a_values_[new_pos];
            k_centers_b_[n] = b_values_[new_pos];
        }
    }
}

// ============================================================================
// Iteratively do super pixel clustering.
// Need post-processing to enforce connectivity.
// ============================================================================
void SuperPixel::clusteringIteration(const int& expected_seg_size, const float& weight_m, int* temp_segmentation_map) {
    int const kTotalPixelNum = width_ * height_;
    const int kSeedsNum = k_centers_l_.size();
    const int kWindowOffset = expected_seg_size * 2;
    //
    // A set of variables containing the segmentation result information
    // of each iteration.
    //
    // The number of pixels dispatched to each center(seed).
    std::vector<int> cluster_size(kSeedsNum, 0);
    // To calculate the average value of color, we need to store the sum
    // of pixel colors.
    std::vector<float> sum_l(kSeedsNum, 0.0);
    std::vector<float> sum_a(kSeedsNum, 0.0);
    std::vector<float> sum_b(kSeedsNum, 0.0);
    // To calculate the geometric center of each cluster, we need to store
    // the sum of x/y offsets.
    std::vector<int> sum_x(kSeedsNum, 0);
    std::vector<int> sum_y(kSeedsNum, 0);
    // Store the distance from each pixel to its nearest clustering center.
    std::vector<float> min_distances(kTotalPixelNum, DBL_MAX);
    // The weighting variable between color hint and space(position) hint.
    float invert_weight = 1.0 /
                          ((expected_seg_size / weight_m) * (expected_seg_size / weight_m));
    // According to the original paper,
    // We need to set windows centered at the clustering centers,
    // and to look up all the pixels in the wondow for clustering.
    // Following variables define the window size and position.
    int x_start, y_start, x_end, y_end;
    // Temp variables for clustering.
    float l, a, b;
    float distance_color;
    float distance_space;
    float distance;
    for (int iter = 0; iter < 10; ++iter) {
        // According to the paper,the convergence error drops sharply in a
        // few iterations. They propose to run 10 iterations for experiemnts.
        for (int n = 0; n < kSeedsNum; ++n) {
            // Do clustering for each of the clusters (seeds).
            y_start = std::max(0, k_centers_y_[n] - kWindowOffset);
            y_end = std::min(height_, k_centers_y_[n] + kWindowOffset);
            x_start = std::max(0, k_centers_x_[n] - kWindowOffset);
            x_end = std::min(width_, k_centers_x_[n] + kWindowOffset);
            for (int row = y_start; row < y_end; ++row) {
                for (int col = x_start; col < x_end; ++col) {
                    int pos = row * width_ + col;
                    l = l_values_[pos];
                    a = a_values_[pos];
                    b = b_values_[pos];
                    distance_color = (l - k_centers_l_[n]) * (l - k_centers_l_[n]) +
                                     (a - k_centers_a_[n]) * (a - k_centers_a_[n]) +
                                     (b - k_centers_b_[n]) * (b - k_centers_b_[n]);
                    distance_space = (col - k_centers_x_[n]) * (col - k_centers_x_[n]) +
                                     (row - k_centers_y_[n]) * (row - k_centers_y_[n]);
                    distance = distance_color + distance_space * invert_weight;
                    if (distance < min_distances[pos]) {
                        min_distances[pos] = distance;
                        temp_segmentation_map[pos] = n;
                    }
                }
            }
        }
        // After assigning pixels, recalculate the cluster centers for next iter.
        sum_l.assign(kSeedsNum, 0.0);
        sum_a.assign(kSeedsNum, 0.0);
        sum_b.assign(kSeedsNum, 0.0);
        sum_x.assign(kSeedsNum, 0);
        sum_y.assign(kSeedsNum, 0);
        cluster_size.assign(kSeedsNum, 0);
        for (int row = 0; row < height_; ++row) {
            for (int col = 0; col < width_; ++col) {
                int pos = row * width_ + col;
                sum_l[temp_segmentation_map[pos]] += l_values_[pos];
                sum_a[temp_segmentation_map[pos]] += a_values_[pos];
                sum_b[temp_segmentation_map[pos]] += b_values_[pos];
                sum_y[temp_segmentation_map[pos]] += row;
                sum_x[temp_segmentation_map[pos]] += col;
                cluster_size[temp_segmentation_map[pos]] += 1;
            }
        }
        for (int k = 0; k < kSeedsNum; ++k) {
            if (cluster_size[k] <= 0) cluster_size[k] = 1;
            k_centers_l_[k] = sum_l[k] / cluster_size[k];
            k_centers_a_[k] = sum_a[k] / cluster_size[k];
            k_centers_b_[k] = sum_b[k] / cluster_size[k];
            k_centers_x_[k] = std::min(width_ - 1, cvRound(sum_x[k] / cluster_size[k]));
            k_centers_y_[k] = std::min(height_ - 1, cvRound(sum_y[k] / cluster_size[k]));
        }
    }
}

// ============================================================================
// Find next connected components(pixel) which belongs to the same cluster.
// This is called recursively to get the size of connected area cluster.
// ============================================================================
void SuperPixel::findNext(const int* temp_segmentation_map,
                          const int& row_index,
                          const int& col_index,
                          const int& segment_index,
                          int* x_pos,
                          int* y_pos,
                          int* count) {
    const int kDx4[4] = {-1, 0, 1, 0};
    const int kDy4[4] = {0, -1, 0, 1};
    int old_index = temp_segmentation_map[row_index * width_ + col_index];
    for (int i = 0; i < 4; ++i) {
        int col_new = col_index + kDx4[i];
        int row_new = row_index + kDy4[i];
        // Find a connected pixel belong to the same segment
        // in temp_segmentation_map.
        if ((row_new < height_ && row_new >= 0) &&
            (col_new < width_ && col_new >= 0)) {
            int new_pos = row_new * width_ + col_new;
            if (segmentation_map_[new_pos] < 0 &&
                temp_segmentation_map[new_pos] == old_index) {
                x_pos[*count] = col_new;
                y_pos[*count] = row_new;
                *count = *count + 1;
                segmentation_map_[new_pos] = segment_index;
                findNext(temp_segmentation_map, row_new, col_new, segment_index, x_pos, y_pos, count);
            }
        }
    }
}

// ============================================================================
// Post-processing. Enforce connectivity.
// After clustering iterations, a few stray labels may remian.
// That is, a few pixels in the vicinity of a large segment having the same
// label but not connected to it. We enforce connectivity finally by relabeling
// disjoint segments with the labels of the largest neighboring cluster.
// ============================================================================
void SuperPixel::enforceConnectivity(const int* temp_segmentation_map,
                                     const int& expected_seg_size) {
    const int kDx4[4] = {-1, 0, 1, 0};
    const int kDy4[4] = {0, -1, 0, 1};
    const int kAverageSize = expected_seg_size * expected_seg_size;
    const int kTotalPixelNum = width_ * height_;
    int segment_index = 0;
    int i = 0;
    int adjacent_index = 0;
    int* x_pos = new int[kTotalPixelNum];
    int* y_pos = new int[kTotalPixelNum];

    for (int row = 0; row < height_; ++row) {
        for (int col = 0; col < width_; ++col) {
            // We initialize all the elements in segmentation_map as -1.
            // Then by Traversing all the pixels, we assign them with segment
            // indexes. Since segmentation_map_ only contains
            // the final segmentation result, if it is less than 0, we
            // need to process the current pixel and get its segmentation result.
            if (segmentation_map_[i] < 0) {
                segmentation_map_[i] = segment_index;
                // Step 1:
                // Quickly find an adjacent label for use later if needed.
                for (int n = 0; n < 4; n++) {
                    int x = col + kDx4[n];
                    int y = row + kDy4[n];
                    if ((x >= 0 && x < width_) && (y >= 0 && y < height_)) {
                        int pos = y * width_ + x;
                        if (segmentation_map_[pos] >= 0)
                            adjacent_index = segmentation_map_[pos];
                    }
                }
                // Step 2: traverse from the current pixel and find all the
                // connected components. Store their x and y positions in
                // "x_pos" and "y_pos". "*count" is the number of pixels in
                // current segment.
                x_pos[0] = col;
                y_pos[0] = row;
                // Store number of pixels in current segment.
                int num_of_pixels = 1;
                int* count;
                count = &num_of_pixels;
                findNext(temp_segmentation_map, row, col, segment_index, x_pos, y_pos, count);
                // Step 3: check if current segment is too small.
                // The limit is defined as half of the expected super pixel size.
                // If the current segment is too small, replace it with adjacent
                // pixel's segment index.
                if (num_of_pixels <= (kAverageSize >> 2)) {
                    for (int c = 0; c < num_of_pixels; ++c) {
                        int ind = y_pos[c] * width_ + x_pos[c];
                        // Replace the segmentation label with adjacent pixel's label.
                        segmentation_map_[ind] = adjacent_index;
                    }
                    segment_index--;
                }
                segment_index++;
            }
            i++;
        }
    }
    num_segments_ = segment_index;
    if (x_pos) {
        delete [] x_pos;
        x_pos = NULL;
    }
    if (y_pos) {
        delete [] y_pos;
        y_pos = NULL;
    }
}
