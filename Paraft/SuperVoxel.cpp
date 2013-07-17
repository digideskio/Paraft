#include "SuperVoxel.h"

//SuperVoxel::SuperVoxel(const Metadata &meta) {
//    pImage_ = cvLoadImage(meta.tfPath().c_str(), CV_LOAD_IMAGE_COLOR);
//    if (!pImage_ || pImage_->nChannels != 3 || pImage_->depth != IPL_DEPTH_8U) {
//        cerr << "Error - Unsupport image format." << endl;
//        exit(EXIT_FAILURE);
//    }

//    dim_ = vector3i(pImage_->width, pImage_->height, 1);
//    numVoxels_ = dim_.VolumeSize();

//    mask_ = vector<int>(numVoxels_, -1);
//    maskTmp_ = vector<int>(numVoxels_, -1);

//    ls_ = vector<float>(numVoxels_, 0.0f);
//    as_ = vector<float>(numVoxels_, 0.0f);
//    bs_ = vector<float>(numVoxels_, 0.0f);
//    gradients_ = vector<float>(numVoxels_, 0.0f);
//}

SuperVoxel::SuperVoxel(const vector3i dim) : dim_(dim) {
    numVoxels_ = dim_.VolumeSize();
    mask_      = vector<int>(numVoxels_, -1);
    maskTmp_   = vector<int>(numVoxels_, -1);
    gradients_ = vector<float>(numVoxels_, 0.0f);
}

SuperVoxel::~SuperVoxel() {
    if (pImage_)    { cvReleaseImage(&pImage_); pImage_ = nullptr; }
    if (pData_)     { delete [] pData_; pData_ = nullptr; }
}

void SuperVoxel::ClusterByNumber(const int numClusters, const float compactness) {
    const int segLength = util::round(powf(static_cast<float>(numVoxels_)/numClusters, 1.0f/3.0f));
    ClusterBySize(segLength, compactness);
}

void SuperVoxel::ClusterBySize(const int segLength, const float compactness) {

    calculateGradientsForEachVoxel();
    dispatchInitialSeeds(segLength);
    perturbSeedsToLocalMinimumGradient();
    clustering(segLength, compactness);
    enforceConnectivity(segLength);

    if (numClusters_ == 0) {
        cout << "no cluster found..." << endl; return;
    }

    // 5. generate segmentation results
    for (int i = 0; i < numClusters_; i++) {
        Cluster cluster = { vector3i() /*center*/, 0 /*numVoxels*/ };
        clusterInfo_.push_back(cluster);
    }

    for (int z = 0; z < dim_.y; z++) {
        for (int y = 0; y < dim_.y; y++) {
            for (int x = 0; x < dim_.x; x++) {
                vector3i currentPos(x,y,z);
                int clusterIndex = mask_[GetVoxelIndex(currentPos)];
                clusterInfo_[clusterIndex].center += currentPos;
                clusterInfo_[clusterIndex].numVoxels++;
            }
        }
    }

    for (int i = 0; i < numClusters_; i++) {
        clusterInfo_[i].center.x = util::round(static_cast<float>(clusterInfo_[i].center.x) / clusterInfo_[i].numVoxels);
        clusterInfo_[i].center.y = util::round(static_cast<float>(clusterInfo_[i].center.y) / clusterInfo_[i].numVoxels);
        clusterInfo_[i].center.z = util::round(static_cast<float>(clusterInfo_[i].center.z) / clusterInfo_[i].numVoxels);
        clusterInfo_[i].center.x = std::min(clusterInfo_[i].center.x, dim_.x - 1);
        clusterInfo_[i].center.y = std::min(clusterInfo_[i].center.y, dim_.y - 1);
        clusterInfo_[i].center.z = std::min(clusterInfo_[i].center.z, dim_.z - 1);
    }

    cout << "~~" << endl;
}

// 1.
void SuperVoxel::calculateGradientsForEachVoxel() {
//    cout << "data.size: " << data_.size() << endl;
    for (int z = 1; z < dim_.z - 1; z++) {
        for (int y = 1; y < dim_.y - 1; y++) {
            for (int x = 1; x < dim_.x - 1; x++) {
                int self   = GetVoxelIndex(vector3i(x, y, z));
                int left   = GetVoxelIndex(vector3i(x-1,y,z));
                int right  = GetVoxelIndex(vector3i(x+1,y,z));
                int top    = GetVoxelIndex(vector3i(x,y+1,z));
                int bottom = GetVoxelIndex(vector3i(x,y-1,z));
                int front  = GetVoxelIndex(vector3i(x,y,z-1));
                int back   = GetVoxelIndex(vector3i(x,y,z+1));
                float dx   = (pData_[right] - pData_[left]) * (pData_[right] - pData_[left]);
                float dy   = (pData_[top] - pData_[bottom]) * (pData_[top] - pData_[bottom]);
                float dz   = (pData_[back] - pData_[front]) * (pData_[back] - pData_[front]);
                gradients_[self] = dx + dy + dz;
            }
        }
    }
}

// 2.
void SuperVoxel::dispatchInitialSeeds(int segLength) {
    vector3i numSeg = dim_ / segLength;            // |--1--|--2--|--3--|-- numSeg = 3
    vector3i remains = dim_ - numSeg * segLength;  //                    -- remains = 2
    float roundFactor = static_cast<float>(segLength) / 2.0f;

    for (int z = 0; z < numSeg.z; z++) {
        float zErr = static_cast<float>(z * remains.z) / numSeg.z;      // avg remain per segment
        for (int y = 0; y < numSeg.y; y++) {
            float yErr = static_cast<float>(y * remains.y) / numSeg.y;
            for (int x = 0; x < numSeg.x; x++) {
                float xErr = static_cast<float>(x * remains.x) / numSeg.x;
                vector3i segIndex; {
                    segIndex.x = std::min(util::round(x * segLength + roundFactor + xErr), dim_.x - 1);
                    segIndex.y = std::min(util::round(y * segLength + roundFactor + yErr), dim_.y - 1);
                    segIndex.z = std::min(util::round(z * segLength + roundFactor + zErr), dim_.z - 1);
                }
                int index = GetVoxelIndex(vector3i(x,y,z));

                centroidCoords_.push_back(segIndex);
                centroidValues_.push_back(pData_[index]);
            }
        }
    }

    cout << "centroid size = " << centroidCoords_.size() << endl;
}

// 3. optional
void SuperVoxel::perturbSeedsToLocalMinimumGradient() {
    for (size_t n = 0; n < centroidCoords_.size(); n++) {
        vector3i originalCentroid = centroidCoords_[n];
        int index = GetVoxelIndex(originalCentroid);
        int indexNew = index;
        for (auto neighbor : kNeighbors_) {
            vector3i neighborPos = originalCentroid + neighbor;
            if (neighborPos.x >= 0 && neighborPos.x < dim_.x &&
                neighborPos.y >= 0 && neighborPos.y < dim_.y &&
                neighborPos.z >= 0 && neighborPos.z < dim_.z) {
                int indexTemp = GetVoxelIndex(neighborPos);
                if (gradients_[indexTemp] < gradients_[indexNew]) {
                    indexNew = indexTemp;   // index of least gradient after iteration.
                }
            }
        }

        if (index != indexNew) {  // shifted to neighbor with less gradient
            centroidCoords_[n] = GetVoxelPosition(n);
            centroidValues_[n] = pData_[indexNew];
        }
    }
}

// 4. iteratively do super pixel clustering
void SuperVoxel::clustering(int segLength, float compactness) {
    const int kNumSeeds = centroidCoords_.size();
    const int kLocalWindowsSize = segLength * 2;

    // A set of variables containing the segmentation result information of each iteration.
    std::vector<int> clusterSize(kNumSeeds, 0);                 // number of voxels dispatched to each cluster(seed)
    std::vector<float> sumValues(kNumSeeds, 0.0f);              // for calculating average value
    std::vector<float> minDistances(numVoxels_, DBL_MAX);    // minDist(voxel, nearest cluster center)
    std::vector<vector3i> sumCentroids(kNumSeeds, vector3i());  // for calculating cluster center

    float weight = static_cast<float>(segLength) / compactness; // weight intensity : position !
    weight = 1.0f / (weight * weight);

    for (int iter = 0; iter < 10; iter++) {
        for (int n = 0; n < kNumSeeds; n++) {
            // windows range
            vector3i min, max; {
                min.x = std::max(centroidCoords_[n].x - kLocalWindowsSize, 0);
                min.y = std::max(centroidCoords_[n].y - kLocalWindowsSize, 0);
                min.z = std::max(centroidCoords_[n].z - kLocalWindowsSize, 0);
                max.x = std::min(centroidCoords_[n].x + kLocalWindowsSize, dim_.x);
                max.y = std::min(centroidCoords_[n].y + kLocalWindowsSize, dim_.y);
                max.z = std::min(centroidCoords_[n].z + kLocalWindowsSize, dim_.z);
            }

            for (int z = min.z; z < max.z; z++) {
                for (int y = min.y; y < max.y; y++) {
                    for (int x = min.x; x < max.x; x++) {
                        vector3i currentPos(x,y,z);
                        vector3i centroidPos = centroidCoords_[n];

                        int currentIndex = GetVoxelIndex(currentPos);

                        float distValue = pData_[currentIndex] - centroidValues_[n];
                        float distSpace = (currentPos - centroidPos).MagnituteSquared();
                        float distance = distValue + distSpace * weight;

                        if (distance < minDistances[currentIndex]) {
                            minDistances[currentIndex] = distance;
                            maskTmp_[currentIndex] = n;
                        }
                    }
                }
            }
        }

        // update cluster centers for next iteration
        sumValues.assign(kNumSeeds, 0.0f);
        sumCentroids.assign(kNumSeeds, vector3i());
        clusterSize.assign(kNumSeeds, 0);

        // todo: try not to re-interate over the whole data
        for (int z = 0; z < dim_.z; z++) {
            for (int y = 0; y < dim_.y; y++) {
                for (int x = 0; x < dim_.x; x++) {
                    vector3i pos(x,y,z);
                    int clusterIndex = maskTmp_[GetVoxelIndex(pos)];
                    sumValues[clusterIndex] += pData_[clusterIndex];
                    sumCentroids[clusterIndex] += pos;
                    clusterSize[clusterIndex]++;
                }
            }
        }

        for (int n = 0; n < kNumSeeds; n++) {
            int size = clusterSize[n];
            if (size <= 0) continue;

            centroidValues_[n] = sumValues[n] / size;
            centroidCoords_[n].x = std::min(util::round(static_cast<float>(sumCentroids[n].x)/size), dim_.x);
            centroidCoords_[n].y = std::min(util::round(static_cast<float>(sumCentroids[n].y)/size), dim_.y);
            centroidCoords_[n].z = std::min(util::round(static_cast<float>(sumCentroids[n].z)/size), dim_.z);
        }
    }
}

void SuperVoxel::enforceConnectivity(int segLength) {
    const int kAverageClusterSize = segLength * segLength * segLength;
    int clusterIndex = 0;
    int neighborClusterIndex = 0;
    int i = 0;  // ?
    std::vector<vector3i> pos(numVoxels_, vector3i());

    for (int z = 0; z < dim_.z; z++) {
        for (int y = 0; y < dim_.y; y++) {
            for (int x = 0; x < dim_.x; x++) {
                int index = GetVoxelIndex(vector3i(x,y,z));

                if (mask_[i] < 0) {    // all init to as -1
                    mask_[i] = clusterIndex;
                    // 1. find neighboring labels
                    vector3i currentPos(x,y,z);
                    for (auto neighbor : kNeighbors_) {
                        vector3i neighborPos = currentPos + neighbor;
                        if (neighborPos.x >= 0 && neighborPos.x < dim_.x &&
                            neighborPos.y >= 0 && neighborPos.y < dim_.y &&
                            neighborPos.z >= 0 && neighborPos.z < dim_.z) {
                            int neighborIndex = GetVoxelIndex(neighborPos);
                            if (mask_[neighborIndex] >= 0) {
                                neighborClusterIndex = mask_[neighborIndex];
                            }
                        }
                    }
                    // 2. traverse from the current pixel and find all the connected components.
                    pos[0] = vector3i(x,y,z);
                    int numVoxelsInCluster = 1;
                    int *count;
                    count = &numVoxelsInCluster;
                    //////////////////////////////////////////////////////////////
                    grow(currentPos, clusterIndex, pos, count);
                    //////////////////////////////////////////////////////////////
                    // 3. check if current cluster is too small
                    if (numVoxelsInCluster <= kAverageClusterSize * 0.25) {
                        for (int i = 0; i < numVoxelsInCluster; i++) {
                            int index = GetVoxelIndex(pos[i]);
                            mask_[index] = neighborClusterIndex;
                        }
                        clusterIndex--;
                        cout << "too small" << endl;
                    }
                    clusterIndex++;
                }
                i++;
            }
        }
    }

    numClusters_ = clusterIndex;
}

void SuperVoxel::grow(const vector3i &seed, const int clusterIndex, std::vector<vector3i> &pos, int *count) {
    int seedIndex = GetVoxelIndex(seed);
    for (auto offset : kNeighbors_) {
        vector3i neighbor = seed + offset;
        if (neighbor.x >= 0 && neighbor.x < dim_.x &&
            neighbor.y >= 0 && neighbor.y < dim_.y &&
            neighbor.z >= 0 && neighbor.z < dim_.z) {
            int neighborIndex = GetVoxelIndex(neighbor);
            if (mask_[neighborIndex] < 0 && maskTmp_[neighborIndex] == maskTmp_[seedIndex]) {
                mask_[neighborIndex] = clusterIndex;
                pos[*count] = neighbor;
                *count = *count + 1;
                grow(neighbor, clusterIndex, pos, count);
            }
        }
    }
}

void SuperVoxel::growRegion(const vector3i& seed, const int clusterIndex, std::vector<vector3i>& pos, int& count) {
    int seedIndex = GetVoxelIndex(seed);
    for (auto neighbor : kNeighbors_) {
        vector3i neighborPos = seed + neighbor;
        if (neighborPos.x >= 0 && neighborPos.x < dim_.x &&
            neighborPos.y >= 0 && neighborPos.y < dim_.y &&
            neighborPos.z >= 0 && neighborPos.z < dim_.z) {
            int neighborIndex = GetVoxelIndex(neighborPos);
            if (mask_[neighborIndex] < 0 && maskTmp_[neighborIndex] == maskTmp_[seedIndex]) {
                mask_[neighborIndex] = clusterIndex;
                pos[count++] = neighborPos;
                growRegion(neighborPos, clusterIndex, pos, count);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------- //

void SuperVoxel::SegmentByNumber(const int expectedClusterNum, const float compactness) {
    int clusterSize = static_cast<int>(sqrt(numVoxels_ / expectedClusterNum));
    if (clusterSize > 100) {
        cerr << "Warning - cluster size is too large." << endl; return;
    }
    SegmentBySize(clusterSize, compactness);
}

void SuperVoxel::SegmentBySize(const int expectedClusterSize, const float compactness) {
    // 1. RGB to LAB
    bgr2lab();

    // 2. Spread seeds
    detectGradients();
    getInitialCenters(expectedClusterSize);

    // 3. clustering
    clusteringIteration(expectedClusterSize, compactness);

    // 4. enforce connectivity
    enforceConnect(expectedClusterSize);

    // 5. generate segmentation results
    for (int i = 0; i < numClusters_; i++) {
        Cluster cluster; {
            cluster.center.x = 0;
            cluster.center.y = 0;
            cluster.numVoxels = 0;
        }
        clusterInfo_.push_back(cluster);
    }
    for (int y = 0; y < dim_.y; y++) {
        for (int x = 0; x < dim_.x; x++) {
            int pos = y * dim_.x + x;
            int index = mask_[pos];
            clusterInfo_[index].center.x += x;
            clusterInfo_[index].center.y += y;
            clusterInfo_[index].numVoxels++;
        }
    }
    for (int i = 0; i < numClusters_; i++) {
        clusterInfo_[i].center.x = util::round(static_cast<float>(clusterInfo_[i].center.x) / clusterInfo_[i].numVoxels);
        clusterInfo_[i].center.y = util::round(static_cast<float>(clusterInfo_[i].center.y) / clusterInfo_[i].numVoxels);
        clusterInfo_[i].center.x = std::min(clusterInfo_[i].center.x, dim_.x - 1);
        clusterInfo_[i].center.y = std::min(clusterInfo_[i].center.y, dim_.y - 1);
    }

    // 6. cleaning
    lcenters_.clear();
    acenters_.clear();
    bcenters_.clear();
    xcenters_.clear();
    ycenters_.clear();
}

// ============================================================================
// Testing function.
// Draw the contours of segmented areas on image
// ============================================================================
void SuperVoxel::DrawContours(const CvScalar& drawing_color, const std::string& save_path) {
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
    IplImage* contour = cvCreateImage(cvSize(dim_.x, dim_.y),
                                      pImage_->depth, pImage_->nChannels);
    cvCopy(pImage_, contour);
    int step = contour->widthStep;
    uchar* data = reinterpret_cast<uchar*>(contour->imageData);
    int const kTotalPixelNum = dim_.x * dim_.y;
    std::vector<bool> istaken(kTotalPixelNum, false);
    for (int row = 0; row < dim_.y; ++row) {
        for (int col = 0; col < dim_.x; ++col) {
            int diff = 0;
            int pos_a = row * dim_.x + col;
            for (int i = 0; i < 8; ++i) {
                int x = col + dx8[i];
                int y = row + dy8[i];
                if ((x >= 0 && x < dim_.x) && (y >= 0 && y < dim_.y)) {
                    int pos_b = y * dim_.x + x;
                    if ((false == istaken[pos_a]) &&
                        (mask_[pos_a] != mask_[pos_b])) {
                        ++diff;
                    }
                }
            }
            if (diff >= 2) {
                istaken[pos_a] = true;
                data[row * step + col * 3 + 0] = drawing_color.val[0];
                data[row * step + col * 3 + 1] = drawing_color.val[1];
                data[row * step + col * 3 + 2] = drawing_color.val[2];
            }
        }
    }
    cvSaveImage(save_path.c_str(), contour);
    cvReleaseImage(&contour);
}

void SuperVoxel::bgr2lab() {
    int step = pImage_->widthStep;
    uchar* data = reinterpret_cast<uchar*>(pImage_->imageData);
    double epsilon = 0.008856;
    double kappa   = 903.3;
    double Xr = 0.950456;
    double Yr = 1.0;
    double Zr = 1.088754;
    for (int y = 0; y < dim_.y; y++) {
        for (int x = 0; x < dim_.x; x++) {
            // Access pixel values.
            double B = static_cast<double>(data[y * step + x * 3 + 0]) / 255.0;
            double G = static_cast<double>(data[y * step + x * 3 + 1]) / 255.0;
            double R = static_cast<double>(data[y * step + x * 3 + 2]) / 255.0;

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

            double fx = xr > epsilon ? pow(xr, 1.0/3.0) : (kappa*xr + 16.0) / 116.0;
            double fy = yr > epsilon ? pow(yr, 1.0/3.0) : (kappa*yr + 16.0) / 116.0;
            double fz = zr > epsilon ? pow(zr, 1.0/3.0) : (kappa*zr + 16.0) / 116.0;

            // Add converted color to 1-D vectors.
            int pos = y * dim_.x + x;
            ls_[pos] = 116.0 * fy - 16.0;
            as_[pos] = 500.0 * (fx - fy);
            bs_[pos] = 200.0 * (fy - fz);
        }
    }
}

void SuperVoxel::detectGradients() {
    for (int y = 1; y < dim_.y - 1; y++) {
        for (int x = 1; x < dim_.x - 1; x++) {
            int i = GetVoxelIndex(vector3i(x,y));
            double dx = (ls_[i-1]-ls_[i+1]) * (ls_[i-1]-ls_[i+1]) +
                        (as_[i-1]-as_[i+1]) * (as_[i-1]-as_[i+1]) +
                        (bs_[i-1]-bs_[i+1]) * (bs_[i-1]-bs_[i+1]);
            double dy = (ls_[i-dim_.x]-ls_[i+dim_.x]) * (ls_[i-dim_.x]-ls_[i+dim_.x]) +
                        (as_[i-dim_.x]-as_[i+dim_.x]) * (as_[i-dim_.x]-as_[i+dim_.x]) +
                        (bs_[i-dim_.x]-bs_[i+dim_.x]) * (bs_[i-dim_.x]-bs_[i+dim_.x]);
            gradients_[i] = (dx + dy);
        }
    }
}

void SuperVoxel::getInitialCentroids(int expectedClusterSize) {
    // Step 1: calculate gradients for each voxel
    for (int z = 1; z < dim_.z - 1; z++) {
        for (int y = 1; y < dim_.y - 1; y++) {
            for (int x = 1; x < dim_.x - 1; x++) {
                int pos = GetVoxelIndex(vector3i(x+1,y,z));
                int neg = GetVoxelIndex(vector3i(x-1,y,z));
                float dx = (pData_[pos] - pData_[neg]) * (pData_[pos] - pData_[neg]);
                    pos = GetVoxelIndex(vector3i(x,y+1,z));
                    neg = GetVoxelIndex(vector3i(x,y-1,z));
                float dy = (pData_[pos] - pData_[neg]) * (pData_[pos] - pData_[neg]);
                    pos = GetVoxelIndex(vector3i(x,y,z+1));
                    neg = GetVoxelIndex(vector3i(x,y,z-1));
                float dz = (pData_[pos] - pData_[neg]) * (pData_[pos] - pData_[neg]);
                gradients_[GetVoxelIndex(vector3i(x,y,z))] = dx + dy + dz;
            }
        }
    }

    // Step 2: evenly dispatch the initial seeds (centroids).
    vector3i numSeg = dim_ / expectedClusterSize;            // |--1--|--2--|--3--|-- numSeg = 3
    vector3i remains = dim_ - numSeg * expectedClusterSize;  //                    -- remains = 2
    float roundFactor = static_cast<float>(expectedClusterSize) / 2.0f;   // like floor(value + 0.5);

    for (int z = 0; z < numSeg.z; z++) {
        float zErr = static_cast<float>(z * remains.z) / numSeg.z;      // avg remain per segment
        for (int y = 0; y < numSeg.y; y++) {
            float yErr = static_cast<float>(y * remains.y) / numSeg.y;
            for (int x = 0; x < numSeg.x; x++) {
                float xErr = static_cast<float>(x * remains.x) / numSeg.x;

                vector3i segIndex; {
                    segIndex.x = std::min(util::round(x * expectedClusterSize + roundFactor + xErr), dim_.x - 1);
                    segIndex.y = std::min(util::round(y * expectedClusterSize + roundFactor + yErr), dim_.y - 1);
                    segIndex.z = std::min(util::round(z * expectedClusterSize + roundFactor + zErr), dim_.z - 1);
                }
                int index = GetVoxelIndex(vector3i(x,y,z));

                centroidCoords_.push_back(segIndex);
                centroidValues_.push_back(pData_[index]);
            }
        }
    }

    // Step 3: Find local lowest gradient positions within 6 neighbors and perturb seeds.
    vector<vector3i> neighbors; {                   // six direct neighbors
        neighbors.push_back(vector3i(-1, 0, 0));    // left
        neighbors.push_back(vector3i( 1, 0, 0));    // right
        neighbors.push_back(vector3i( 0, 1, 0));    // top
        neighbors.push_back(vector3i( 0,-1, 0));    // bottom
        neighbors.push_back(vector3i( 0, 0,-1));    // front
        neighbors.push_back(vector3i( 0, 0, 1));    // back
    }

    for (unsigned int n = 0; n < centroidCoords_.size(); n++) {
        vector3i originalCentroid = centroidCoords_[n];
        int index = GetVoxelIndex(originalCentroid);
        int indexNew = index;
        for (unsigned int i = 0; i < neighbors.size(); i++) {
            vector3i tempCentroid = originalCentroid + neighbors[i];
            if (tempCentroid.x >= 0 && tempCentroid.x < dim_.x &&
                tempCentroid.y >= 0 && tempCentroid.y < dim_.y &&
                tempCentroid.z >= 0 && tempCentroid.z < dim_.z) {
                int indexTemp = GetVoxelIndex(tempCentroid);
                if (gradients_[indexTemp] < gradients_[indexNew]) {
                    indexNew = indexTemp;   // index of least gradient after iteration.
                }
            }
        }
        if (index != indexNew) {  // shifted to neighbor
            centroidCoords_[n] = GetVoxelPosition(n);
            centroidValues_[n] = centroidValues_[indexNew];
        }
    }
}

// ============================================================================
// Get the initial centers(seeds) based on given expected super pixel size.
// ============================================================================
void SuperVoxel::getInitialCenters(int expectedClusterSize) {
    // Step 1: evenly dispatch the initial seeds(centers).
    vector3i strips = dim_ / expectedClusterSize;               // |--1--|--2--|--3--|--    strips = 3
    vector3i deviat = dim_ - strips * expectedClusterSize;      // --                       deviat = 2
    vector3f offset = vector3f(expectedClusterSize/2.0, expectedClusterSize/2.0, expectedClusterSize/2.0);
    for (int y = 0; y < strips.y; y++) {
        float y_err = static_cast<float>(y) * deviat.y / strips.y;
        for (int x = 0; x < strips.x; x++) {
            float x_err = static_cast<float>(x) * deviat.x / strips.x;  // index * 2 / 5
            int x_pos = std::min(util::round(x * expectedClusterSize + offset.x + x_err), dim_.x - 1);
            int y_pos = std::min(util::round(y * expectedClusterSize + offset.y + y_err), dim_.y - 1);
            xcenters_.push_back(x_pos);
            ycenters_.push_back(y_pos);
            int position = y_pos * dim_.x + x_pos;
            lcenters_.push_back(ls_[position]);
            acenters_.push_back(as_[position]);
            bcenters_.push_back(bs_[position]);
        }
    }

    // Step 2: Find local lowest gradient positions in 3x3 area and perturb seeds.
    const int kDx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    const int kDy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
    const int kSeedsNum = lcenters_.size();
    for (int n = 0; n < kSeedsNum; n++) {
        int original_x = xcenters_[n];
        int original_y = ycenters_[n];
        int original_pos = original_y * dim_.x + original_x;
        int new_pos = original_pos;
        for (int i = 0; i < 8; ++i) {
            int temp_x = original_x + kDx8[i];
            int temp_y = original_y + kDy8[i];
            if (temp_x >= 0 && temp_x < dim_.x && temp_y >= 0 && temp_y < dim_.y) {
                int temp_pos = temp_y * dim_.x + temp_x;
                if (gradients_[temp_pos] < gradients_[new_pos]) {
                    new_pos = temp_pos;
                }
            }
        }
        if (original_pos != new_pos) {
            xcenters_[n] = new_pos % dim_.x;
            ycenters_[n] = new_pos / dim_.x;
            lcenters_[n] = ls_[new_pos];
            acenters_[n] = as_[new_pos];
            bcenters_[n] = bs_[new_pos];
        }
    }
}

// ============================================================================
// Iteratively do super pixel clustering.
// Need post-processing to enforce connectivity.
// ============================================================================
void SuperVoxel::clusteringIteration(int expectedClusterSize, float compactness) {
    const int kSeedsNum = lcenters_.size();
    const int kWindowOffset = expectedClusterSize * 2;

    // A set of variables containing the segmentation result information of each iteration.
    // The number of pixels dispatched to each center(seed).
    std::vector<int> cluster_size(kSeedsNum, 0);

    // To calculate the average value of color, we need to store the sum of pixel colors.
    std::vector<float> sum_l(kSeedsNum, 0.0);
    std::vector<float> sum_a(kSeedsNum, 0.0);
    std::vector<float> sum_b(kSeedsNum, 0.0);

    // To calculate the geometric center of each cluster, we need to store the sum of x/y offsets.
    std::vector<int> sum_x(kSeedsNum, 0);
    std::vector<int> sum_y(kSeedsNum, 0);

    // Store the distance from each pixel to its nearest clustering center.
    std::vector<float> min_distances(numVoxels_, DBL_MAX);

    // The weighting variable between color hint and space(position) hint.
    float weight = static_cast<float>(expectedClusterSize) / compactness;
//    float invert_weight = 1.0 / (weight * weight);

    // According to the paper ,we need to set windows centered at the clustering centers,
    // and to look up all the pixels in the wondow for clustering.
    // Following variables define the window size and position.
    for (int iter = 0; iter < 10; ++iter) {
        // According to the paper,the convergence error drops sharply in a
        // few iterations. They propose to run 10 iterations for experiemnts.
        for (int n = 0; n < kSeedsNum; ++n) {
            // Do clustering for each of the clusters (seeds).
            int x_start = std::max(0, xcenters_[n] - kWindowOffset);
            int y_start = std::max(0, ycenters_[n] - kWindowOffset);
            int x_end = std::min(dim_.x, xcenters_[n] + kWindowOffset);
            int y_end = std::min(dim_.y, ycenters_[n] + kWindowOffset);

            for (int y = y_start; y < y_end; y++) {
                for (int x = x_start; x < x_end; x++) {
                    int pos = y * dim_.x + x;
                    vector3f currentColor = vector3f(ls_[pos], as_[pos], bs_[pos]);
                    vector3f centerColor  = vector3f(lcenters_[n], acenters_[n], bcenters_[n]);
                    vector3f currentPos = vector3f(x, y, 0);
                    vector3f centerPos  = vector3f(xcenters_[n], ycenters_[n], 0.0);

                    float distanceColor = (currentColor - centerColor).MagnituteSquared();
                    float distanceSpace = (currentPos - centerPos).MagnituteSquared();
                    float distance = distanceColor + distanceSpace / (weight * weight);

                    if (distance < min_distances[pos]) {
                        min_distances[pos] = distance;
                        maskTmp_[pos] = n;
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
        for (int y = 0; y < dim_.y; y++) {
            for (int x = 0; x < dim_.x; x++) {
                int pos = y * dim_.x + x;
                sum_l[maskTmp_[pos]] += ls_[pos];
                sum_a[maskTmp_[pos]] += as_[pos];
                sum_b[maskTmp_[pos]] += bs_[pos];
                sum_y[maskTmp_[pos]] += y;
                sum_x[maskTmp_[pos]] += x;
                cluster_size[maskTmp_[pos]] += 1;
            }
        }
        for (int k = 0; k < kSeedsNum; k++) {
            if (cluster_size[k] <= 0) cluster_size[k] = 1;
            lcenters_[k] = sum_l[k] / cluster_size[k];
            acenters_[k] = sum_a[k] / cluster_size[k];
            bcenters_[k] = sum_b[k] / cluster_size[k];
            xcenters_[k] = std::min(dim_.x - 1, cvRound(sum_x[k] / cluster_size[k]));
            ycenters_[k] = std::min(dim_.y - 1, cvRound(sum_y[k] / cluster_size[k]));
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
void SuperVoxel::enforceConnect(int expectedClusterSize) {
    const int kDx4[4] = {-1, 0, 1, 0};
    const int kDy4[4] = {0, -1, 0, 1};
    const int kAverageSize = expectedClusterSize * expectedClusterSize;
    int i = 0;
    int clusterIndex = 0;
    int adjacent_index = 0;
    int* x_pos = new int[numVoxels_];
    int* y_pos = new int[numVoxels_];

    for (int y = 0; y < dim_.y; y++) {
        for (int x = 0; x < dim_.x; x++) {
            // We initialize all the elements in segmentation_map as -1.
            // Then by Traversing all the pixels, we assign them with segment
            // indexes. Since segmentation_map_ only contains
            // the final segmentation result, if it is less than 0, we
            // need to process the current pixel and get its segmentation result.
            if (mask_[i] < 0) {
                mask_[i] = clusterIndex;
                // Step 1:
                // Quickly find an adjacent label for use later if needed.
                for (int n = 0; n < 4; n++) {
                    int x = x + kDx4[n];
                    int y = y + kDy4[n];
                    if ((x >= 0 && x < dim_.x) && (y >= 0 && y < dim_.y)) {
                        int pos = y * dim_.x + x;
                        if (mask_[pos] >= 0) {
                            adjacent_index = mask_[pos];
                        }
                    }
                }
                // Step 2: traverse from the current pixel and find all the
                // connected components. Store their x and y positions in
                // "x_pos" and "y_pos". "*count" is the number of pixels in
                // current segment.
                x_pos[0] = x;
                y_pos[0] = y;
                // Store number of pixels in current segment.
                int num_of_pixels = 1;
                int* count;
                count = &num_of_pixels;
                findNext(x, y, clusterIndex, x_pos, y_pos, count);
                // Step 3: check if current segment is too small.
                // The limit is defined as half of the expected super pixel size.
                // If the current segment is too small, replace it with adjacent
                // pixel's segment index.
                if (num_of_pixels <= (kAverageSize >> 2)) {
                    for (int c = 0; c < num_of_pixels; ++c) {
                        int ind = y_pos[c] * dim_.x + x_pos[c];
                        // Replace the segmentation label with adjacent pixel's label.
                        mask_[ind] = adjacent_index;
                    }
                    clusterIndex--;
                }
                clusterIndex++;
            }
            i++;
        }
    }
    numClusters_ = clusterIndex;
    if (x_pos) {
        delete [] x_pos;
        x_pos = NULL;
    }
    if (y_pos) {
        delete [] y_pos;
        y_pos = NULL;
    }
}

// ============================================================================
// Find next connected components(pixel) which belongs to the same cluster.
// This is called recursively to get the size of connected area cluster.
// ============================================================================
void SuperVoxel::findNext(int x, int y, int clusterIndex, int* x_pos, int* y_pos, int* count) {
    const int kDx4[4] = {-1, 0, 1, 0};
    const int kDy4[4] = {0, -1, 0, 1};
    int old_index = maskTmp_[y * dim_.x + x];
    for (int i = 0; i < 4; ++i) {
        int col_new = x + kDx4[i];
        int row_new = y + kDy4[i];
        // Find a connected pixel belong to the same segment in pClustersTmp_.
        if ((row_new < dim_.y && row_new >= 0) && (col_new < dim_.x && col_new >= 0)) {
            int new_pos = row_new * dim_.x + col_new;
            if (mask_[new_pos] < 0 &&
                maskTmp_[new_pos] == old_index) {
                x_pos[*count] = col_new;
                y_pos[*count] = row_new;
                *count = *count + 1;
                mask_[new_pos] = clusterIndex;
                findNext(col_new, row_new, clusterIndex, x_pos, y_pos, count);
            }
        }
    }
}
