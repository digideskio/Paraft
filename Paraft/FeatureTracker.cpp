#include "FeatureTracker.h"

FeatureTracker::FeatureTracker(Vector3i dim) : blockDim(dim) {
    globalMaskValue = 0.0f;
    tfRes = 0;
    pTfMap = NULL;
    volumeSize = blockDim.Product();
    pMaskCurrent = new float[volumeSize]();
    pMaskPrevious = new float[volumeSize]();
}

FeatureTracker::~FeatureTracker() {
    delete [] pMaskCurrent;
    delete [] pMaskPrevious;

    if (featureSequence.size() > 0) {
        for (FeatureVectorSequence::iterator it = featureSequence.begin(); it != featureSequence.end(); it++) {
            vector<Feature> featureVector = it->second;
            for (size_t i = 0; i < featureVector.size(); i++) {
                Feature f = featureVector[i];
                f.edgeVoxels.clear();
                f.bodyVoxels.clear();
            }
        }
    }
}

void FeatureTracker::ExtractAllFeatures() {
    for (int z = 0; z < blockDim.z; z++) {
        for (int y = 0; y < blockDim.y; y++) {
            for (int x = 0; x < blockDim.x; x++) {
                int index = GetVoxelIndex(Vector3i(x, y, z));
                if (pMaskCurrent[index] > 0) {  // point already within a feature
                    continue;                   // most points should stop here
                }
                int tfIndex = (int)(pVolumeData[index] * (float)(tfRes-1));
                if (pTfMap[tfIndex] >= OPACITY_THRESHOLD) {
                    FindNewFeature(Vector3i(x,y,z));
                }
            }
        }
    }
}

void FeatureTracker::FindNewFeature(Vector3i seed) {
    Feature f; {
        f.id         = 0;
        f.numVoxels  = 0;
        f.centroid   = Vector3i();
        f.edgeVoxels = list<Vector3i>();
        f.bodyVoxels = list<Vector3i>();
        f.maskValue  = globalMaskValue += 1.0f;
    }

    f.edgeVoxels.push_back(seed);
    f.numVoxels = 1;
    pMaskCurrent[GetVoxelIndex(seed)] = f.maskValue;

    expandRegion(f);

    if (f.numVoxels < MIN_NUM_VOXEL_IN_FEATURE) {
        globalMaskValue -= 1.0f; return;
    }

    currentFeatures.push_back(f);
    backup1Features = currentFeatures;
    backup2Features = currentFeatures;
    backup3Features = currentFeatures;
}

void FeatureTracker::TrackFeature(float* pData, int direction, int mode) {
    if (pTfMap == NULL || tfRes <= 0) {
        cout << "Set TF pointer first." << endl; exit(3);
    }

    pVolumeData = pData;

    // save current 0-1 matrix to previous, then clear current maxtrix
    std::copy(pMaskCurrent, pMaskCurrent+volumeSize, pMaskPrevious);
    std::fill(pMaskCurrent, pMaskCurrent+volumeSize, 0);

    for (size_t i = 0; i < currentFeatures.size(); i++) {
        Feature f = currentFeatures[i];

        Vector3i offset = predictRegion(i, direction, mode);
        fillRegion(f, offset);
        shrinkRegion(f);
        expandRegion(f);

        if (f.numVoxels == 0) {
            // todo f.numVoxels < MIN_NUM_VOXEL_IN_FEATURE
            // currentFeaturesHolder.erase(currentFeaturesHolder.begin()+i);
            continue;
        }

        f.centroid /= f.numVoxels;
        f.id = GetVoxelIndex(f.centroid);
        currentFeatures[i] = f;
    }

    backupFeatureInfo(direction);
    ExtractAllFeatures();
}

inline Vector3i FeatureTracker::predictRegion(int index, int direction, int mode) {
    int timestepsAvailable = direction == FT_BACKWARD ? timeLeft2Backward : timeLeft2Forward;

    Vector3i off;
    Feature b1f = backup1Features[index];
    Feature b2f = backup2Features[index];
    Feature b3f = backup3Features[index];

    int tmp;
    switch (mode) {
        case FT_DIRECT: // PREDICT_DIRECT
            break;
        case FT_LINEAR: // PREDICT_LINEAR
            if (timestepsAvailable > 1) {
                if (direction == FT_BACKWARD) {
                    off = b2f.centroid - b1f.centroid;
                } else {  // Tracking forward as default
                    off = b3f.centroid - b2f.centroid;
                }
                for (list<Vector3i>::iterator p = b3f.edgeVoxels.begin(); p != b3f.edgeVoxels.end(); p++) {
                    tmp = (*p).x + (int)floor(off.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim.x ? tmp : blockDim.x-1);
                    tmp = (*p).y + (int)floor(off.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim.y ? tmp : blockDim.y-1);
                    tmp = (*p).z + (int)floor(off.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim.z ? tmp : blockDim.z-1);
                }
            }
        break;
        case FT_POLYNO: // PREDICT_POLY
            if (timestepsAvailable > 1) {
                if (timestepsAvailable > 2) {
                    off = b3f.centroid*2 - b2f.centroid*3 + b1f.centroid;
                } else {    // [1,2)
                    if (direction == FT_BACKWARD) {
                        off = b2f.centroid - b1f.centroid;
                    } else {  // Tracking forward as default
                        off = b3f.centroid - b2f.centroid;
                    }
                }
                for (list<Vector3i>::iterator p = b3f.edgeVoxels.begin(); p != b3f.edgeVoxels.end(); p++) {
                    tmp = (*p).x + (int)floor(off.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim.x ? tmp : blockDim.x-1);
                    tmp = (*p).y + (int)floor(off.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim.y ? tmp : blockDim.y-1);
                    tmp = (*p).z + (int)floor(off.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim.z ? tmp : blockDim.z-1);
                }
            }
        break;
    }
    return off;
}

inline void FeatureTracker::fillRegion(Feature &f, const Vector3i &offset) {
//    cout << f.numVoxels << " -> ";
    f.numVoxels = 0;
//    cout << f.numVoxels << endl;

    // predicted to be on edge
    for (list<Vector3i>::iterator p = f.edgeVoxels.begin(); p != f.edgeVoxels.end(); p++) {
        int index = GetVoxelIndex(*p);
        if (pMaskCurrent[index] == 0) {
            pMaskCurrent[index] = f.maskValue;
        }
        f.bodyVoxels.push_back(*p);
        f.centroid += (*p);
        f.numVoxels++;
    }

    // currently not on edge but previously on edge
    for (list<Vector3i>::iterator p = f.edgeVoxels.begin(); p != f.edgeVoxels.end(); p++) {
        int index = GetVoxelIndex(*p);
        int indexPrev = GetVoxelIndex((*p)-offset);
        while ((*p).x >= 0 && (*p).x <= blockDim.x && (*p).x - offset.x >= 0 && (*p).x - offset.x <= blockDim.x &&
               (*p).y >= 0 && (*p).y <= blockDim.y && (*p).y - offset.y >= 0 && (*p).y - offset.y <= blockDim.y &&
               (*p).z >= 0 && (*p).z <= blockDim.z && (*p).z - offset.z >= 0 && (*p).z - offset.z <= blockDim.z &&
               pMaskCurrent[index] == 0 && pMaskPrevious[indexPrev] == f.maskValue) {

            // Mark all points: 1. currently = 1; 2. currently = 0 but previously = 1;
            pMaskCurrent[index] = f.maskValue;
            f.bodyVoxels.push_back(*p);
            f.centroid += (*p);
            f.numVoxels++;
        }
    }
}

inline void FeatureTracker::shrinkRegion(Feature &f) {
    // mark all edge points as 0
    while (f.edgeVoxels.empty() == false) {
        Vector3i seed = f.edgeVoxels.front();
        f.edgeVoxels.pop_front();
        shrinkEdge(f, seed);
    }

    while (!f.bodyVoxels.empty()) {
        Vector3i seed = f.bodyVoxels.front();
        f.bodyVoxels.pop_front();

        int index = GetVoxelIndex(seed);
        bool isPointOnEdge = false;
        if (getOpacity(pVolumeData[index]) < OPACITY_THRESHOLD) {
            isPointOnEdge = false;
            // if point is invisible, mark its adjacent points as 0
            shrinkEdge(f, seed);                                              // center
            if (++seed.x < blockDim.x) { shrinkEdge(f, seed); } seed.x--;   // right
            if (++seed.y < blockDim.y) { shrinkEdge(f, seed); } seed.y--;   // top
            if (++seed.z < blockDim.z) { shrinkEdge(f, seed); } seed.z--;   // back
            if (--seed.x >= 0)         { shrinkEdge(f, seed); } seed.x++;   // left
            if (--seed.y >= 0)         { shrinkEdge(f, seed); } seed.y++;   // bottom
            if (--seed.z >= 0)         { shrinkEdge(f, seed); } seed.z++;   // front
        } else if (pMaskCurrent[index] == 0) {
            isPointOnEdge = true;
        }

        if (isPointOnEdge) {
            f.edgeVoxels.push_back(seed);
        }
    }

    for (list<Vector3i>::iterator p = f.edgeVoxels.begin(); p != f.edgeVoxels.end(); p++) {
        int index = GetVoxelIndex(*p);
        if (pMaskCurrent[index] != f.maskValue) {
            pMaskCurrent[index] = f.maskValue;
            f.bodyVoxels.push_back(*p);
            f.centroid += (*p);
            f.numVoxels++;
        }
    }
}

inline void FeatureTracker::shrinkEdge(Feature &f, const Vector3i &seed) {
    int index = GetVoxelIndex(seed);
    if (pMaskCurrent[index] == f.maskValue) {
        pMaskCurrent[index] = 0;  // shrink
        list<Vector3i>::iterator p = find(f.bodyVoxels.begin(), f.bodyVoxels.end(), seed);
        f.bodyVoxels.erase(p);
        f.edgeVoxels.push_back(seed);
        f.centroid -= seed;
        f.numVoxels--;
    }
}

inline void FeatureTracker::expandRegion(Feature &f) {
    list<Vector3i> tempVoxels;    // to store updated edge voxels
    while (!f.edgeVoxels.empty()) {
        Vector3i seed = f.edgeVoxels.front();
        f.edgeVoxels.pop_front();
        bool voxelOnEdge = false;
        if (++seed.x < blockDim.x) { voxelOnEdge |= expandEdge(f, seed); } seed.x--;  // right
        if (++seed.y < blockDim.y) { voxelOnEdge |= expandEdge(f, seed); } seed.y--;  // top
        if (++seed.z < blockDim.z) { voxelOnEdge |= expandEdge(f, seed); } seed.z--;  // front
        if (--seed.x >= 0)         { voxelOnEdge |= expandEdge(f, seed); } seed.x++;  // left
        if (--seed.y >= 0)         { voxelOnEdge |= expandEdge(f, seed); } seed.y++;  // bottom
        if (--seed.z >= 0)         { voxelOnEdge |= expandEdge(f, seed); } seed.z++;  // back
        if (voxelOnEdge)           { tempVoxels.push_back(seed); }
    }
    f.edgeVoxels.swap(tempVoxels);
}

inline bool FeatureTracker::expandEdge(Feature &f, const Vector3i &seed) {
    int index = GetVoxelIndex(seed);
    if (pMaskCurrent[index] > 0 || getOpacity(pVolumeData[index]) < OPACITY_THRESHOLD) {
        return false;
    }  // already labeled by a feature or opacity not large enough to be labeled

    pMaskCurrent[index] = f.maskValue;
    f.edgeVoxels.push_back(seed);
    f.bodyVoxels.push_back(seed);
    f.centroid += seed;
    f.numVoxels++;
    return true;
}

void FeatureTracker::backupFeatureInfo(int direction) {
    backup1Features = backup2Features;
    backup2Features = backup3Features;
    backup3Features = currentFeatures;

    if (direction == FT_FORWARD) {
        if (timeLeft2Forward  < 3) timeLeft2Forward++;
        if (timeLeft2Backward > 0) timeLeft2Backward--;
    } else {    // direction is either FORWARD or BACKWARD
        if (timeLeft2Forward  > 0) timeLeft2Forward--;
        if (timeLeft2Backward < 3) timeLeft2Backward++;
    }
}
