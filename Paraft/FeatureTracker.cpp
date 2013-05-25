#include "FeatureTracker.h"

FeatureTracker::FeatureTracker(Vector3i dim) : blockDim(dim) {
    maskValue = 0.0f;
    tfRes = 0;
    pTfMap = NULL;
    volumeSize = blockDim.Product();
    threshold = OPACITY_THRESHOLD;

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
                f.EdgeVoxels.clear();
                f.BodyVoxels.clear();
            }
        }
    }
}

void FeatureTracker::Reset() {
    maskValue = 0.0f;
    numVoxelinFeature = 0;
    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
    sumValue = Vector3i();

    fill(pMaskCurrent, pMaskCurrent+volumeSize, 0);
    fill(pMaskPrevious, pMaskPrevious+volumeSize, 0);

    currentFeaturesHolder.clear();
    backup1FeaturesHolder.clear();
    backup2FeaturesHolder.clear();
    backup3FeaturesHolder.clear();
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
                if (pTfMap[tfIndex] >= threshold) {
                    FindNewFeature(Vector3i(x,y,z));
                }
            }
        }
    }
}

void FeatureTracker::FindNewFeature(Vector3i seed) {
    list<Vector3i> edgeVoxels;
    list<Vector3i> bodyVoxels;

    // Only one point now, use as surface point
    numVoxelinFeature = 1;
    edgeVoxels.push_back(seed);

    maskValue += 1.0f;
    pMaskCurrent[GetVoxelIndex(seed)] = maskValue;

    expandRegion(edgeVoxels, bodyVoxels);

    if (bodyVoxels.size() < (size_t)MIN_NUM_VOXEL_IN_FEATURE) {
        maskValue -= 1.0f; return;
    }

    Feature newFeature; {
        newFeature.ID         = GetVoxelIndex(centroid);
        newFeature.Centroid   = centroid;
        newFeature.EdgeVoxels = edgeVoxels;
        newFeature.BodyVoxels = bodyVoxels;
        newFeature.MaskValue  = maskValue;
    }

    currentFeaturesHolder.push_back(newFeature);
    backup1FeaturesHolder = currentFeaturesHolder;
    backup2FeaturesHolder = currentFeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;

    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
}

void FeatureTracker::TrackFeature(float* pData, int direction, int mode) {
    if (pTfMap == NULL || tfRes <= 0) {
        cout << "Set TF pointer first." << endl; exit(3);
    }

    pVolumeData = pData;

    // save current 0-1 matrix to previous, then clear current maxtrix
    std::copy(pMaskCurrent, pMaskCurrent+volumeSize, pMaskPrevious);
    std::fill(pMaskCurrent, pMaskCurrent+volumeSize, 0);

    for (size_t i = 0; i < currentFeaturesHolder.size(); i++) {
        Feature f = currentFeaturesHolder[i];

        maskValue = f.MaskValue;

        predictRegion(i, direction, mode);
        fillRegion(f.EdgeVoxels, f.BodyVoxels);
        shrinkRegion(f.EdgeVoxels, f.BodyVoxels);
        expandRegion(f.EdgeVoxels, f.BodyVoxels);

        f.ID       = GetVoxelIndex(centroid);
        f.Centroid = centroid;

        currentFeaturesHolder[i] = f;
    }

    backupFeatureInfo(direction);
    ExtractAllFeatures();
}

inline void FeatureTracker::predictRegion(int index, int direction, int mode) {
    int timestepsAvailable = direction == FT_BACKWARD ? timestepsAvailableBackward : timestepsAvailableForward;

    delta = Vector3i();
    Feature b1f = backup1FeaturesHolder[index];
    Feature b2f = backup2FeaturesHolder[index];
    Feature b3f = backup3FeaturesHolder[index];

    int tmp;
    switch (mode) {
        case FT_DIRECT: // PREDICT_DIRECT
            break;
        case FT_LINEAR: // PREDICT_LINEAR
            if (timestepsAvailable > 1) {
                if (direction == FT_BACKWARD) {
                    delta = b2f.Centroid - b1f.Centroid;
                } else {    // Tracking forward as default
                    delta = b3f.Centroid - b2f.Centroid;
                }
                for (list<Vector3i>::iterator p = b3f.EdgeVoxels.begin(); p != b3f.EdgeVoxels.end(); p++) {
                    tmp = (*p).x + (int)floor(delta.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim.x ? tmp : blockDim.x-1);
                    tmp = (*p).y + (int)floor(delta.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim.y ? tmp : blockDim.y-1);
                    tmp = (*p).z + (int)floor(delta.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim.z ? tmp : blockDim.z-1);
                }

                for (list<Vector3i>::iterator p = b3f.EdgeVoxels.begin(); p != b3f.EdgeVoxels.end(); p++) {
                    tmp = (*p).x + (int)floor(delta.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim.x ? tmp : blockDim.x-1);
                    tmp = (*p).y + (int)floor(delta.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim.y ? tmp : blockDim.y-1);
                    tmp = (*p).z + (int)floor(delta.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim.z ? tmp : blockDim.z-1);
                }
            }
        break;
        case FT_POLYNO: // PREDICT_POLY
            if (timestepsAvailable > 1) {
                if (timestepsAvailable > 2) {
                    delta = b3f.Centroid*2 - b2f.Centroid*3 + b1f.Centroid;
                } else {    // [1,2)
                    if (direction == FT_BACKWARD) {
                        delta = b2f.Centroid - b1f.Centroid;
                    } else {    // Tracking forward as default
                        delta = b3f.Centroid - b2f.Centroid;
                    }
                }

                for (list<Vector3i>::iterator p = b3f.EdgeVoxels.begin(); p != b3f.EdgeVoxels.end(); p++) {
                    tmp = (*p).x + (int)floor(delta.x); (*p).x = tmp <= 0 ? 0 : (tmp < blockDim.x ? tmp : blockDim.x-1);
                    tmp = (*p).y + (int)floor(delta.y); (*p).y = tmp <= 0 ? 0 : (tmp < blockDim.y ? tmp : blockDim.y-1);
                    tmp = (*p).z + (int)floor(delta.z); (*p).z = tmp <= 0 ? 0 : (tmp < blockDim.z ? tmp : blockDim.z-1);
                }
            }
        break;
    }
}

inline void FeatureTracker::fillRegion(list<Vector3i> &edgeVoxels, list<Vector3i> &bodyVoxels) {
    sumValue = Vector3i();
    numVoxelinFeature = 0;

    // predicted to be on edge
    for (list<Vector3i>::iterator p = edgeVoxels.begin(); p != edgeVoxels.end(); p++) {
        int index = GetVoxelIndex(*p);
        if (pMaskCurrent[index] == 0) {
            pMaskCurrent[index] = maskValue;
        }
        bodyVoxels.push_back(*p);
        sumValue += (*p);
        numVoxelinFeature++;
    }

    // currently not on edge but previously on edge
    for (list<Vector3i>::iterator p = edgeVoxels.begin(); p != edgeVoxels.end(); p++) {
        int index = GetVoxelIndex(*p);
        int indexPrev = GetVoxelIndex((*p)-delta);
        while ((*p).x >= 0 && (*p).x <= blockDim.x && (*p).x - delta.x >= 0 && (*p).x - delta.x <= blockDim.x &&
               (*p).y >= 0 && (*p).y <= blockDim.y && (*p).y - delta.y >= 0 && (*p).y - delta.y <= blockDim.y &&
               (*p).z >= 0 && (*p).z <= blockDim.z && (*p).z - delta.z >= 0 && (*p).z - delta.z <= blockDim.z &&
               pMaskCurrent[index] == 0 && pMaskPrevious[indexPrev] == maskValue) {

            // Mark all points: 1. currently = 1; 2. currently = 0 but previously = 1;
            pMaskCurrent[index] = maskValue;
            bodyVoxels.push_back(*p);
            sumValue += (*p);
            numVoxelinFeature++;
        }
    }

    if (numVoxelinFeature > 0) {
        centroid = sumValue / numVoxelinFeature;
    }
}

inline void FeatureTracker::shrinkRegion(list<Vector3i> &edgeVoxels, list<Vector3i> &bodyVoxels) {
    // mark all edge points as 0
    while (edgeVoxels.empty() == false) {
        Vector3i voxel = edgeVoxels.front();
        edgeVoxels.pop_front();
        shrinkEdge(edgeVoxels, bodyVoxels, voxel);
    }

    while (!bodyVoxels.empty()) {
        Vector3i voxel = bodyVoxels.front();
        bodyVoxels.pop_front();

        int index = GetVoxelIndex(voxel);
        bool isPointOnEdge = false;
        if (getOpacity(pVolumeData[index]) < threshold) {
            isPointOnEdge = false;
            // if point is invisible, mark its adjacent points as 0
            shrinkEdge(edgeVoxels, bodyVoxels, voxel);                                              // center
            if (++voxel.x < blockDim.x) { shrinkEdge(edgeVoxels, bodyVoxels, voxel); } voxel.x--;   // right
            if (++voxel.y < blockDim.y) { shrinkEdge(edgeVoxels, bodyVoxels, voxel); } voxel.y--;   // top
            if (++voxel.z < blockDim.z) { shrinkEdge(edgeVoxels, bodyVoxels, voxel); } voxel.z--;   // back
            if (--voxel.x >= 0)         { shrinkEdge(edgeVoxels, bodyVoxels, voxel); } voxel.x++;   // left
            if (--voxel.y >= 0)         { shrinkEdge(edgeVoxels, bodyVoxels, voxel); } voxel.y++;   // bottom
            if (--voxel.z >= 0)         { shrinkEdge(edgeVoxels, bodyVoxels, voxel); } voxel.z++;   // front
        } else if (pMaskCurrent[index] == 0) {
            isPointOnEdge = true;
        }

        if (isPointOnEdge) {
            edgeVoxels.push_back(voxel);
        }
    }

    for (list<Vector3i>::iterator p = edgeVoxels.begin(); p != edgeVoxels.end(); p++) {
        int index = GetVoxelIndex(*p);
        if (pMaskCurrent[index] != maskValue) {
            pMaskCurrent[index] = maskValue;
            bodyVoxels.push_back(*p);
            sumValue += (*p);
            numVoxelinFeature++;
        }
    }

    if (numVoxelinFeature > 0) {
        centroid = sumValue / numVoxelinFeature;
    }
}

inline void FeatureTracker::shrinkEdge(list<Vector3i> &edgeVoxels, list<Vector3i> &bodyVoxels, const Vector3i &voxel) {
    int index = GetVoxelIndex(voxel);
    if (pMaskCurrent[index] == maskValue) {
        pMaskCurrent[index] = 0;  // shrink
        list<Vector3i>::iterator p = find(bodyVoxels.begin(), bodyVoxels.end(), voxel);
        bodyVoxels.erase(p);
        edgeVoxels.push_back(voxel);
        sumValue -= voxel;
        numVoxelinFeature--;
    }
}

inline void FeatureTracker::expandRegion(list<Vector3i> &edgeVoxels, list<Vector3i> &bodyVoxels) {
    list<Vector3i> tempVoxels;    // to store updated edge voxels
    while (!edgeVoxels.empty()) {
        Vector3i voxel = edgeVoxels.front();
        edgeVoxels.pop_front();
        bool voxelOnEdge = false;
        if (++voxel.x < blockDim.x) { voxelOnEdge |= expandEdge(edgeVoxels, bodyVoxels, voxel); } voxel.x--;  // right
        if (++voxel.y < blockDim.y) { voxelOnEdge |= expandEdge(edgeVoxels, bodyVoxels, voxel); } voxel.y--;  // top
        if (++voxel.z < blockDim.z) { voxelOnEdge |= expandEdge(edgeVoxels, bodyVoxels, voxel); } voxel.z--;  // front
        if (--voxel.x >= 0)         { voxelOnEdge |= expandEdge(edgeVoxels, bodyVoxels, voxel); } voxel.x++;  // left
        if (--voxel.y >= 0)         { voxelOnEdge |= expandEdge(edgeVoxels, bodyVoxels, voxel); } voxel.y++;  // bottom
        if (--voxel.z >= 0)         { voxelOnEdge |= expandEdge(edgeVoxels, bodyVoxels, voxel); } voxel.z++;  // back
        if (voxelOnEdge)            { tempVoxels.push_back(voxel); }
    }
    edgeVoxels.swap(tempVoxels);

    if (numVoxelinFeature > 0) {
        centroid = sumValue / numVoxelinFeature;
    }
}

inline bool FeatureTracker::expandEdge(list<Vector3i> &edgeVoxels, list<Vector3i> &bodyVoxels, const Vector3i &voxel) {
    int index = GetVoxelIndex(voxel);
    if (pMaskCurrent[index] > 0 || getOpacity(pVolumeData[index]) < threshold) {
        return false;
    }  // already labeled by a feature or opacity not large enough to be labeled

    pMaskCurrent[index] = maskValue;
    edgeVoxels.push_back(voxel);
    bodyVoxels.push_back(voxel);
    sumValue += voxel;
    numVoxelinFeature++;
    return true;
}

void FeatureTracker::SetCurrentFeatureInfo(vector<Feature> *pFeatures) {
    currentFeaturesHolder.clear();
    for (size_t i = 0; i < pFeatures->size(); i++) {
        currentFeaturesHolder.push_back(pFeatures->at(i));
    }

    backup1FeaturesHolder = currentFeaturesHolder;
    backup2FeaturesHolder = currentFeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;
    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
}

void FeatureTracker::backupFeatureInfo(int direction) {
    backup1FeaturesHolder = backup2FeaturesHolder;
    backup2FeaturesHolder = backup3FeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;

    if (direction == FT_FORWARD) {
        if (timestepsAvailableForward  < 3) timestepsAvailableForward++;
        if (timestepsAvailableBackward > 0) timestepsAvailableBackward--;
    } else {    // direction is either FORWARD or BACKWARD
        if (timestepsAvailableForward  > 0) timestepsAvailableForward--;
        if (timestepsAvailableBackward < 3) timestepsAvailableBackward++;
    }
}
