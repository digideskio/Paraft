#include "FeatureTracker.h"

FeatureTracker::FeatureTracker(Vector3i size) : xs(size.x), ys(size.y), zs(size.z) {
    maskValue = 0.0f;
    tfResolution = 0;
    pTFOpacityMap = NULL;
    volumeSize = size.Product();

    pMaskVolumeCurrent = new float[volumeSize]();
    pMaskVolumePrevious = new float[volumeSize]();

    if (pMaskVolumeCurrent == NULL || pMaskVolumePrevious == NULL) {
        return;
    }
}

FeatureTracker::~FeatureTracker() {
    delete [] pMaskVolumeCurrent;
    delete [] pMaskVolumePrevious;

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();
    diffPoints.clear();

    if (featureSequence.size() > 0) {
        FeatureVectorSequence::iterator it;
        for (it = featureSequence.begin(); it != featureSequence.end(); it++) {
            vector<Feature> featureVector = it->second;
            for (size_t i = 0; i < featureVector.size(); i++) {
                featureVector.at(i).SurfacePoints.clear();
                featureVector.at(i).InnerPoints.clear();
            }
        }
    }
}

void FeatureTracker::Reset() {
    maskValue = 0.0f;
    numVoxelinFeature = 0;
    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
    sumCoordinateValue.x = sumCoordinateValue.y = sumCoordinateValue.z = 0;

    std::fill(pMaskVolumeCurrent, pMaskVolumeCurrent+volumeSize, 0);
    std::fill(pMaskVolumePrevious, pMaskVolumePrevious+volumeSize, 0);

    currentFeaturesHolder.clear();
    backup1FeaturesHolder.clear();
    backup2FeaturesHolder.clear();
    backup3FeaturesHolder.clear();

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();
    diffPoints.clear();
}

void FeatureTracker::FindNewFeature(DataPoint point) {
    sumCoordinateValue = point;

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();

    /////////////////////////////////
    // Only one point now, take as edge point
    numVoxelinFeature = 1;
    surfacePoints.push_back(point);
    /////////////////////////////////

    int index = GetPointIndex(point);
    if (pMaskVolumeCurrent[index] == 0) {
        maskValue += 1.0f;
        pMaskVolumeCurrent[index] = maskValue;
    } else {
        return;
    }

    /////////////////////////////////
    expandRegion(maskValue);
    /////////////////////////////////

    if (innerPoints.size() < (size_t)FEATURE_MIN_VOXEL_NUM) {
        maskValue -= 1.0f;
        return;
    }

    Feature newFeature; {
        newFeature.ID               = GetPointIndex(centroid);
        newFeature.Centroid         = centroid;
        newFeature.SurfacePoints    = surfacePoints;
        newFeature.InnerPoints      = innerPoints;
        newFeature.MaskValue        = maskValue;
        newFeature.Min              = featureMin;
        newFeature.Max              = featureMax;
    }

    currentFeaturesHolder.push_back(newFeature);
    backup1FeaturesHolder = currentFeaturesHolder;
    backup2FeaturesHolder = currentFeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;

    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
}

void FeatureTracker::TrackFeature(float* pData, int direction, int mode) {
    pVolumeData = pData;
    innerPoints.clear();

    // save current 0-1 matrix to previous, then clear current maxtrix
    std::copy(pMaskVolumeCurrent, pMaskVolumeCurrent+volumeSize, pMaskVolumePrevious);
    std::fill(pMaskVolumeCurrent, pMaskVolumeCurrent+volumeSize, 0);

    for (size_t i = 0; i < currentFeaturesHolder.size(); i++) {
        Feature f = currentFeaturesHolder[i];

        predictRegion(i, direction, mode);
        fillRegion(f.MaskValue);
        shrinkRegion(f.MaskValue);
        expandRegion(f.MaskValue);

        f.ID              = GetPointIndex(centroid);
        f.Centroid        = centroid;
        f.SurfacePoints   = surfacePoints;
        f.InnerPoints     = innerPoints;
        f.Min             = featureMin;
        f.Max             = featureMax;
        currentFeaturesHolder[i] = f;
        innerPoints.clear();
    }
    backupFeatureInfo(direction);
}

inline void FeatureTracker::predictRegion(int index, int direction, int mode) {
    int timestepsAvailable = direction == TRACKING_BACKWARD ?
                                 timestepsAvailableBackward : timestepsAvailableForward;

    delta.x = delta.y = delta.z = 0;
    Feature b1f = backup1FeaturesHolder[index];
    Feature b2f = backup2FeaturesHolder[index];
    Feature b3f = backup3FeaturesHolder[index];

    int tmp;
    switch (mode) {
        case TRACKING_MODE_DIRECT: // PREDICT_DIRECT
            break;
        case TRACKING_MODE_LINEAR: // PREDICT_LINEAR
            if (timestepsAvailable > 1) {
                if (direction == TRACKING_BACKWARD) {
                    delta.x = b2f.Centroid.x - b1f.Centroid.x;
                    delta.y = b2f.Centroid.y - b1f.Centroid.y;
                    delta.z = b2f.Centroid.z - b1f.Centroid.z;
                } else {    // Tracking forward as default
                    delta.x = b3f.Centroid.x - b2f.Centroid.x;
                    delta.y = b3f.Centroid.y - b2f.Centroid.y;
                    delta.z = b3f.Centroid.z - b2f.Centroid.z;
                }

                for (list<DataPoint>::iterator p = b3f.SurfacePoints.begin(); p != b3f.SurfacePoints.end(); p++) {
                    tmp = (*p).x + (int)floor(delta.x); (*p).x = tmp <= 0 ? 0 : (tmp < xs ? tmp : xs-1);
                    tmp = (*p).y + (int)floor(delta.y); (*p).y = tmp <= 0 ? 0 : (tmp < ys ? tmp : ys-1);
                    tmp = (*p).z + (int)floor(delta.z); (*p).z = tmp <= 0 ? 0 : (tmp < zs ? tmp : zs-1);
                }
            }
        break;
        case TRACKING_MODE_POLYNO: // PREDICT_POLY
            if (timestepsAvailable > 1) {
                if (timestepsAvailable > 2) {
                    delta.x = b3f.Centroid.x*2 - b2f.Centroid.x*3 + b1f.Centroid.x;
                    delta.y = b3f.Centroid.y*2 - b2f.Centroid.y*3 + b1f.Centroid.y;
                    delta.z = b3f.Centroid.z*2 - b2f.Centroid.z*3 + b1f.Centroid.z;
                } else {    // [1,2)
                    if (direction == TRACKING_BACKWARD) {
                        delta.x = b2f.Centroid.x - b1f.Centroid.x;
                        delta.y = b2f.Centroid.y - b1f.Centroid.y;
                        delta.z = b2f.Centroid.z - b1f.Centroid.z;
                    } else {    // Tracking forward as default
                        delta.x = b3f.Centroid.x - b2f.Centroid.x;
                        delta.y = b3f.Centroid.y - b2f.Centroid.y;
                        delta.z = b3f.Centroid.z - b2f.Centroid.z;
                    }
                }

                for (list<DataPoint>::iterator p = b3f.SurfacePoints.begin(); p != b3f.SurfacePoints.end(); p++) {
                    tmp = (*p).x + (int)floor(delta.x); (*p).x = tmp <= 0 ? 0 : (tmp < xs ? tmp : xs-1);
                    tmp = (*p).y + (int)floor(delta.y); (*p).y = tmp <= 0 ? 0 : (tmp < ys ? tmp : ys-1);
                    tmp = (*p).z + (int)floor(delta.z); (*p).z = tmp <= 0 ? 0 : (tmp < zs ? tmp : zs-1);
                }
            }
        break;
    }
}

inline void FeatureTracker::fillRegion(float maskValue) {
    sumCoordinateValue.x = 0;
    sumCoordinateValue.y = 0;
    sumCoordinateValue.z = 0;
    numVoxelinFeature = 0;
    int index = 0;

    list<DataPoint>::iterator p;

    // predicted to be on edge
    for (p = surfacePoints.begin(); p != surfacePoints.end(); p++) {
        index = GetPointIndex(*p);
        if (pMaskVolumeCurrent[index] == 0) {
            pMaskVolumeCurrent[index] = maskValue;
            updateDiffPointList(index, maskValue);
        }
        sumCoordinateValue += (*p);
        innerPoints.push_back(*p);
        numVoxelinFeature++;
    }

    // currently not on edge but previously on edge
    for (p = surfacePoints.begin(); p != surfacePoints.end(); p++) {
        index = GetPointIndex(*p);
        (*p).x++;
        while ((*p).x >= 0 && (*p).x <= xs && (*p).x - delta.x >= 0 && (*p).x - delta.x <= xs &&
               (*p).y >= 0 && (*p).y <= ys && (*p).y - delta.y >= 0 && (*p).y - delta.y <= ys &&
               (*p).z >= 0 && (*p).z <= zs && (*p).z - delta.z >= 0 && (*p).z - delta.z <= zs &&
               pMaskVolumeCurrent[index] == 0 &&
               pMaskVolumePrevious[(int)floor(xs*ys*((*p).z-delta.z)+xs*((*p).y-delta.y)+((*p).x-delta.x))] == maskValue) {

            // Mark all points: 1. currently = 1; 2. currently = 0 but previously = 1;
            pMaskVolumeCurrent[index] = maskValue;
            updateDiffPointList(index, maskValue);

            sumCoordinateValue += (*p);
            innerPoints.push_back(*p);
            numVoxelinFeature++;
            (*p).x++;
        }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid = sumCoordinateValue / numVoxelinFeature;
}

inline void FeatureTracker::shrinkRegion(float maskValue) {
    DataPoint point;
    int index;
    bool isPointOnEdge;

    // mark all edge points as 0
    while (surfacePoints.empty() == false) {
        point = surfacePoints.front();
        shrinkEdge(point, maskValue);
        surfacePoints.pop_front();
    }

    while (dataPointList.empty() == false) {
        point = dataPointList.front();
        dataPointList.pop_front();

        index = GetPointIndex(point);
        if (getOpacity(pVolumeData[index]) < lowerThreshold || getOpacity(pVolumeData[index]) > upperThreshold) {
            isPointOnEdge = false;
            /////////////////////////////////////////////////////////////////////
            // if point is invisible, mark its adjacent points as 0            //
            shrinkEdge(point, maskValue);                                      // center
            if (++point.x < xs) { shrinkEdge(point, maskValue); } point.x--;   // right
            if (++point.y < ys) { shrinkEdge(point, maskValue); } point.y--;   // top
            if (++point.z < zs) { shrinkEdge(point, maskValue); } point.z--;   // back
            if (--point.x >  0) { shrinkEdge(point, maskValue); } point.x++;   // left
            if (--point.y >  0) { shrinkEdge(point, maskValue); } point.y++;   // bottom
            if (--point.z >  0) { shrinkEdge(point, maskValue); } point.z++;   // front
            /////////////////////////////////////////////////////////////////////
        } else if (pMaskVolumeCurrent[index] == 0) { isPointOnEdge = true; }

        if (isPointOnEdge == true) { surfacePoints.push_back(point); }
    }

    for (list<DataPoint>::iterator p = surfacePoints.begin(); p != surfacePoints.end(); ++p) {
        index = GetPointIndex((*p));
        if (pMaskVolumeCurrent[(xs)*(ys)*point.z+(xs)*point.y+point.x] != maskValue) {
            sumCoordinateValue += (*p);
            innerPoints.push_back(*p);
            numVoxelinFeature++;
            pMaskVolumeCurrent[(xs)*(ys)*point.z+(xs)*point.y+point.x] = maskValue;
            updateDiffPointList(index, maskValue);
        }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid = sumCoordinateValue / numVoxelinFeature;
}

inline void FeatureTracker::shrinkEdge(DataPoint point, float maskValue) {
    int index = GetPointIndex(point);
    if (pMaskVolumeCurrent[index] == maskValue) {
        pMaskVolumeCurrent[index] = 0;    // shrink
        sumCoordinateValue -= point;
        numVoxelinFeature--;
        for (list<DataPoint>::iterator p = innerPoints.begin(); p != innerPoints.end(); p++) {
            if (point.x == (*p).x && point.y == (*p).y && point.z == (*p).z) {
                innerPoints.erase(p); break;
            }
        }
        dataPointList.push_back(point);
    }
}

// Grow edge where possible for all the features in the CurrentFeaturesHolder
// Say if we have several features in one time step, the number we call GrowRegion is the number of features
// Each time before we call this function, we should copy the edges points of one feature we want to grow in edge
inline void FeatureTracker::expandRegion(float maskValue) {
    // put edge points to feature body
    while (surfacePoints.empty() == false) {
        dataPointList.push_back(surfacePoints.front());
        surfacePoints.pop_front();
    } // edgePointList should be empty

    while (dataPointList.empty() == false) {
        DataPoint point = dataPointList.front();
        dataPointList.pop_front();
        bool onBoundary = false;
        if (++point.x < xs) { onBoundary |= expandEdge(point, maskValue); } point.x--;  // right
        if (++point.y < ys) { onBoundary |= expandEdge(point, maskValue); } point.y--;  // top
        if (++point.z < zs) { onBoundary |= expandEdge(point, maskValue); } point.z--;  // front
        if (--point.x >  0) { onBoundary |= expandEdge(point, maskValue); } point.x++;  // left
        if (--point.y >  0) { onBoundary |= expandEdge(point, maskValue); } point.y++;  // bottom
        if (--point.z >  0) { onBoundary |= expandEdge(point, maskValue); } point.z++;  // back
        if (onBoundary) { surfacePoints.push_back(point); }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid = sumCoordinateValue / numVoxelinFeature;
}

inline bool FeatureTracker::expandEdge(DataPoint point, float maskValue) {
    int index = GetPointIndex(point);
    if (pMaskVolumeCurrent[index] != 0) {
        return false;   // not on edge, inside feature, no need to adjust
    }

    if (getOpacity(pVolumeData[index]) >= lowerThreshold && getOpacity(pVolumeData[index]) <= upperThreshold) {
        pMaskVolumeCurrent[index] = maskValue;
        updateDiffPointList(index, maskValue);
        dataPointList.push_back(point);
        innerPoints.push_back(point);
        sumCoordinateValue += point;
        numVoxelinFeature++;
        return false;
    } else {
        return true;
    }
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

inline float FeatureTracker::getOpacity(float value) {
    if (pTFOpacityMap == NULL || tfResolution <= 0) {
        cout << "Set TF pointer first." << endl;
        exit(3);
    }
    int factor = (int)((float)(tfResolution-1) * value);
    return pTFOpacityMap[factor];
 }

void FeatureTracker::updateDiffPointList(int index, float value) {
    IndexValueMap::iterator it = diffPoints.find(index);
    if (it == diffPoints.end()) {   // !contain
        diffPoints[index] = value;
    } else if (diffPoints[index] != value) {
        diffPoints[index] = value;
    } else {
        diffPoints.erase(it);
    }
 }

void FeatureTracker::backupFeatureInfo(int direction) {
    if (direction != TRACKING_FORWARD && direction != TRACKING_BACKWARD) {
        cout << "Direction is neither FORWARD or BACKWARD?" << endl;
        return;
    }
    backup1FeaturesHolder = backup2FeaturesHolder;
    backup2FeaturesHolder = backup3FeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;

    if (direction == TRACKING_FORWARD) {
        if (timestepsAvailableForward  < 3) timestepsAvailableForward++;
        if (timestepsAvailableBackward > 0) timestepsAvailableBackward--;
    } else {    // direction is either FORWARD or BACKWARD
        if (timestepsAvailableForward  > 0) timestepsAvailableForward--;
        if (timestepsAvailableBackward < 3) timestepsAvailableBackward++;
    }
}

void FeatureTracker::SaveExtractedFeatures(int index) {
    featureSequence[index] = currentFeaturesHolder;
}
