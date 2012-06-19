#include "FeatureTracker.h"

using namespace std;

FeatureTracker::FeatureTracker(int xsize, int ysize, int zsize, QObject *parent)
    : QObject(parent), xs(xsize), ys(ysize), zs(zsize) {
    maskValue = 0.0f;
    volumeSize = xs * ys * zs;

    resetFeatureBoundaryInfo();

    pMaskMatrixCurrent = (float*)malloc(volumeSize * sizeof(float));
    pMaskMatrixPrevious = (float*)malloc(volumeSize * sizeof(float));

    if (pMaskMatrixCurrent == NULL || pMaskMatrixPrevious == NULL) { return; }

    memset(pMaskMatrixCurrent, 0, volumeSize * sizeof(float));
    memset(pMaskMatrixPrevious, 0, volumeSize * sizeof(float));

    pTFColorMap = NULL;
    tfResolution = 0;
}

FeatureTracker::~FeatureTracker() {
    free(pMaskMatrixCurrent);
    free(pMaskMatrixPrevious);

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();
    differentPoints.clear();
}

void FeatureTracker::Reset() {
    maskValue = 0.0f;
    numVoxelinFeature = 0;

    resetFeatureBoundaryInfo();

    memset(pMaskMatrixCurrent, 0, xs*ys*zs * sizeof(float));   // clear current data
    memset(pMaskMatrixPrevious, 0, xs*ys*zs * sizeof(float));  // clear previous data

    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
    sumCoordinateValue.x = 0;
    sumCoordinateValue.y = 0;
    sumCoordinateValue.z = 0;

    currentFeaturesHolder.clear();
    backup1FeaturesHolder.clear();
    backup2FeaturesHolder.clear();
    backup3FeaturesHolder.clear();

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();
    differentPoints.clear();
}

void FeatureTracker::resetFeatureBoundaryInfo() {
    featureMin.x = xs + 1;
    featureMin.y = ys + 1;
    featureMin.z = zs + 1;
    featureMax.x = -1;
    featureMax.y = -1;
    featureMax.z = -1;

    for (int i = 0; i < 6; i++) {
        boundaryMin[i].x = xs + 1;
        boundaryMin[i].y = ys + 1;
        boundaryMin[i].z = zs + 1;
        boundaryMax[i].x = -1;
        boundaryMax[i].y = -1;
        boundaryMax[i].z = -1;

        // weighted ghost center
        sumBoundaryXYZValue[i].x = 0;
        sumBoundaryXYZValue[i].y = 0;
        sumBoundaryXYZValue[i].z = 0;
        numVoxelonBoundary[i] = 0;
    }   // 6 surfaces for a single block

    touchedSurfaces.clear();
}

void FeatureTracker::FindNewFeature(int x, int y, int z, float lowerValue, float upperValue) {
    DataPoint point; { point.x = x; point.y = y; point.z = z; }
    upperThreshold = upperValue;
    lowerThreshold = lowerValue;

    sumCoordinateValue.x = x;
    sumCoordinateValue.y = y;
    sumCoordinateValue.z = z;

    dataPointList.clear();
    surfacePoints.clear();
    innerPoints.clear();

    resetFeatureBoundaryInfo();

    /////////////////////////////////
    // Only one point now, take as edge point
    numVoxelinFeature = 1;
    surfacePoints.push_back(point);
    /////////////////////////////////

    int index = GetPointIndex(point);
    if (pMaskMatrixCurrent[index] == 0) {
        maskValue += 1.0f;
        pMaskMatrixCurrent[index] = maskValue;
    } else {
        qDebug("Current maskValue = %f\n", maskValue);
        return;
    }

    /////////////////////////////////
    expandRegion(maskValue);
    /////////////////////////////////

    if (innerPoints.size() < FEATURE_MIN_VOXEL_NUM) {
        maskValue -= 1.0f;
        return;
    }

    updateTouchedSurfaces();

    Feature newFeature; {
        newFeature.ID               = GetPointIndex(centroid);
        newFeature.Centroid         = centroid;
        newFeature.SurfacePoints    = surfacePoints;
        newFeature.InnerPoints      = innerPoints;
        newFeature.MaskValue        = maskValue;
        newFeature.Min              = featureMin;
        newFeature.Max              = featureMax;
        newFeature.TouchedSurfaces  = touchedSurfaces;
        for (int i = 0; i < 6; i++) {
            newFeature.BoundaryCentroid[i]  = boundaryCentroid[i];
        }
    }

    currentFeaturesHolder.push_back(newFeature);
    backup1FeaturesHolder = currentFeaturesHolder;
    backup2FeaturesHolder = currentFeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;

    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
}

void FeatureTracker::TrackFeature(float* pDataSet, float lowerValue,
                                  float upperValue, int direction, int mode) {
    pVolumeData = pDataSet;
    upperThreshold = upperValue;
    lowerThreshold = lowerValue;
    innerPoints.clear();

    // save current 0-1 matrix to previous, then clear current maxtrix
    memcpy(pMaskMatrixPrevious, pMaskMatrixCurrent, volumeSize*sizeof(float));
    memset(pMaskMatrixCurrent, 0, volumeSize*sizeof(float));

    Feature fi;
    for (int i = 0; i < currentFeaturesHolder.size(); i++) {
        fi = currentFeaturesHolder[i];

        resetFeatureBoundaryInfo();
        predictRegion(i, direction, mode);
        fillRegion(fi.MaskValue);
        shrinkRegion(fi.MaskValue);
        expandRegion(fi.MaskValue);
        updateTouchedSurfaces();

        fi.ID              = GetPointIndex(centroid);
        fi.Centroid        = centroid;
        fi.SurfacePoints   = surfacePoints;
        fi.InnerPoints     = innerPoints;
        fi.Min             = featureMin;
        fi.Max             = featureMax;
        fi.TouchedSurfaces = touchedSurfaces;

        for (int surface = 0; surface < 6; surface++) {
            fi.BoundaryCentroid[surface] = boundaryCentroid[surface];
        }
        currentFeaturesHolder[i] = fi;
        innerPoints.clear();
    }
    backupFeatureInfo(direction);
}

void FeatureTracker::updateTouchedSurfaces() {
    for (int i = 0; i < 6; i++) {
        if (numVoxelonBoundary[i] == 0) { continue; }
        boundaryCentroid[i].x = sumBoundaryXYZValue[i].x / numVoxelonBoundary[i];
        boundaryCentroid[i].y = sumBoundaryXYZValue[i].y / numVoxelonBoundary[i];
        boundaryCentroid[i].z = sumBoundaryXYZValue[i].z / numVoxelonBoundary[i];
        touchedSurfaces.push_back(i);
    }
}

inline void FeatureTracker::predictRegion(int index, int direction, int mode) {
    int timestepsAvailable = direction == TRACKING_DIRECTION_BACKWARD ?
                timestepsAvailableBackward : timestepsAvailableForward;

    delta.x = delta.y = delta.z = 0;
    Feature b1f = backup1FeaturesHolder[index];
    Feature b2f = backup2FeaturesHolder[index];
    Feature b3f = backup3FeaturesHolder[index];

    int temp;
    switch (mode) {
        case TRACKING_MODE_DIRECT: // PREDICT_DIRECT
            break;
        case TRACKING_MODE_LINEAR: // PREDICT_LINEAR
            if (timestepsAvailable > 1) {
                if (direction == TRACKING_DIRECTION_BACKWARD) {
                    delta.x = b2f.Centroid.x - b1f.Centroid.x;
                    delta.y = b2f.Centroid.y - b1f.Centroid.y;
                    delta.z = b2f.Centroid.z - b1f.Centroid.z;
                } else {    // Tracking forward as default
                    delta.x = b3f.Centroid.x - b2f.Centroid.x;
                    delta.y = b3f.Centroid.y - b2f.Centroid.y;
                    delta.z = b3f.Centroid.z - b2f.Centroid.z;
                }
                foreach (DataPoint point, b3f.SurfacePoints) {
                    temp = point.x + (int)floor(delta.x); point.x = temp <= 0 ? 0 : (temp < xs ? temp : xs - 1);
                    temp = point.y + (int)floor(delta.y); point.y = temp <= 0 ? 0 : (temp < ys ? temp : ys - 1);
                    temp = point.z + (int)floor(delta.z); point.z = temp <= 0 ? 0 : (temp < zs ? temp : zs - 1);
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
                    if (direction == TRACKING_DIRECTION_BACKWARD) {
                        delta.x = b2f.Centroid.x - b1f.Centroid.x;
                        delta.y = b2f.Centroid.y - b1f.Centroid.y;
                        delta.z = b2f.Centroid.z - b1f.Centroid.z;
                    } else {    // Tracking forward as default
                        delta.x = b3f.Centroid.x - b2f.Centroid.x;
                        delta.y = b3f.Centroid.y - b2f.Centroid.y;
                        delta.z = b3f.Centroid.z - b2f.Centroid.z;
                    }
                }
                foreach (DataPoint point, b3f.SurfacePoints) {
                    temp = point.x + (int)floor(delta.x); point.x = temp <= 0 ? 0 : (temp < xs ? temp : xs - 1);
                    temp = point.y + (int)floor(delta.y); point.y = temp <= 0 ? 0 : (temp < ys ? temp : ys - 1);
                    temp = point.z + (int)floor(delta.z); point.z = temp <= 0 ? 0 : (temp < zs ? temp : zs - 1);
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

    foreach (DataPoint point, surfacePoints) {  // predicted to be on edge
        index = GetPointIndex(point);
        if (pMaskMatrixCurrent[index] == 0) {
            pMaskMatrixCurrent[index] = maskValue;
            updateDiffPointList(index, maskValue);
        }
        sumCoordinateValue.x += point.x;
        sumCoordinateValue.y += point.y;
        sumCoordinateValue.z += point.z;
        numVoxelinFeature++;
        innerPoints.push_back(point);
    }

    foreach (DataPoint point, surfacePoints) {  // currently not on edge but previously on edge
        index = GetPointIndex(point);
        point.x++;
        while (point.x >= 0 && point.x <= xs && point.x - delta.x >= 0 && point.x - delta.x <= xs &&
               point.y >= 0 && point.y <= ys && point.y - delta.y >= 0 && point.y - delta.y <= ys &&
               point.z >= 0 && point.z <= zs && point.z - delta.z >= 0 && point.z - delta.z <= zs &&
               pMaskMatrixCurrent[index] == 0 &&
               pMaskMatrixPrevious[(int)floor(xs*ys*(point.z-delta.z)+xs*(point.y-delta.y)+(point.x-delta.x))] == maskValue) {

            // Mark all points: 1. currently = 1; 2. currently = 0 but previously = 1;
            pMaskMatrixCurrent[index] = maskValue;
            updateDiffPointList(index, maskValue);

            sumCoordinateValue.x += point.x;
            sumCoordinateValue.y += point.y;
            sumCoordinateValue.z += point.z;
            innerPoints.push_back(point);
            numVoxelinFeature++;
            point.x++;
        }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid.x = sumCoordinateValue.x / numVoxelinFeature;
    centroid.y = sumCoordinateValue.y / numVoxelinFeature;
    centroid.z = sumCoordinateValue.z / numVoxelinFeature;
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
        if (getOpacity(pVolumeData[index]) < lowerThreshold ||
            getOpacity(pVolumeData[index]) > upperThreshold) { // invisible
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
        } else if (pMaskMatrixCurrent[index] == 0) { isPointOnEdge = true; }

        if (isPointOnEdge == true) { surfacePoints.push_back(point); }
    }

    foreach (DataPoint point, surfacePoints) {
        if (pMaskMatrixCurrent[(xs)*(ys)*point.z+(xs)*point.y+point.x] != maskValue) {
            sumCoordinateValue.x += point.x;
            sumCoordinateValue.y += point.y;
            sumCoordinateValue.z += point.z;
            numVoxelinFeature++;
            pMaskMatrixCurrent[(xs)*(ys)*point.z+(xs)*point.y+point.x] = maskValue;
            updateDiffPointList(index, maskValue);
            innerPoints.push_back(point);
        }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid.x = sumCoordinateValue.x / numVoxelinFeature;
    centroid.y = sumCoordinateValue.y / numVoxelinFeature;
    centroid.z = sumCoordinateValue.z / numVoxelinFeature;
}

inline void FeatureTracker::shrinkEdge(DataPoint point, float maskValue) {
    int index = GetPointIndex(point);
    if (pMaskMatrixCurrent[index] == maskValue) {
        pMaskMatrixCurrent[index] = 0;    // shrink
        sumCoordinateValue.x -= point.x;
        sumCoordinateValue.y -= point.y;
        sumCoordinateValue.z -= point.z;
        numVoxelinFeature--;
        for (list<DataPoint>::iterator it = innerPoints.begin(); it != innerPoints.end(); it++) {
            if (point.x == (*it).x && point.y == (*it).y && point.z == (*it).z) {
                innerPoints.erase(it); break;
            }
        }
        dataPointList.push_back(point);
    }
}

// Grow edge where possible for all the features in the CurrentFeaturesHolder
// Say if we have several features in one time step, the number we call GrowRegion is the number of features
// Each time before we call this function, we should copy the edges points of one feature we want to grow in edge
inline void FeatureTracker::expandRegion(float maskValue) {
    DataPoint point;
    bool onBoundary;

    // put edge points to feature body
    while (surfacePoints.empty() == false) {
        dataPointList.push_back(surfacePoints.front());
        surfacePoints.pop_front();
    } // edgePointList should be empty

    while (dataPointList.empty() == false) {
        point = dataPointList.front();
        dataPointList.pop_front();
        onBoundary = false;
        if (++point.x < xs) { onBoundary |= expandEdge(point, maskValue); } point.x--;  // right
        if (++point.y < ys) { onBoundary |= expandEdge(point, maskValue); } point.y--;  // top
        if (++point.z < zs) { onBoundary |= expandEdge(point, maskValue); } point.z--;  // front
        if (--point.x >  0) { onBoundary |= expandEdge(point, maskValue); } point.x++;  // left
        if (--point.y >  0) { onBoundary |= expandEdge(point, maskValue); } point.y++;  // bottom
        if (--point.z >  0) { onBoundary |= expandEdge(point, maskValue); } point.z++;  // back
        if (onBoundary == true) { surfacePoints.push_back(point); }
    }

    if (numVoxelinFeature == 0) { return; }
    centroid.x = sumCoordinateValue.x / numVoxelinFeature;
    centroid.y = sumCoordinateValue.y / numVoxelinFeature;
    centroid.z = sumCoordinateValue.z / numVoxelinFeature;
}

inline bool FeatureTracker::expandEdge(DataPoint point, float maskValue) {
    int index = GetPointIndex(point);
    if (pMaskMatrixCurrent[index] != 0) {
        return false;   // not on edge, inside feature, no need to adjust
    }

    if (getOpacity(pVolumeData[index]) >= lowerThreshold && getOpacity(pVolumeData[index]) <= upperThreshold) {
        pMaskMatrixCurrent[index] = maskValue;
        updateDiffPointList(index, maskValue);
        updateFeatureMinMax(point);
        ////////////// Boudary Detection //////////////
        if (point.x == 0)    { updateBoundaryMinMax(point, SURFACE_LEFT);   }  // left
        if (point.y == 0)    { updateBoundaryMinMax(point, SURFACE_BOTTOM); }  // bottom
        if (point.z == 0)    { updateBoundaryMinMax(point, SURFACE_FRONT);  }  // front
        if (point.x == xs-1) { updateBoundaryMinMax(point, SURFACE_RIGHT);  }  // right
        if (point.y == ys-1) { updateBoundaryMinMax(point, SURFACE_TOP);    }  // top
        if (point.z == zs-1) { updateBoundaryMinMax(point, SURFACE_BACK);   }  // back
        ///////////////////////////////////////////////
        dataPointList.push_back(point);
        innerPoints.push_back(point);
        sumCoordinateValue.x += point.x;
        sumCoordinateValue.y += point.y;
        sumCoordinateValue.z += point.z;
        numVoxelinFeature++;
        return false;
    } else {
        return true;
    }
}

void FeatureTracker::updateFeatureMinMax(DataPoint point) {
    featureMin.x = featureMin.x < point.x ? featureMin.x : point.x;
    featureMin.y = featureMin.y < point.y ? featureMin.y : point.y;
    featureMin.z = featureMin.z < point.z ? featureMin.z : point.z;
    featureMax.x = featureMax.x > point.x ? featureMax.x : point.x;
    featureMax.y = featureMax.y > point.y ? featureMax.y : point.y;
    featureMax.z = featureMax.z > point.z ? featureMax.z : point.z;
}

void FeatureTracker::updateBoundaryMinMax(DataPoint point, int surface) {
    boundaryMin[surface].x = boundaryMin[surface].x < point.x ? boundaryMin[surface].x : point.x;
    boundaryMin[surface].y = boundaryMin[surface].y < point.y ? boundaryMin[surface].y : point.y;
    boundaryMin[surface].z = boundaryMin[surface].z < point.z ? boundaryMin[surface].z : point.z;
    boundaryMax[surface].x = boundaryMax[surface].x > point.x ? boundaryMax[surface].x : point.x;
    boundaryMax[surface].y = boundaryMax[surface].y > point.y ? boundaryMax[surface].y : point.y;
    boundaryMax[surface].z = boundaryMax[surface].z > point.z ? boundaryMax[surface].z : point.z;

    sumBoundaryXYZValue[surface].x += point.x;
    sumBoundaryXYZValue[surface].y += point.y;
    sumBoundaryXYZValue[surface].z += point.z;
    numVoxelonBoundary[surface]++;
}

void FeatureTracker::SetCurrentFeatureInfo(vector<Feature> *pFeatures) {
    currentFeaturesHolder.clear();
    for (int i = 0; i < pFeatures->size(); i++) {
        currentFeaturesHolder.push_back(pFeatures->at(i));
    }

    backup1FeaturesHolder = currentFeaturesHolder;
    backup2FeaturesHolder = currentFeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;
    timestepsAvailableForward = 0;
    timestepsAvailableBackward = 0;
}

 float FeatureTracker::getOpacity(float DataValue) {
    if (pTFColorMap == NULL || tfResolution == 0) {
        qDebug("Set TF pointer first!");
        return -1;
    }
    int factor = (int)(tfResolution * DataValue);
    return pTFColorMap[4*factor+3];
 }

 void FeatureTracker::updateDiffPointList(int index, float value) {
    if(differentPoints.contains(index) == false) {
        differentPoints.insert(index, value);
    } else if (differentPoints[index] != value) {
        differentPoints[index] = value;
    } else {    // delete those not changed
        differentPoints.remove(index);
    }
 }

void FeatureTracker::backupFeatureInfo(int direction) {
    if (direction != TRACKING_DIRECTION_FORWARD && direction != TRACKING_DIRECTION_BACKWARD) {
        qDebug("Direction is neither FORWARD or BACKWARD? Something must be wrong here...");
        return;
    }
    backup1FeaturesHolder = backup2FeaturesHolder;
    backup2FeaturesHolder = backup3FeaturesHolder;
    backup3FeaturesHolder = currentFeaturesHolder;

    if (direction == TRACKING_DIRECTION_FORWARD) {
        if (timestepsAvailableForward  < 3) timestepsAvailableForward++;
        if (timestepsAvailableBackward > 0) timestepsAvailableBackward--;
    } else {    // direction is either FORWARD or BACKWARD
        if (timestepsAvailableForward  > 0) timestepsAvailableForward--;
        if (timestepsAvailableBackward < 3) timestepsAvailableBackward++;
    }
}