#include "DataHolder.h"

DataHolder::DataHolder(QObject *parent) : QObject(parent) {
    pAllocatedBuffer = NULL;
    pMaskMatrix = NULL;
    pDataVector.clear();
    pLocalMinMax.clear();
    pFeatureVectors.clear();
}

DataHolder::~DataHolder() {
    if (pDataVector.size() != 0) {
        qDebug("Clean data vector");
        for (uint i = 0; i < pDataVector.size(); i++) {
            delete [] pDataVector.at(i);
        }
    }
    pLocalMinMax.clear();

    if (pFeatureVectors.size() != 0) {
        for (uint i = 0; i < pFeatureVectors.size(); i++) {
            for (uint j = 0; j < pFeatureVectors.at(i).size(); j++) {
                pFeatureVectors.at(i).at(j).SurfacePoints.clear();
                pFeatureVectors.at(i).at(j).InnerPoints.clear();
                pFeatureVectors.at(i).at(j).Uncertainty.clear();
            }
        }
    }

    if (pMaskMatrix != NULL) {
        delete [] pMaskMatrix;
    }
}

float* DataHolder::AllocNewDataBuffer(int xDim, int yDim, int zDim) {
    // Allocate new space
    pAllocatedBuffer = new float[xDim * yDim * zDim];
    if (pAllocatedBuffer == NULL) {
        qDebug("DataHolder: Allocate memory failed!");
        return pAllocatedBuffer;
    }

    // Push into the data list
    pDataVector.push_back(pAllocatedBuffer);

    // Return the pointer
    return pAllocatedBuffer;
}

void DataHolder::SetVolumeDimension(int x, int y, int z) {
    dimX = x; dimY = y; dimZ = z;
    volumeSize = dimX * dimY * dimZ;
}

void DataHolder::CreateNewMaskMatrix(int size) {
    pMaskMatrix = new float[size];
    memset(pMaskMatrix, 0, sizeof(float)*size);
}

void DataHolder::CalculateLocalMinMax() {
    float min = pAllocatedBuffer[0];
    float max = pAllocatedBuffer[0];

    for (int i = 1; i < volumeSize; i++) {
        min = min < pAllocatedBuffer[i] ? min : pAllocatedBuffer[i];
        max = max > pAllocatedBuffer[i] ? max : pAllocatedBuffer[i];
    }

    QPointF minMax; {
        minMax.setX(min);
        minMax.setY(max);
    }

    pLocalMinMax.push_back(minMax);
}

