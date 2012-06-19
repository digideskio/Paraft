#ifndef DATAHOLDER_H
#define DATAHOLDER_H

#include <QObject>
#include <QPointF>
#include "Consts.h"

class DataHolder : public QObject {
    Q_OBJECT

public:
    DataHolder(QObject *parent = 0);
    ~DataHolder();

    float* AllocNewDataBuffer(int xDim, int yDim, int zDim);
    float* GetDataPointerByIndex(int index) { return pDataVector.at(index); }
    float* GetMaskMatrixPointer() { return pMaskMatrix; }

    QPointF* getMinMaxByIndex(int index) { return &(pLocalMinMax.at(index)); }

    void SaveData(float* pt) { pDataVector.push_back(pt);}
    int GetCurrentDataLength() { return pDataVector.size(); }

    void SetVolumeDimension(int x, int y, int z);

    int GetVolumeSize() { return volumeSize; }
    int GetVolumeDimX() { return dimX; }
    int GetVolumeDimY() { return dimY; }
    int GetVolumeDimZ() { return dimZ; }

    void CalculateLocalMinMax();

    // Set features
    void StoreFeatures(vector<Feature> f) { pFeatureVectors.push_back(f);}

    // Get features
    vector<Feature> *GetFeatureVector(int itime) { return &(pFeatureVectors.at(itime)); }
    Feature *getFeature(uint itime, uint idx) { return &(pFeatureVectors.at(itime).at(idx)); }
    int GetFeatureLength() { return pFeatureVectors.size(); }

    void CreateNewMaskMatrix(int size);

private:
    vector<float*> pDataVector;
    vector<QPointF> pLocalMinMax;
    vector< vector<Feature> > pFeatureVectors;

    float* pAllocatedBuffer;
    float* pMaskMatrix;
    int dimX, dimY, dimZ;
    int volumeSize;
};

#endif // DATAHOLDER_H
