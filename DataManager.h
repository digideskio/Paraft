#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <QObject>
#include "DataHolder.h"
#include "mpi.h"

class DataManager : public QObject {
    Q_OBJECT

public:
    DataManager(QObject *parent=0);
    ~DataManager();

    // Get the data pointer by index
    float* GetVolumeDataPointer(int index) { return pDataHolder->GetDataPointerByIndex(index); }

    bool ReadDataSequence(string filePath, string prefix, string suffix,
                          int iStart, int iEnd, Vector3d dimXYZ,
                          Vector3d workerNumProcXYZ, Vector3d workerIDXYZ);

    bool ReadOneDataFile(QString strFilePath, Vector3d segLength,
                         Vector3d workerNumProcessesXYZ, Vector3d workerIDXYZ);

    int GetVolumeDimX() { return pDataHolder->GetVolumeDimX(); }
    int GetVolumeDimY() { return pDataHolder->GetVolumeDimY(); }
    int GetVolumeDimZ() { return pDataHolder->GetVolumeDimZ(); }
    int GetVolumeSize() { return pDataHolder->GetVolumeSize(); }

    // Set features
    void SaveExtractedFeatures(vector<Feature> f) { pDataHolder->StoreFeatures(f); }

    // Get features
    vector<Feature> *GetFeatureVector(int index) { return pDataHolder->GetFeatureVector(index); }

    int GetFeatureVectorLength() { return pDataHolder->GetFeatureLength(); }

    float* GetMaskMatrixPointer() { return pDataHolder->GetMaskMatrixPointer(); }
    void   CreateNewMaskMatrix(int size) { pDataHolder->CreateNewMaskMatrix(size); }

private:
    DataHolder *pDataHolder;
    QPointF globalMinMax;
    float* pCurrentDataPointer;

    // Global normalization of the data
    void normalizeData();
};

#endif // DATAMANAGER_H
