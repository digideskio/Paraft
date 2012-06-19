#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <QObject>
#include "Consts.h"
#include "mpi.h"

class DataManager : public QObject {
    Q_OBJECT

public:
    DataManager(QObject *parent = 0);
    ~DataManager();

    // Get the data pointer by timestep
    float* GetVolumeDataPointer(int index) { return pDataVector.at(index); }
    float* GetMaskMatrixPointer() { return pMaskMatrix; }
    float* AllocateNewDataBuffer(int x, int y, int z);

    Vector2f getMinMaxByIndex(int index) { return pLocalMinMax.at(index); }

    void SaveData(float* pt) { pDataVector.push_back(pt); }
    int getCurrentDataLength() { return pDataVector.size(); }

    void setVolumeDimension(int x, int y, int z);
    void calculateLocalMinMax();

    int getVolumeSize() { return volumeSize; }
    int GetVolumeDimX() { return dimX; }
    int GetVolumeDimY() { return dimY; }
    int GetVolumeDimZ() { return dimZ; }
    int GetFeatureVectorLength() { return pFeatureVectors.size(); }


    bool ReadDataSequence(string filePath, string prefix, string suffix,
                          int iStart, int iEnd, Vector3i dimXYZ,
                          Vector3i workerNumProcXYZ, Vector3i workerIDXYZ);

    bool ReadOneDataFile(QString strFilePath, Vector3i segLength,
                         Vector3i workerNumProcessesXYZ, Vector3i workerIDXYZ);

    // Set features
    void SaveExtractedFeatures(vector<Feature> f) { pFeatureVectors.push_back(f); }

    // Get features
    vector<Feature> *GetFeatureVector(int iTime) { return &(pFeatureVectors.at(iTime)); }
    Feature *getFeature(int iTime, int index) { return &(pFeatureVectors.at(iTime).at(index)); }

    void CreateNewMaskMatrix(int size);

private:
    Vector2f globalMinMax;
    float* pCurrentDataPointer;

    // Global normalization of the data
    void normalizeData();

    vector<float*> pDataVector;
    vector<Vector2f> pLocalMinMax;
    vector< vector<Feature> > pFeatureVectors;

    float* pAllocatedBuffer;
    float* pMaskMatrix;
    int dimX, dimY, dimZ;
    int volumeSize;
};

#endif // DATAMANAGER_H
