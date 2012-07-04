#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include "Consts.h"
#include "mpi.h"

class DataManager {

public:
    DataManager();
    ~DataManager();

    void CreateNewMaskMatrix();

    float* GetVolumeDataPointer(int index) { return pDataVector.at(index); }
    float* GetMaskMatrixPointer() { return pMaskMatrix; }

    int GetVolumeSize() { return volumeSize; }
    int GetFeatureVectorLength() { return pFeatureVectors.size(); }

    Vector3i GetVolumeDimension() { return volumeDim; }

    void ReadDataSequence(Vector3i blockCoord, Vector3i partition,
                          Vector3i origVolumeDim, DataSet ds);

    bool readOneDataFile(Vector3i blockCoord, Vector3i partition, string filePath);

    vector<Feature> *GetFeatureVector(int iTime) { return &(pFeatureVectors.at(iTime)); }
    Feature *GetFeature(int iTime, int index) { return &(pFeatureVectors.at(iTime).at(index)); }
    void SaveExtractedFeatures(vector<Feature> f) { pFeatureVectors.push_back(f); }

private:
    Vector3i volumeDim;
    int volumeSize;

    vector<float*> pDataVector;
    vector<MinMax> pMinMaxVector;
    vector< vector<Feature> > pFeatureVectors;

    float* pAllocatedBuffer;
    float* pMaskMatrix;

    void normalizeData();
    void calculateLocalMinMax();
    float* allocateNewDataBuffer(int bufferSize);
};

#endif // DATAMANAGER_H
