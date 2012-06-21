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
    int GetVolumeDimX() { return volumeDim.x; }
    int GetVolumeDimY() { return volumeDim.y; }
    int GetVolumeDimZ() { return volumeDim.z; }
    int GetFeatureVectorLength() { return pFeatureVectors.size(); }

    Vector3i GetVolumeDimension() { return volumeDim; }

    void ReadDataSequence(DataSet ds, Vector3i origVolumeDim,
                          Vector3i workerNumProcXYZ, Vector3i workerIDXYZ);

    bool ReadOneDataFile(string filePath, Vector3i volumeDim,
                         Vector3i workerNumProcXYZ, Vector3i workerIDXYZ);

    vector<Feature> *GetFeatureVector(int iTime) { return &(pFeatureVectors.at(iTime)); }
    Feature *GetFeature(int iTime, int index) { return &(pFeatureVectors.at(iTime).at(index)); }
    void SaveExtractedFeatures(vector<Feature> f) { pFeatureVectors.push_back(f); }

private:
    Vector3i volumeDim;
    int volumeSize;

    Vector2f globalMinMax;      // not used yet

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
