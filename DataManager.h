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


    bool ReadDataSequence(string filePath, string prefix, string suffix,
                          int iStart, int iEnd, Vector3i dimXYZ,
                          Vector3i workerNumProcXYZ, Vector3i workerIDXYZ);

    bool ReadOneDataFile(string filePath, Vector3i segLength,
                         Vector3i workerNumProcessesXYZ, Vector3i workerIDXYZ);

    vector<Feature> *GetFeatureVector(int iTime) { return &(pFeatureVectors.at(iTime)); }
    Feature *getFeature(int iTime, int index) { return &(pFeatureVectors.at(iTime).at(index)); }
    void SaveExtractedFeatures(vector<Feature> f) { pFeatureVectors.push_back(f); }

private:
    Vector3i volumeDim;
    int volumeSize;

    // not used yet
    Vector2f globalMinMax;

    vector<float*> pDataVector;
    vector<MinMax> pMinMaxVector;
    vector< vector<Feature> > pFeatureVectors;

    float* pAllocatedBuffer;
    float* pMaskMatrix;

    void normalizeData();
    void calculateLocalMinMax();
    float* allocateNewDataBuffer(Vector3i dim);
};

#endif // DATAMANAGER_H
