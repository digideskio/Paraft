#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <mpi.h>
#include "Consts.h"

class DataManager {

public:
    DataManager();
    ~DataManager();

    void CreateNewMaskMatrix();

    float* GetVolumeDataPointer(int index) { return dataSequence.at(index); }
    float* GetMaskMatrixPointer() { return pMaskMatrix; }

    int GetVolumeSize() { return volumeSize; }
    int GetFeatureVectorLength() { return featureVectors.size(); }

    Vector3i GetVolumeDimension() { return volumeDim; }

    void MpiReadDataSequence(Vector3i blockCoord, Vector3i partition, DataSet ds);

    vector<Feature> *GetFeatureVector(int iTime) { return &(featureVectors.at(iTime)); }
    Feature *GetFeature(int iTime, int index) { return &(featureVectors.at(iTime).at(index)); }
    void SaveExtractedFeatures(vector<Feature> f) { featureVectors.push_back(f); }

private:
    Vector3i volumeDim;
    int volumeSize;

    DataVector dataSequence;
    MinMaxVector minMaxSequence;
    vector< vector<Feature> > featureVectors;

    float* pDataBuffer;
    float* pMaskMatrix;

    void normalizeData(DataSet ds);
    void calculateLocalMinMax();
};

#endif // DATAMANAGER_H
