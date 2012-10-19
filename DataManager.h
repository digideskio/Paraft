#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <mpi.h>
#include "Consts.h"

class DataManager {

public:
    DataManager();
    ~DataManager();

    void CreateNewMaskMatrix();

    float* GetVolumeDataPointer(int index) { return dataSequenceMap[index]; }
    float* GetMaskMatrixPointer() { return pMaskMatrix; }

    int GetVolumeSize() { return volumeSize; }

    // TF
    int GetTFResolution() { return tfResolution; }
    float* GetTFOpacityMap() { return pTFOpacityMap; }

    Vector3i GetVolumeDimension() { return volumeDim; }

    void MpiReadDataSequence(Vector3i blockCoord, Vector3i partition, DataSet ds);
    void InitTFSettings(string filename);

private:
    Vector3i volumeDim;
    int volumeSize;

//    DataVector dataSequence;
    DataSequenceMap dataSequenceMap;
    MinMaxVector minMaxSequence;
//    vector< vector<Feature> > featureVectors;

    float* pDataBuffer;
    float* pMaskMatrix;

    // TF
    int tfResolution;
    float *pTFOpacityMap;

    void normalizeData(DataSet ds);
    void calculateLocalMinMax();
};

#endif // DATAMANAGER_H
