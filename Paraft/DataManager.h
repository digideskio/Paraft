#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <mpi.h>
#include "Consts.h"

class DataManager {

public:
    DataManager();
    ~DataManager();

    float* GetDataPointer(int index) { return dataSequence[index]; }
    float* GetTFOpacityMap() { return pTFOpacityMap; }
    int GetTFResolution() { return tfResolution; }

    Vector3 GetVolumeDimension() { return blockDim; }

    void CreateNewMaskVolume();
    void InitTFSettings(string filename);
    void PreloadDataSequence(Vector3 gridDim, Vector3 blockIdx, Metadata metadata, int timestep);

private:
    DataSequence dataSequence;
    Vector3 blockDim;

    int volumeSize;
    int tfResolution;

    float *pMaskVolume;
    float *pTFOpacityMap;

    void normalizeData(float *pData, Metadata meta);
};

#endif // DATAMANAGER_H
