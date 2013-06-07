#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include "Utils.h"
#include "Metadata.h"

class DataManager {

public:
    DataManager();
   ~DataManager();

    float* GetDataPointer(int index) { return dataSequence[index]; }
    float* GetTFOpacityMap()         { return pTFOpacityMap; }
    int GetTFResolution()            { return tfResolution; }
    Vector3i GetBlockDimension()     { return blockDim; }

    void CreateNewMaskVolume();
    void InitTFSettings(const string &filename);
    void LoadDataSequence(const Metadata &meta, const int timestep);
    void SaveMaskVolume(float *pData, const Metadata &meta, const int timestep);
private:
    void preprocessData(float *pData, bool remapping);
    void normalize(float *pData);

    DataSequence dataSequence;
    Vector3i blockDim;

    int volumeSize;
    int tfResolution;

    float *pMaskVolume;
    float *pTFOpacityMap;
};

#endif // DATAMANAGER_H
