#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <mpi.h>
#include "Consts.h"

class DataManager {

public:
    DataManager();
    ~DataManager();

    float* GetDataPointer(int index) { return dataSequence[index]; }
    float* GetTFOpacityMap()         { return pTFOpacityMap; }
    int GetTFResolution()            { return tfResolution; }
    Vector3i GetVolumeDimension()    { return blockDim; }

    void CreateNewMaskVolume();
    void InitTFSettings(string filename);
    void LoadDataSequence(Metadata *meta, int timestep);

private:
    void nomalize(float *pData);

    DataSequence dataSequence;
    Vector3i blockDim;

    int volumeSize;
    int tfResolution;

    float *pMaskVolume;
    float *pTFOpacityMap;
};

#endif // DATAMANAGER_H
