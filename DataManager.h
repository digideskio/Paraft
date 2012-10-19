#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <mpi.h>
#include "Consts.h"

class DataManager {

public:
    DataManager();
    ~DataManager();

    void CreateNewMaskVolume();

    float* GetDataPointer(int index) { return dataSequence[index]; }
    float* GetTFOpacityMap() { return pTFOpacityMap; }
    int GetTFResolution() { return tfResolution; }

    Vector3i GetVolumeDimension() { return blockSize; }

    void InitTFSettings(string filename);
    void PreloadDataSequence(Vector3i partition, Vector3i blockCoord, DataSet ds, int timestep);

private:
    DataSequence dataSequence;
    Vector3i blockSize;

    int volumeSize;
    int tfResolution;

    float *pMaskVolume;
    float *pTFOpacityMap;

    void normalizeData(DataSet ds);
};

#endif // DATAMANAGER_H
