#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include "Utils.h"
#include "Metadata.h"

class DataManager {

public:
    DataManager();
   ~DataManager();

    float* GetDataPtr(int t)    { return dataSequence_[t]; }
    float* GetTFMap(int t)      { return tfSequence_[t]; }
    int GetTFRes()              { return tfRes_ < 1 ? DEFAULT_TF_RES : tfRes_; }
    Vector3i GetBlockDim()      { return blockDim_; }

    void InitTF(const Metadata &meta);
    void LoadDataSequence(const Metadata &meta, const int timestep);
    void SaveMaskVolume(float *pData, const Metadata &meta, const int timestep);
private:
    int preprocessData(float *pData, bool remapping);   // returns peak value position
    void normalize(float *pData);

    DataSeq dataSequence_;
    DataSeq tfSequence_;
    Vector3i blockDim_;

    int volumeSize_;
    int tfRes_;
    float *pStaticTfMap_;
};

#endif // DATAMANAGER_H
