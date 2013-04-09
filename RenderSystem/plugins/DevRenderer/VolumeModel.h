#ifndef VOLUMEMODEL_H
#define VOLUMEMODEL_H

#include "VolumeMetadata.h"
#include "VolumeData.h"
#include "VolumeDataBlock.h"

#include <QtCore>       //// QMap

class VolumeModel {
public:
    VolumeModel(const String &fileName);
    ~VolumeModel();

    const String   &name()                          const { return _volumeMetadata.name(); }
    int             stepCount()                     const { return _volumeMetadata.stepCount(); }
    int             varCount()                      const { return _volumeMetadata.varCount(); }
    const String   &varName(int varIndex)           const { return _volumeMetadata.getVarName(varIndex); }
    int             varIndex(const String &varName) const { return _volumeMetadata.getVarIndex(varName); }

    const Vector3i &dim(int timeStep = 0, int varIndex = 0) const { return _volumeMetadata.getVolumeMetadata(timeStep, varIndex).dim(); }
    Vector3f        scaledDim(int timeStep = 0, int varIndex = 0) const;
    double          max(int timeStep = 0, int varIndex = 0) const { return _volumeMetadata.getVolumeMetadata(timeStep, varIndex).max(); }
    double          min(int timeStep = 0, int varIndex = 0) const { return _volumeMetadata.getVolumeMetadata(timeStep, varIndex).min(); }

    RegularGridData &volumeData(int timeStep = 0, int varIndex = 0);
    float *data(int timeStep = 0, int varIndex = 0);

    void initSubblocks(const Vector3i &gridDim, int padding = 4);
    int blockCount(int timeStep = 0, int varIndex = 0) const { return _pvolumes[timeStep][varIndex]->blockCount(); } //{ return (int)_blocks[timeStep][varIndex].size(); }

    void loadData(int timeStep = 0, int varIndex = 0);
    RegularGridDataBlock &volumeDataBlock(int blockIndex, int timeStep = 0, int varIndex = 0);
    float *dataBlock(int blockIndex, int timeStep = 0, int varIndex = 0);

protected:
    void _setTimeStamp(PRegularGridData *volume, int timeStamp);

protected:
    TVMVVolumeMetadata _volumeMetadata;
    Hash<PRegularGridData *, int> _timeStamps;
    QMap<int, PRegularGridData *> _pqueue;
    int _currentTime;

    Vector< Vector<PRegularGridData *> > _pvolumes;     // for segmented ray casting
};

#endif // VOLUMEMODEL_H
