#include "VolumeModel.h"

////
#include <QtGui>

VolumeModel::VolumeModel(const String &fileName) {
    _currentTime = 0;
    _volumeMetadata.readFile(fileName);

    // use the metadata file's path as default path
    QFileInfo fileInfo(QString::fromStdString(fileName));
    QDir dir = fileInfo.dir();
    for (int i = 0; i < stepCount(); i++) {
        for (int j = 0; j < varCount(); j++) {
            QFileInfo fi(dir, QString::fromStdString(_volumeMetadata.getVolumeMetadata(i, j).fileName()));
            _volumeMetadata.getVolumeMetadata(i, j).setFileName(fi.filePath().toStdString());
        }
    }

    // set universal (among all time steps) range for normalization
    Vector<double> minVal, maxVal;
    for (int i = 0; i < varCount(); i++) {
        minVal.append(_volumeMetadata.getVolumeMetadata(0, i).min());
        maxVal.append(_volumeMetadata.getVolumeMetadata(0, i).max());
    }
    for (int i = 1; i < stepCount(); i++) {
        for (int j = 0; j < varCount(); j++) {
            minVal[j] = std::min(minVal[j], _volumeMetadata.getVolumeMetadata(i, j).min());
            maxVal[j] = std::max(maxVal[j], _volumeMetadata.getVolumeMetadata(i, j).max());
        }
    }
    for (int i = 0; i < stepCount(); i++) {
        for (int j = 0; j < varCount(); j++) {
            _volumeMetadata.getVolumeMetadata(i, j).setRange(minVal[j], maxVal[j]);
        }
    }
    for (int i = 0; i < stepCount(); i++) {
        _pvolumes.append(Vector<PRegularGridData *>());
        for (int j = 0; j < varCount(); j++) {
            _pvolumes[i].append(new PRegularGridData());
            _timeStamps[_pvolumes[i][j]] = -1;
        }
    }
}

VolumeModel::~VolumeModel() {
    qDebug("VolumeModel(%s) deleted.", name().c_str());
    for (size_t i = 0; i < _pvolumes.size(); i++) {
        for (size_t j = 0; j < _pvolumes[i].size(); j++) {
            delete _pvolumes[i][j];
        }
    }
}

Vector3f VolumeModel::scaledDim(int timeStep, int varIndex) const {
    const Vector3i &d = dim(timeStep, varIndex);
    float maxDim = (float)std::max(d.x, std::max(d.y, d.z));
    return (Vector3f(d) / maxDim);
}

RegularGridData &VolumeModel::volumeData(int timeStep, int varIndex) {
    if (_pvolumes[timeStep][varIndex]->isLoaded()) {
        _setTimeStamp(_pvolumes[timeStep][varIndex], _currentTime++);
    } else {
        if (_pqueue.size() >= 4) {
            PRegularGridData *popVolume = *_pqueue.begin();
            _setTimeStamp(popVolume, -1);
            popVolume->unload();    // unload the least recently used
        }
        _setTimeStamp(_pvolumes[timeStep][varIndex], _currentTime++);
        qDebug("load %s...", _volumeMetadata.getVolumeMetadata(timeStep, varIndex).fileName().c_str());
        _pvolumes[timeStep][varIndex]->load(_volumeMetadata.getVolumeMetadata(timeStep, varIndex));
    }
    return *_pvolumes[timeStep][varIndex];
}

float *VolumeModel::data(int timeStep, int varIndex) {
    return volumeData(timeStep, varIndex).data();
}

// used in volumeData()
void VolumeModel::_setTimeStamp(PRegularGridData *volume, int timeStamp) {
    int oldTimeStamp = _timeStamps[volume];
    if (oldTimeStamp >= 0) {  // already loaded
        _pqueue.remove(oldTimeStamp);
    }
    if (timeStamp >= 0) {  // not going to be unloaded
        _pqueue.insert(timeStamp, volume);
    }
    _timeStamps[volume] = timeStamp;
}

void VolumeModel::initSubblocks(const Vector3i &gridDim, int padding) {
    for (int i = 0; i < stepCount(); i++) {
        for (int j = 0; j < varCount(); j++) {
            _pvolumes[i][j]->initSubblocks(_volumeMetadata.getVolumeMetadata(i, j), gridDim, padding);
        }
    }
}

void VolumeModel::loadData(int timeStep, int varIndex) {
    if (_pvolumes[timeStep][varIndex]->isLoaded()) {
        _setTimeStamp(_pvolumes[timeStep][varIndex], _currentTime++);
    } else {
        if (_pqueue.size() >= 4) {
            PRegularGridData *popVolume = *_pqueue.begin();
            _setTimeStamp(popVolume, -1);
            popVolume->unload();    // unload the least recently used
        }
        _setTimeStamp(_pvolumes[timeStep][varIndex], _currentTime++);

        qDebug("load %s...", _volumeMetadata.getVolumeMetadata(timeStep, varIndex).fileName().c_str());
        if (!_pvolumes[timeStep][varIndex]->load(_volumeMetadata.getVolumeMetadata(timeStep, varIndex))) {
            qDebug("Error: Cannot load data file");
        }

    }
}

RegularGridDataBlock &VolumeModel::volumeDataBlock(int blockIndex, int timeStep, int varIndex) {
    return _pvolumes[timeStep][varIndex]->volumeDataBlock(blockIndex);
}

float *VolumeModel::dataBlock(int blockIndex, int timeStep, int varIndex) {
    return _pvolumes[timeStep][varIndex]->volumeDataBlock(blockIndex).data();
}
