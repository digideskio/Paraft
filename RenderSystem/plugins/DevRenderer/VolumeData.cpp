#include <iostream>
#include <fstream>
#include <cmath>
#include "VolumeData.h"

#define nullptr 0

RegularGridData::RegularGridData() : _data(nullptr), _dataSize(0) { }
RegularGridData::~RegularGridData() { unload(); }

bool RegularGridData::load(const VolumeMetadata &metadata) {
    unload();

    size_t offset = (size_t)metadata.offset();
    _dim = metadata.dim();

    if (_dim.x < 0 || _dim.y < 0 || _dim.z < 0) {
        return false;
    }

    size_t volumeSize = (size_t)(_dim.x * _dim.y * _dim.z);

    std::ifstream ifs;
    ifs.open(metadata.fileName().c_str(), std::ios::in | std::ios::binary);
    if (ifs.fail()) {
        return false;
    }

    size_t unitSize = 0;
    switch (metadata.type()) {
        case VolumeMetadata::UNSIGNED_8BIT:
        case VolumeMetadata::SIGNED_8BIT:
            unitSize = sizeof(char);
            break;
        case VolumeMetadata::UNSIGNED_16BIT:
        case VolumeMetadata::SIGNED_16BIT:
            unitSize = sizeof(char) * 2;
            break;
        case VolumeMetadata::UNSIGNED_32BIT:
        case VolumeMetadata::SIGNED_32BIT:
            unitSize = sizeof(char) * 4;
            break;
        case VolumeMetadata::FLOAT:
            unitSize = sizeof(float);
            break;
        case VolumeMetadata::DOUBLE:
            unitSize = sizeof(double);
            break;
        default:
            return false;
    }

    size_t rawDataSize = volumeSize * unitSize;

    ifs.seekg(0, std::ios::end);
    size_t fileSize = ifs.tellg();
    if (fileSize < offset + rawDataSize) {
        return false;
    }

    char *rawData = new char[rawDataSize];
    if (rawData == nullptr) {
        return false;
    }

    ifs.seekg(offset, std::ios::beg);
    ifs.read(rawData, rawDataSize);
    if (ifs.fail()) {
        return false;
    }
    ifs.close();

    // reverse byte order if necessary
    if (metadata.byteOrder() != VolumeMetadata::nativeByteOrder() &&
        metadata.byteOrder() != VolumeMetadata::UNKNOWN_ORDER) {
        for (size_t i = 0; i < rawDataSize; i += unitSize) {
            char elem[8];
            for (size_t j = 0; j < unitSize; j++) {
                elem[unitSize - 1 - j] = rawData[i + j];
            }
            for (size_t j = 0; j < unitSize; j++) {
                rawData[i + j] = elem[j];
            }
        }
    }

    _dataSize = volumeSize * sizeof(float);
    _data = new float[volumeSize];

    if (_data == nullptr) {
        return false;
    }

    switch (metadata.type()) {
        case VolumeMetadata::UNSIGNED_8BIT:
            for (size_t i = 0; i < volumeSize; i++)
                _data[i] = (float)((unsigned char *)rawData)[i];
            break;
        case VolumeMetadata::SIGNED_8BIT:
            for (size_t i = 0; i < volumeSize; i++)
                _data[i] = (float)((char *)rawData)[i];
            break;
        case VolumeMetadata::UNSIGNED_16BIT:
            for (size_t i = 0; i < volumeSize; i++)
                _data[i] = (float)((unsigned short *)rawData)[i];
            break;
        case VolumeMetadata::SIGNED_16BIT:
            for (size_t i = 0; i < volumeSize; i++)
                _data[i] = (float)((short *)rawData)[i];
            break;
        case VolumeMetadata::UNSIGNED_32BIT:
            for (size_t i = 0; i < volumeSize; i++)
                _data[i] = (float)((unsigned int *)rawData)[i];
            break;
        case VolumeMetadata::SIGNED_32BIT:
            for (size_t i = 0; i < volumeSize; i++)
                _data[i] = (float)((int *)rawData)[i];
            break;
        case VolumeMetadata::FLOAT:
            memcpy((void *)_data, (void *)rawData, _dataSize);
            break;
        case VolumeMetadata::DOUBLE:
            for (size_t i = 0; i < volumeSize; i++)
                _data[i] = (float)((double *)rawData)[i];
            break;
        default:    // unknown
            return false;
    }

//    remapping();

    Vector2f range;  // min, max
    if (metadata.rangeDefined())
        range = (Vector2f)metadata.range();
    else
        range = getRange();
    normalize(range.x, range.y);

    delete [] rawData;

    return true;
}

void RegularGridData::remapping() {
    if (!isLoaded()) { return; }
    size_t elemCount = _dim.x * _dim.y * _dim.z;
    for (size_t i = 0; i < elemCount; i++) {
        _data[i] = powf(_data[i], 0.1);
    }
}

void RegularGridData::normalize(float min, float max) {
    if (!isLoaded() || min == max) { return; }
    float invRange = 1.0f / (max - min);
    size_t elemCount = _dim.x * _dim.y * _dim.z;
    for (size_t i = 0; i < elemCount; i++) {
        _data[i] = (_data[i] - min) * invRange;
    }
}

Vector2f RegularGridData::getRange() const {
    if (!isLoaded()) {
        return Vector2f();
    }
    float min = _data[0];
    float max = _data[0];
    size_t elemCount = _dim.x * _dim.y * _dim.z;
    for (size_t i = 0; i < elemCount; i++) {
        min = std::min(min, _data[i]);
        max = std::max(max, _data[i]);
    }
    return Vector2f(min, max);
}

void RegularGridData::unload() {
    if (isLoaded()) {
        delete [] _data;
        _data = nullptr;
        _dataSize = 0;
    }
}

//} // namespace MSLib
