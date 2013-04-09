#ifndef VOLUMEDATA_H
#define VOLUMEDATA_H

#include "VolumeMetadata.h"

#define nullptr 0

typedef std::string String;

class MSVolumeData {
public:
    virtual ~MSVolumeData() {}
};

class RegularGridData : public MSVolumeData {
public:
    RegularGridData();
    virtual ~RegularGridData();

    float *data() { return _data; }
    size_t dataSize() const { return _dataSize; }
    const Vector3i &dimensions() const { return _dim; }
    const Vector3i &dim() const { return _dim; }            // alias of dimensions()
    bool load(const VolumeMetadata &metadata);
    void unload();
    bool isLoaded() const { return (_data != nullptr); }
    void remapping();
    void normalize(float min, float max);
    Vector2f getRange() const;  // x: min, y: max

protected:
    float *_data;
    size_t _dataSize;           // data size in bytes
    Vector3i _dim;
};

#endif // VOLUMEDATA_H
