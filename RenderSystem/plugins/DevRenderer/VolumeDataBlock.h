#ifndef VOLUMEDATABLOCK_H
#define VOLUMEDATABLOCK_H

#include "MSVectors.h"
#include "Containers.h"
#include "VolumeData.h"

class RegularGridDataBlock : public MSVolumeData {
public:
    RegularGridDataBlock(RegularGridData *volumeData, const Vector3i &lo, const Vector3i &hi, const Vector3f &boxLo, const Vector3f &boxHi);

    float *data() { return _data; }
    size_t dataSize() const { return _dataSize; }
    RegularGridData *wholeVolumeData() { return _volumeData; }
    float *wholeData() { return _volumeData->data(); }
    const Vector3i &wholeDim() const { return _volumeData->dim(); }
    const Vector3i &lo() const { return _lo; }
    const Vector3i &hi() const { return _hi; }
    const Vector3f &boxLo() const { return _boxLo; }
    const Vector3f &boxHi() const { return _boxHi; }
    Vector3i dim() const { return (_hi - _lo + Vector3i(1, 1, 1)); }
    Vector3f scaledLo() const { return (Vector3f(_lo) / max(wholeDim())); }
    Vector3f scaledHi() const { return (Vector3f(_hi + Vector3i(1, 1, 1)) / max(wholeDim())); }
    Vector3f scaledDim() const { return (Vector3f(dim()) / max(wholeDim())); }
    Vector3f boxCenter() const { return ((_boxLo + _boxHi) * 0.5f); }

    bool load();
    void unload();
    bool isLoaded() const { return (_data != nullptr); }

protected:
    static float max(const Vector3i &v);

protected:
    float *_data;
    size_t _dataSize;
    RegularGridData *_volumeData;
    Vector3i _lo;           // lo and hi defines the padded subblock
    Vector3i _hi;
    Vector3f _boxLo;        // box defines the actual valid area (scaled) of the subblock
    Vector3f _boxHi;        // the area outside the box is the padding
};

inline float RegularGridDataBlock::max(const Vector3i &v) {
    return (float)std::max(v.x, std::max(v.y, v.z));
}

// partitioned regular grid data
class PRegularGridData : public RegularGridData {
public:
    PRegularGridData();
    virtual ~PRegularGridData();

    void initSubblocks(const VolumeMetadata &metadata, const Vector3i &gridDim, int padding = 4);
    int blockCount() const { return (int)_blocks.size(); }
    RegularGridDataBlock &volumeDataBlock(int blockIndex);
    Vector3f scaledDim() const;
    virtual void unload();

protected:
    const VolumeMetadata *_metadata;
    Vector<RegularGridDataBlock> _blocks;
};

#endif // VOLUMEDATABLOCK_H
