#include <cstring>

#include "VolumeDataBlock.h"

RegularGridDataBlock::RegularGridDataBlock(RegularGridData *volumeData, const Vector3i &lo, const Vector3i &hi, const Vector3f &boxLo, const Vector3f &boxHi)
    : _data(nullptr),
      _dataSize(0),
      _volumeData(volumeData),
      _lo(lo),
      _hi(hi),
      _boxLo(boxLo),
      _boxHi(boxHi) { }

bool RegularGridDataBlock::load() {
    unload();

    if (!_volumeData->isLoaded())
        return false;

    Vector3i bdim = dim();
    size_t elemCount = (size_t)(bdim.x * bdim.y * bdim.z);
    _dataSize = elemCount * sizeof(float);
    _data = new float[elemCount];
    if (_data == nullptr) {
        return false;
    }

    float *wdata = wholeData();
    const Vector3i &wdim = wholeDim();
    Vector3i blo = lo();
    Vector3i bhi = hi();
    for (int z = 0, zz = blo.z; zz <= bhi.z; z++, zz++)
        for (int y = 0, yy = blo.y; yy <= bhi.y; y++, yy++)
            memcpy(&_data[(z * bdim.y + y) * bdim.x], &wdata[(zz * wdim.y + yy) * wdim.x + blo.x], sizeof(float) * bdim.x);

    return true;
}

void RegularGridDataBlock::unload() {
    if (isLoaded()) {
        delete [] _data;
        _data = nullptr;
        _dataSize = 0;
    }
}

PRegularGridData::PRegularGridData() : RegularGridData() { }
PRegularGridData::~PRegularGridData() { //unload();
}

void PRegularGridData::initSubblocks(const VolumeMetadata &metadata,
                                     const Vector3i &gridDim, int padding) {
    _metadata = &metadata;
    Vector3f gridDimf(gridDim);
    const Vector3i &dataDim = _metadata->dim();
    Vector3f dataDimf(dataDim);
    Vector3f boxDim = scaledDim();

    for (int z = 0; z < gridDim.z; z++) {
        for (int y = 0; y < gridDim.y; y++) {
            for (int x = 0; x < gridDim.x; x++) {
                Vector3i grid(x, y, z);
                Vector3f gridf(grid);
                Vector3f fLo = dataDimf * gridf / gridDimf;
                Vector3f fHi = (dataDimf * Vector3f(grid + Vector3i(1, 1, 1)) / gridDimf - Vector3f(1.0f, 1.0f, 1.0f));
                Vector3i lo((int)floor(fLo.x), (int)floor(fLo.y), (int)floor(fLo.z));
                Vector3i hi((int)ceil(fHi.x), (int)ceil(fHi.y), (int)ceil(fHi.z));
                lo.x = std::max(lo.x - padding, 0);
                lo.y = std::max(lo.y - padding, 0);
                lo.z = std::max(lo.z - padding, 0);
                hi.x = std::min(hi.x + padding, dataDim.x - 1);
                hi.y = std::min(hi.y + padding, dataDim.y - 1);
                hi.z = std::min(hi.z + padding, dataDim.z - 1);
                Vector3f boxLo = boxDim * gridf / gridDimf;
                Vector3f boxHi = boxDim * Vector3f(grid + Vector3i(1, 1, 1)) / gridDimf;
                _blocks.append(RegularGridDataBlock(this, lo, hi, boxLo, boxHi));
            }
        }
    }
}

RegularGridDataBlock &PRegularGridData::volumeDataBlock(int blockIndex) {
    if (!isLoaded())
        load(*_metadata);
    if (!_blocks[blockIndex].isLoaded())
        _blocks[blockIndex].load();
    return _blocks[blockIndex];
}

Vector3f PRegularGridData::scaledDim() const {
    const Vector3i &d = _metadata->dim();
    float maxDim = (float)std::max(d.x, std::max(d.y, d.z));
    return (Vector3f(d) / maxDim);
}

void PRegularGridData::unload() {
    std::cout << "PRegularGridData::unload()" << std::endl;
    RegularGridData::unload();
    for (int i = 0; i < blockCount(); i++)
        _blocks[i].unload();
}
