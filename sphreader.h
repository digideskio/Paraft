#ifndef SPHREADER_H
#define SPHREADER_H

#include <cassert>
#include <iostream>
#include <fstream>
#include "Consts.h"

// for sph dataset
typedef struct { int size0; int svType; int dType; int size1; } DataProperty_;
typedef struct { int size0; int x; int y; int z; int size1; } Dimension_;
typedef struct { int size0; float x; float y; float z; int size1; } Origin_, Pitch_;
typedef struct { int size0; int step; float time; int size1; } Time_;

class SphReader {


public:
    SphReader();
    float* loadData(string filePath);
    int getVolumeSize() { return dim.x * dim.y * dim.z; }

private:
    DataProperty_ dp;
    Dimension_ dim;
    Origin_ orig;
    Pitch_ pitch;
    Time_ t;
};

#endif // SPHREADER_H
