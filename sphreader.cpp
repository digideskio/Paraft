#include "sphreader.h"

SphReader::SphReader() {
    // DataProperty
    dp.size0 = 0; dp.svType = 0; dp.dType = 0; dp.size1 = 0;
    // Dimension
    dim.size0 = 0; dim.x = 0; dim.y = 0; dim.z = 0; dim.size1 = 0;
    // Origin
    orig.size0 = 0; orig.x = 0.0; orig.y = 0.0; orig.z = 0.0; orig.size1 = 0;
    // Pitch
    pitch.size0 = 0; pitch.x = 0.0; pitch.y = 0.0; pitch.z = 0.0; pitch.size1 = 0;
    // Time
    t.size0 = 0; t.step = 0; t.time = 0.0; t.size1 = 0;
}

float* SphReader::loadData(string filePath) {
    char *filename = new char[filePath.size() + 1];
    copy(filePath.begin(), filePath.end(), filename);
    filename[filePath.size()] = '\0';

    ifstream sph(filename, ios::binary);
    if (!sph) { cerr << "Fail to open " << filename << endl; return 0; }

    sph.read(reinterpret_cast<char *>(&dp), sizeof(DataProperty_));
    if (dp.size0 != dp.size1) {
        cerr << "DataProperty load error." << endl; return 0;
    }

    sph.read(reinterpret_cast<char *>(&dim), sizeof(Dimension_));
    if (dim.size0 != dim.size1) {
        cerr << "Dimension load error." << endl; return 0;
    }

    sph.read(reinterpret_cast<char *>(&orig), sizeof(Origin_));
    if (orig.size0 != orig.size1) {
        cerr << "Origin load error." << endl; return 0;
    }

    sph.read(reinterpret_cast<char *>(&pitch), sizeof(Pitch_));
    if (pitch.size0 != pitch.size1) {
        cerr << "Pitch load error." << endl; return 0;
    }

    sph.read(reinterpret_cast<char *>(&t), sizeof(Time_));
    if (t.size0 != t.size1) {
        cerr << "Time load error." << endl; return 0;
    }

    int unitSize = dp.dType == 1 ? sizeof(float) : sizeof(double);
    int svCoeff = dp.svType == 1 ? 1 /*scalar*/ : 3 /*vector*/;

    int volumeSize = dim.x * dim.y * dim.z;
    int expectedDataSize = volumeSize * unitSize * svCoeff;
    int actualDataSize = 0;
    sph.read(reinterpret_cast<char *>(&actualDataSize), sizeof(int));

    if (actualDataSize != expectedDataSize) {
        cerr << "volume size not match." << endl;
        cerr << "  expected = " << expectedDataSize << endl;
        cerr << "    actual = " << actualDataSize << endl;
        return 0;
    }

    float *pData = new float[volumeSize];
    sph.read(reinterpret_cast<char *>(pData), actualDataSize);

    int eofDataSize = 0;	// test if reach EOF
    sph.read(reinterpret_cast<char *>(&eofDataSize), sizeof(int));
    if (eofDataSize != actualDataSize) {
        cerr << "has not reach eof, actualDataSize not correct";
    }

    sph.close();
    delete [] filename;
    return pData;
}
