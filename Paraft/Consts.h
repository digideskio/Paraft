#ifndef CONSTS_H
#define CONSTS_H

#include <hash_map.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <list>

const int FEATURE_MIN_VOXEL_NUM = 10;

const float LOW_THRESHOLD  = 0.2;
const float HIGH_THRESHOLD = 0.8;
const float DIST_THRESHOLD = 4.0;

const int TRACKING_MODE_DIRECT = 0;
const int TRACKING_MODE_LINEAR = 1;
const int TRACKING_MODE_POLYNO = 2;

const int TRACKING_FORWARD  = 0;
const int TRACKING_BACKWARD = 1;

// Surface
const int SURFACE_NULL   = -1;  // default
const int SURFACE_LEFT   = 0;   // x = 0
const int SURFACE_RIGHT  = 1;   // x = xs
const int SURFACE_BOTTOM = 2;   // y = 0
const int SURFACE_TOP    = 3;   // y = ys
const int SURFACE_FRONT  = 4;   // z = 0
const int SURFACE_BACK   = 5;   // z = zs

const bool IS_BIG_ENDIAN = false;

using namespace std;

template<class T>
class Range {
    T _begin, _end;
public:
    Range(T begin = 0, T end = 0) : _begin(begin), _end(end) { }
    T Begin() const { return _begin; }
    T End() const { return _end; }
    void SetRange(T begin, T end) { _begin = begin; _end = end; }
    void PrintRange() { cout << "(" << _begin << "," << _end << endl; }
};

namespace util {
    template<class T>
    class Vector3 {
    public:
        T x, y, z;
        Vector3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) { }
        int*     GetPointer()                           { return &x; }
        int      Product()                              { return x * y * z; }
        float    Magnitute()                            { return sqrt(x*x + y*y + z*z); }
        float    DistanceFrom(Vector3 const& rhs) const { return (*this - rhs).Magnitute(); }
        Vector3  operator -  ()                         { return Vector3(-x, -y, -z); }
        Vector3  operator +  (Vector3 const& rhs) const { Vector3 t(*this); t+=rhs; return t; }
        Vector3  operator -  (Vector3 const& rhs) const { Vector3 t(*this); t-=rhs; return t; }
        Vector3  operator *  (Vector3 const& rhs) const { Vector3 t(*this); t*=rhs; return t; }
        Vector3  operator /  (Vector3 const& rhs) const { Vector3 t(*this); t/=rhs; return t; }
        Vector3  operator *  (int scale)          const { Vector3 t(*this); t*=scale; return t; }
        Vector3  operator /  (int scale)          const { Vector3 t(*this); t/=scale; return t; }
        Vector3& operator += (Vector3 const& rhs)       { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
        Vector3& operator -= (Vector3 const& rhs)       { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
        Vector3& operator *= (Vector3 const& rhs)       { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
        Vector3& operator /= (Vector3 const& rhs)       { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
        Vector3& operator *= (int scale)                { x*=scale, y*=scale, z*=scale; return *this; }
        Vector3& operator /= (int scale)                { x/=scale, y/=scale, z/=scale; return *this; }
    };
}

typedef util::Vector3<int> Vector3i, DataPoint;

class Edge {
public:
    int id, start, end;
    Vector3i centroid;

    bool operator ==(Edge const& rhs) const {
        Edge lhs(*this);
        if (lhs.id==rhs.id && lhs.start==rhs.start && lhs.end==rhs.end) {
            return true;
        } else return false;
    }
};    // start ---id---> end @ centroid

struct Metadata {
    Range<int>      timeRange;
    string          prefix;
    string          surfix;
    string          path;
    string          tf;
    Vector3i        volumeDim;
    int             tsLength;
};

struct Feature {
    int             ID;             // Unique ID for each feature
    float           MaskValue;      // Used to record the color of the feature
    list<DataPoint> SurfacePoints;  // Edge information of the feature
    list<DataPoint> InnerPoints;    // All the voxels in the feature
    list<float>     Uncertainty;    // Uncertainty measure of each edge points
    Vector3i        Centroid;       // Centers position of the feature
    Vector3i        Min;            // Minimum position (x,y,z) on boundary
    Vector3i        Max;            // Maximum position (x,y,z) on boundary
    Vector3i        BoundaryCentroid[6];   // center point on boundary surface
    Vector3i        BoundaryMin[6];    // min value on boundary surface
    Vector3i        BoundaryMax[6];    // max value on boundary surface
    vector<int>     TouchedSurfaces;
};

typedef struct {
    Vector3i    gridDim;
    int         numProc;
    int         numFeature;
    double      t1;
    double      t2;
    double      t3;
    double      t4;
} CSVWriter;

typedef hash_map<int, int> IntMap;
typedef hash_map<int, float> IndexValueMap;
typedef hash_map<int, vector<int> > FeatureTable;
typedef hash_map<int, float*> DataSequence;
typedef hash_map<int, vector<Feature> > FeatureVectorSequence;

template <class T>
void ReverseEndian(T *pObject) {
    unsigned char *pChar = reinterpret_cast<unsigned char*>(pObject);
    std::reverse(pChar, pChar + sizeof(T));
}

#endif // CONSTS_H
