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
    Range(T begin, T end) : _begin(begin), _end(end) { }
    T Begin() const { return _begin; }
    T End() const { return _end; }
    void SetRange(T begin, T end) { _begin = begin; _end = end; }
    void PrintRange() { cout << "(" << _begin << "," << _end << endl; }
};

template<class T>
class Vector3 {
    T _x, _y, _z;
public:
    Vector3(T x = 0, T y = 0, T z = 0) : _x(x), _y(y), _z(z) { }
    T X() { return _x; }
    T Y() { return _y; }
    T Z() { return _z; }
    int* GetPointer() { return &_x; }
    int Product() { return _x * _y * _z; }
    float DistanceFrom(Vector3 const& rhs) const {
        Vector3 lhs(*this);
        return sqrt((lhs.X() - rhs.X()) * (lhs.X() - rhs.X()) +
                    (lhs.Y() - rhs.Y()) * (lhs.Y() - rhs.Y()) +
                    (lhs.Z() - rhs.Z()) * (lhs.Z() - rhs.Z()));
    }
    Vector3 operator-() { return Vector3(-_x, -_y, -_z); }
    Vector3 operator+(Vector3 const& rhs) const { Vector3 t(*this); t+=rhs; return t; }
    Vector3 operator-(Vector3 const& rhs) const { Vector3 t(*this); t-=rhs; return t; }
    Vector3 operator*(Vector3 const& rhs) const { Vector3 t(*this); t*=rhs; return t; }
    Vector3 operator/(Vector3 const& rhs) const { Vector3 t(*this); t/=rhs; return t; }
    Vector3 operator*(int scale) const { Vector3 t(*this); t*=scale; return t; }
    Vector3 operator/(int scale) const { Vector3 t(*this); t/=scale; return t; }
    Vector3& operator+=(Vector3 const& rhs) { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
    Vector3& operator-=(Vector3 const& rhs) { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
    Vector3& operator*=(Vector3 const& rhs) { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
    Vector3& operator/=(Vector3 const& rhs) { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
    Vector3& operator*=(int scale) { x*=scale, y*=scale, z*=scale; return *this; }
    Vector3& operator/=(int scale) { x/=scale, y/=scale, z/=scale; return *this; }
}; typedef Vector3 DataPoint;

//class Vector3i {
//public:
//    int x, y, z;
//    int *toArray() { return &x; }
//    int volume() { return x*y*z; }
//    float distanceFrom(Vector3i const& r) const { Vector3i l(*this);
//        return sqrt((l.x-r.x)*(l.x-r.x)+(l.y-r.y)*(l.y-r.y)+(l.z-r.z)*(l.z-r.z));
//    }
//    Vector3i(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}
//    Vector3i operator-() { return Vector3i(-x, -y, -z); }
//    Vector3i operator+(Vector3i const& rhs) const { Vector3i t(*this); t+=rhs; return t; }
//    Vector3i operator-(Vector3i const& rhs) const { Vector3i t(*this); t-=rhs; return t; }
//    Vector3i operator*(Vector3i const& rhs) const { Vector3i t(*this); t*=rhs; return t; }
//    Vector3i operator/(Vector3i const& rhs) const { Vector3i t(*this); t/=rhs; return t; }
//    Vector3i operator*(int scale) const { Vector3i t(*this); t*=scale; return t; }
//    Vector3i operator/(int scale) const { Vector3i t(*this); t/=scale; return t; }
//    Vector3i& operator+=(Vector3i const& rhs) { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
//    Vector3i& operator-=(Vector3i const& rhs) { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
//    Vector3i& operator*=(Vector3i const& rhs) { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
//    Vector3i& operator/=(Vector3i const& rhs) { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
//    Vector3i& operator*=(int scale) { x*=scale, y*=scale, z*=scale; return *this; }
//    Vector3i& operator/=(int scale) { x/=scale, y/=scale, z/=scale; return *this; }
//}; typedef Vector3i DataPoint;

//class Vector3f {
//public:
//    float x, y, z;
//    float *toArray() { return &x; }
//    float volume() { return x*y*z; }
//    Vector3f(float x_ = .0f, float y_ = .0f, float z_ = .0f) : x(x_), y(y_), z(z_) {}
//    Vector3f operator-() { return Vector3f(-x, -y, -z); }
//    Vector3f operator+(Vector3f const& rhs) const { Vector3f t(*this); t+=rhs; return t; }
//    Vector3f operator-(Vector3f const& rhs) const { Vector3f t(*this); t-=rhs; return t; }
//    Vector3f operator*(Vector3f const& rhs) const { Vector3f t(*this); t*=rhs; return t; }
//    Vector3f operator/(Vector3f const& rhs) const { Vector3f t(*this); t/=rhs; return t; }
//    Vector3f operator*(float scale) const { Vector3f t(*this); t*=scale; return t; }
//    Vector3f operator/(float scale) const { Vector3f t(*this); t/=scale; return t; }
//    Vector3f& operator+=(Vector3f const& rhs) { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
//    Vector3f& operator-=(Vector3f const& rhs) { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
//    Vector3f& operator*=(Vector3f const& rhs) { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
//    Vector3f& operator/=(Vector3f const& rhs) { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
//    Vector3f& operator*=(float scale) { x*=scale, y*=scale, z*=scale; return *this; }
//    Vector3f& operator/=(float scale) { x/=scale, y/=scale, z/=scale; return *this; }
//};

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
    Range   timeRange;
    Range   valueRange;
    string  prefix;
    string  surfix;
    string  path;
    string  tf;
    Vector3 volumeDim;
};

struct Feature {
    int             ID;             // Unique ID for each feature
    float           MaskValue;      // Used to record the color of the feature
    list<DataPoint> SurfacePoints;  // Edge information of the feature
    list<DataPoint> InnerPoints;    // All the voxels in the feature
    list<float>     Uncertainty;    // Uncertainty measure of each edge points
    Vector3        Centroid;       // Centers position of the feature
    Vector3        Min;            // Minimum position (x,y,z) on boundary
    Vector3        Max;            // Maximum position (x,y,z) on boundary
    Vector3        BoundaryCentroid[6];   // center point on boundary surface
    Vector3        BoundaryMin[6];    // min value on boundary surface
    Vector3        BoundaryMax[6];    // max value on boundary surface
    vector<int>     TouchedSurfaces;
};

typedef struct {
    Vector3    gridDim;
    int         num_proc;
    int         num_feature;
    double      t1;
    double      t2;
    double      t3;
    double      t4;
} CSVWriter;

typedef struct { float x; float y; } Vector2f;
typedef hash_map<int, int> IntMap;
typedef hash_map<int, float> IndexValueMap;
typedef hash_map<int, vector<int> > FeatureTable;
typedef hash_map<int, float*> DataSequence;
typedef hash_map<int, vector<Feature> > FeatureVectorSequence;
typedef vector<Range> RangeVector;
typedef unsigned int uint;

template <class T>
void ReverseEndian(T *pObject) {
    unsigned char *pChar = reinterpret_cast<unsigned char*>(pObject);
    std::reverse(pChar, pChar + sizeof(T));
}

#endif // CONSTS_H
