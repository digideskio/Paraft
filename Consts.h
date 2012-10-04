#ifndef CONSTS_H
#define CONSTS_H

#include <hash_map.h>
#include <math.h>
#include <string>
#include <vector>
#include <list>
#include <map>

const int FEATURE_MIN_VOXEL_NUM = 10;

const int INT_SIZE = 1;
const int FLOAT_SIZE = sizeof(float);

const int TF_RESOLUTION = 1024;

const float LOW_THRESHOLD  = 0.2;
const float HIGH_THRESHOLD = 1.0;
const float DIST_THRESHOLD = 9.0;

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

using namespace std;

class MinMax {
public:
    float min, max;
    MinMax(float min_, float max_) : min(min_), max(max_) {}
};

class Vector3i {
public:
    int x, y, z;
    int *toArray() { return &x; }
    int volume() { return x*y*z; }
    float distanceFrom(Vector3i const& r) const { Vector3i l(*this);
        return sqrt((l.x-r.x)*(l.x-r.x)+(l.y-r.y)*(l.y-r.y)+(l.z-r.z)*(l.z-r.z));
    }
    Vector3i(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}
    Vector3i operator-() { return Vector3i(-x, -y, -z); }
    Vector3i operator+(Vector3i const& rhs) const { Vector3i t(*this); t+=rhs; return t; }
    Vector3i operator-(Vector3i const& rhs) const { Vector3i t(*this); t-=rhs; return t; }
    Vector3i operator*(Vector3i const& rhs) const { Vector3i t(*this); t*=rhs; return t; }
    Vector3i operator/(Vector3i const& rhs) const { Vector3i t(*this); t/=rhs; return t; }
    Vector3i operator*(int scale) const { Vector3i t(*this); t*=scale; return t; }
    Vector3i operator/(int scale) const { Vector3i t(*this); t/=scale; return t; }
    Vector3i& operator+=(Vector3i const& rhs) { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
    Vector3i& operator-=(Vector3i const& rhs) { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
    Vector3i& operator*=(Vector3i const& rhs) { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
    Vector3i& operator/=(Vector3i const& rhs) { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
    Vector3i& operator*=(int scale) { x*=scale, y*=scale, z*=scale; return *this; }
    Vector3i& operator/=(int scale) { x/=scale, y/=scale, z/=scale; return *this; }
}; typedef Vector3i DataPoint;

class Vector3f {
public:
    float x, y, z;
    float *toArray() { return &x; }
    float volume() { return x*y*z; }
    Vector3f(float x_ = .0f, float y_ = .0f, float z_ = .0f) : x(x_), y(y_), z(z_) {}
    Vector3f operator-() { return Vector3f(-x, -y, -z); }
    Vector3f operator+(Vector3f const& rhs) const { Vector3f t(*this); t+=rhs; return t; }
    Vector3f operator-(Vector3f const& rhs) const { Vector3f t(*this); t-=rhs; return t; }
    Vector3f operator*(Vector3f const& rhs) const { Vector3f t(*this); t*=rhs; return t; }
    Vector3f operator/(Vector3f const& rhs) const { Vector3f t(*this); t/=rhs; return t; }
    Vector3f operator*(float scale) const { Vector3f t(*this); t*=scale; return t; }
    Vector3f operator/(float scale) const { Vector3f t(*this); t/=scale; return t; }
    Vector3f& operator+=(Vector3f const& rhs) { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
    Vector3f& operator-=(Vector3f const& rhs) { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
    Vector3f& operator*=(Vector3f const& rhs) { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
    Vector3f& operator/=(Vector3f const& rhs) { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
    Vector3f& operator*=(float scale) { x*=scale, y*=scale, z*=scale; return *this; }
    Vector3f& operator/=(float scale) { x/=scale, y/=scale, z/=scale; return *this; }
}; typedef Vector3f FloatPoint;

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

struct DataSet {
    int     start;
    int     end;
    string  prefix;
    string  surfix;
    string  path;
    Vector3i dim;
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
    Vector3i    partition;
    int         num_proc;
    int         num_feature;
    double      time_1;
    double      time_2;
    double      time_3;
} CSVWriter;

typedef struct { float x; float y; } Vector2f;
typedef hash_map<int, int> IntMap;
typedef hash_map<int, float> IndexValueMap;
typedef hash_map<int, vector<int> > FeatureTable;
typedef vector<float*> DataVector;
typedef vector<MinMax> MinMaxVector;
typedef unsigned int uint;

#endif // CONSTS_H
