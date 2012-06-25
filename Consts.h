#ifndef CONSTS_H
#define CONSTS_H

#include <hash_map.h>
#include <math.h>
#include <string>
#include <vector>
#include <list>
#include <map>

const int HOST_NODE = 0;

const int FEATURE_MIN_VOXEL_NUM = 10;
const int NUM_TRACK_STEPS = 5;

const int INT_SIZE = 1;
const int FLOAT_SIZE = sizeof(float);

const int DATA_DIM_X = 128;
const int DATA_DIM_Y = 128;
const int DATA_DIM_Z = 128;

const int TF_RESOLUTION = 1024;

const float LOW_THRESHOLD   = 0.2;
const float HIGH_THRESHOLD  = 1.0;

const int TRACKING_MODE_DIRECT = 0;
const int TRACKING_MODE_LINEAR = 1;
const int TRACKING_MODE_POLYNO = 2;

const int TRACKING_FORWARD  = 0;
const int TRACKING_BACKWARD = 1;

// Surface
const int SURFACE_NULL   = -1;   // for init
const int SURFACE_LEFT   = 0;   // x = 0
const int SURFACE_RIGHT  = 1;   // x = xs
const int SURFACE_BOTTOM = 2;   // y = 0
const int SURFACE_TOP    = 3;   // y = ys
const int SURFACE_FRONT  = 4;   // z = 0
const int SURFACE_BACK   = 5;   // z = zs

// MPI TAG
const int MPI_TAG_NULL = -1;
const int MPI_TAG_TIMESTEP_0 = 0;
const int MPI_TAG_TF_RESOLUTION = 2;
const int MPI_TAG_TF_COLOR_MAP = 3;
const int MPI_TAG_SEGMENT_MATRIX = 4;
const int MPI_TAG_ROUTER = 5;
const int MPI_TAG_HIGHLIGHT_FEATURE = 6;
const int MPI_TAG_SELECTED_FEATURE_INFO = 7;
const int MPI_TAG_SELECTED_FEATURE_INFO_SIZE = 8;
const int MPI_TAG_SYNC_TIMESTEP = 9;
const int MPI_TAG_TRACK_FORWARD = 10;
const int MPI_TAG_GET_FEATURE_ID = 11;
const int MPI_TAG_SET_FEATURE_ID = 12;

using namespace std;

typedef struct { float x; float y; } Vector2f;
typedef hash_map<int, int> IntMap;
typedef hash_map<int, float> IndexValueMap;

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
    Vector3i operator*(float scale) const { Vector3i t(*this); t*=scale; return t; }
    Vector3i operator/(float scale) const { Vector3i t(*this); t/=scale; return t; }
    Vector3i& operator+=(Vector3i const& rhs) { x+=rhs.x, y+=rhs.y, z+=rhs.z; return *this; }
    Vector3i& operator-=(Vector3i const& rhs) { x-=rhs.x, y-=rhs.y, z-=rhs.z; return *this; }
    Vector3i& operator*=(Vector3i const& rhs) { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
    Vector3i& operator/=(Vector3i const& rhs) { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
    Vector3i& operator*=(float scale) { x*=scale, y*=scale, z*=scale; return *this; }
    Vector3i& operator/=(float scale) { x/=scale, y/=scale, z/=scale; return *this; }
}; typedef Vector3i DataPoint;

class Vector3f {
public:
    float x, y, z;
    float *toArray() { return &x; }
    int volume() { return x*y*z; }
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

struct Edge {
    int id;
    int start;
    int end;
    Vector3i centroid;
};    // start ---id---> end @ centroid

struct DataSet {
    int     index_start;
    int     index_end;
    string  prefix;
    string  surfix;
    string  data_path;
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
    int         num_worker;
    int         num_feature;
    double      time_1;
    double      time_2;
    double      time_3;
} CSVWriter;

#endif // CONSTS_H
