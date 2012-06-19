#ifndef CONSTS_H
#define CONSTS_H

#include <vector>
#include <list>
#include <map>

#define HOST_NODE 0

#define FEATURE_MIN_VOXEL_NUM 10
#define NUM_TRACK_STEPS 5

#define DATA_DIM_X 128
#define DATA_DIM_Y 128
#define DATA_DIM_Z 128

#define TF_RESOLUTION 1024

#define TRACKING_MODE_DIRECT    0
#define TRACKING_MODE_LINEAR    1
#define TRACKING_MODE_POLYNO    2

#define TRACKING_DIRECTION_FORWARD  0
#define TRACKING_DIRECTION_BACKWARD 1

// Surface
#define SURFACE_NULL   -1   // for init
#define SURFACE_LEFT    0   // x = 0
#define SURFACE_RIGHT   1   // x = xs
#define SURFACE_BOTTOM  2   // y = 0
#define SURFACE_TOP     3   // y = ys
#define SURFACE_FRONT   4   // z = 0
#define SURFACE_BACK    5   // z = zs

using namespace std;

//typedef struct { int x; int y; int z; } DataPoint, Vector3i;
typedef struct { float x; float y; } Vector2f;

class Vector3i {
public:
//    int &x, &y, &z;
//    int v[3];
//    union { struct { int x, y, z; }; int v[3]; };
    int x, y, z;
    Vector3i(int X = 0, int Y = 0, int Z = 0) : x(X), y(Y), z(Z) {}
    Vector3i operator-() { return Vector3i(-x, -y, -z); }
    Vector3i operator+(Vector3i const& rhs) const { Vector3i t(*this); t+=rhs; return t; }
    Vector3i operator-(Vector3i const& rhs) const { Vector3i t(*this); t-=rhs; return t; }
    Vector3i operator*(Vector3i const& rhs) const { Vector3i t(*this); t*=rhs; return t; }
    Vector3i operator/(Vector3i const& rhs) const { Vector3i t(*this); t/=rhs; return t; }
    Vector3i& operator+=(Vector3i const& rhs) { x+=rhs.x, y+=rhs.y, z+= rhs.z; return *this; }
    Vector3i& operator-=(Vector3i const& rhs) { x-=rhs.x, y-=rhs.y, z-= rhs.z; return *this; }
    Vector3i& operator*=(Vector3i const& rhs) { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
    Vector3i& operator/=(Vector3i const& rhs) { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
    Vector3i operator*(float scale) const { Vector3i t(*this); t*=scale; return t; }
    Vector3i operator/(float scale) const { Vector3i t(*this); t/=scale; return t; }
    Vector3i& operator*=(float scale) { x*=scale, y*=scale, z*=scale; return *this; }
    Vector3i& operator/=(float scale) { x/=scale, y/=scale, z/=scale; return *this; }
}; typedef Vector3i DataPoint;

class Vector3f {
public:
    float x, y, z;
    Vector3f(float X = 0.0, float Y = 0.0, float Z = 0.0) : x(X), y(Y), z(Z) {}
    Vector3f operator-() { return Vector3f(-x, -y, -z); }
    Vector3f operator+(Vector3f const& rhs) const { Vector3f t(*this); t+=rhs; return t; }
    Vector3f operator-(Vector3f const& rhs) const { Vector3f t(*this); t-=rhs; return t; }
    Vector3f operator*(Vector3f const& rhs) const { Vector3f t(*this); t*=rhs; return t; }
    Vector3f operator/(Vector3f const& rhs) const { Vector3f t(*this); t/=rhs; return t; }
    Vector3f& operator+=(Vector3f const& rhs) { x+=rhs.x, y+=rhs.y, z+= rhs.z; return *this; }
    Vector3f& operator-=(Vector3f const& rhs) { x-=rhs.x, y-=rhs.y, z-= rhs.z; return *this; }
    Vector3f& operator*=(Vector3f const& rhs) { x*=rhs.x, y*=rhs.y, z*=rhs.z; return *this; }
    Vector3f& operator/=(Vector3f const& rhs) { x/=rhs.x, y/=rhs.y, z/=rhs.z; return *this; }
    Vector3f operator*(float scale) const { Vector3f t(*this); t*=scale; return t; }
    Vector3f operator/(float scale) const { Vector3f t(*this); t/=scale; return t; }
    Vector3f& operator*=(float scale) { x*=scale, y*=scale, z*=scale; return *this; }
    Vector3f& operator/=(float scale) { x/=scale, y/=scale, z/=scale; return *this; }
};

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
    Vector3i    num_Seg;
    int         num_worker;
    int         num_feature;
    double      time_1;
    double      time_2;
    double      time_3;
} CSVWriter;

#endif // CONSTS_H
