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

typedef struct { int x; int y; int z; } DataPoint, Vector3d;
typedef struct { float x; float y; float z; } FloatPoint, Vector3f;
typedef struct { float sim; int idx; } Info;

struct Edge {
    int id;
    int start;
    int end;
    Vector3d centroid;
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
    Vector3d        Centroid;       // Centers position of the feature
    Vector3d        Min;            // Minimum position (x,y,z) on boundary
    Vector3d        Max;            // Maximum position (x,y,z) on boundary
    Vector3d        BoundaryCentroid[6];   // center point on boundary surface
    Vector3d        BoundaryMin[6];    // min value on boundary surface
    Vector3d        BoundaryMax[6];    // max value on boundary surface
    vector<int>     TouchedSurfaces;
};

typedef struct {
    Vector3d    num_Seg;
    int         num_worker;
    int         num_feature;
    double      time_1;
    double      time_2;
    double      time_3;
} CSVWriter;

#endif // CONSTS_H
