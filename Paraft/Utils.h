#ifndef CONSTS_H
#define CONSTS_H

#include <hash_map.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <list>
#include <map>

#define nullptr NULL

const float OPACITY_THRESHOLD  = 0.1;
const int MIN_NUM_VOXEL_IN_FEATURE = 10;
const int FT_DIRECT = 0;
const int FT_LINEAR = 1;
const int FT_POLYNO = 2;
const int FT_FORWARD  = 0;
const int FT_BACKWARD = 1;
const int DEFAULT_TF_RES = 1024;

using namespace std;

namespace util {
    template<class T>
    class Vector3 {
    public:
        T x, y, z;
        Vector3(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) { }
        T*       GetPointer()                           { return &x; }
        T        Product()                              { return x * y * z; }
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
        bool     operator == (Vector3 const& rhs) const { return x==rhs.x && y==rhs.y && z==rhs.z; }
        bool     operator != (Vector3 const& rhs) const { return !(*this == rhs); }
    };

    static inline string ltrim(const string &s) {    // trim string from left
        int start = s.find_first_not_of(' ');
        return s.substr(start, s.size() - start);
    }

    static inline string rtrim(const string &s) {    // trim string from right
        return s.substr(0, s.find_last_not_of(' ')+1);
    }

    static inline string trim(const string &s) {     // trim all whitesapces
        return ltrim(rtrim(s));
    }

    static inline bool ascending(const pair<float, int> &lhs, const pair<float, int> &rhs) {
        return lhs.second < rhs.second;
    }

    static inline bool descending(const pair<float, int> &lhs, const pair<float, int> &rhs) {
        return !ascending(lhs, rhs);
    }
}

typedef util::Vector3<int> Vector3i;
typedef util::Vector3<float> Vector3f;
typedef util::Vector3<double> Vector3d;

struct Feature {
    int             id;         // Unique ID for each feature
    float           maskValue;  // Used to record the color of the feature
    list<Vector3i>  edgeVoxels; // Edge information of the feature
    list<Vector3i>  bodyVoxels; // All the voxels in the feature
    Vector3i        centroid;   // Centers position of the feature
};

typedef hash_map<int, float*> DataSequence;
typedef hash_map<int, vector<Feature> > FeatureVectorSequence;

#endif // CONSTS_H
