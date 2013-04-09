#ifndef VECTORS_H
#define VECTORS_H

#include <cmath>

class Vector2i;
class Vector2f;
class Vector2d;
class Vector3i;
class Vector3f;
class Vector3d;
class Vector4i;
class Vector4f;
class Vector4d;

class Vector2i
{
public:
    int x, y;

public:
    Vector2i()                             : x(0),    y(0)    {}
    Vector2i(int xpos, int ypos)           : x(xpos), y(ypos) {}
    Vector2i(const Vector2i &v)            : x(v.x),  y(v.y)  {}
    Vector2i(const int *v, int offset = 1) : x(v[0]), y(v[offset]) {}

    bool      operator != (const Vector2i &v) const;
    Vector2i  operator *  (int factor)        const;
    Vector2i  operator *  (const Vector2i &v) const;        // element-wise
    Vector2i &operator *= (int factor);
    Vector2i &operator *= (const Vector2i &v);              // element-wise
    Vector2i  operator +  (const Vector2i &v) const;
    Vector2i &operator += (const Vector2i &v);
    Vector2i  operator -  (const Vector2i &v) const;
    Vector2i  operator -  ()                  const;        // unary minus
    Vector2i &operator -= (const Vector2i &v);
    Vector2i  operator /  (int divisor)       const;        // integer division
    Vector2i  operator /  (const Vector2i &v) const;        // element-wise integer division
    Vector2i &operator /= (int divisor);                    // integer division
    Vector2i &operator /= (const Vector2i &v);              // element-wise integer division
    Vector2i &operator =  (const Vector2i &v);
    bool      operator == (const Vector2i &v) const;

    int       &operator [] (int index)       { return (&x)[index]; }
    const int &operator [] (int index) const { return (&x)[index]; }

    operator int * () { return &x; }                        // convert the vector into an array

    void      copyTo(int *v, int offset = 1) const;         // copy the elements to an array
};

class Vector2f
{
public:
    float x, y;

public:
    Vector2f()                               : x(0.0f), y(0.0f) {}
    Vector2f(float xpos, float ypos)         : x(xpos), y(ypos) {}
    Vector2f(const Vector2f &v)              : x(v.x),  y(v.y)  {}
    Vector2f(const float *v, int offset = 1) : x(v[0]), y(v[offset]) {}

    Vector2f(const Vector2d &v);

    bool      operator != (const Vector2f &v) const;
    Vector2f  operator *  (float factor)      const;
    Vector2f  operator *  (const Vector2f &v) const;        // element-wise
    Vector2f &operator *= (float factor);
    Vector2f &operator *= (const Vector2f &v);              // element-wise
    Vector2f  operator +  (const Vector2f &v) const;
    Vector2f &operator += (const Vector2f &v);
    Vector2f  operator -  (const Vector2f &v) const;
    Vector2f  operator -  ()                  const;        // unary minus
    Vector2f &operator -= (const Vector2f &v);
    Vector2f  operator /  (float divisor)     const;
    Vector2f  operator /  (const Vector2f &v) const;        // element-wise
    Vector2f &operator /= (float divisor);
    Vector2f &operator /= (const Vector2f &v);              // element-wise
    Vector2f &operator =  (const Vector2f &v);
    bool      operator == (const Vector2f &v) const;

    float       &operator [] (int index)       { return (&x)[index]; }
    const float &operator [] (int index) const { return (&x)[index]; }

    operator float * () { return &x; }                      // convert the vector into an array

    void      copyTo(float *v, int offset = 1) const;       // copy the elements to an array
    float     dot(const Vector2f &v)           const;       // dot product
    Vector2f &normalize();                                  // normalize the vector in place
    Vector2f  normalized()                     const;       // normalized unit vector
    float     length()                         const;       // length of the vector from the origin
    float     lengthSquared()                  const;       // squared length of the vector from the origin
};

class Vector2d
{
public:
    double x, y;

public:
    Vector2d()                                : x(0.0),  y(0.0)  {}
    Vector2d(double xpos, double ypos)        : x(xpos), y(ypos) {}
    Vector2d(const Vector2d &v)               : x(v.x),  y(v.y)  {}
    Vector2d(const double *v, int offset = 1) : x(v[0]), y(v[offset]) {}

    Vector2d(const Vector2f &v);

    bool      operator != (const Vector2d &v) const;
    Vector2d  operator *  (double factor)     const;
    Vector2d  operator *  (const Vector2d &v) const;        // element-wise
    Vector2d &operator *= (double factor);
    Vector2d &operator *= (const Vector2d &v);              // element-wise
    Vector2d  operator +  (const Vector2d &v) const;
    Vector2d &operator += (const Vector2d &v);
    Vector2d  operator -  (const Vector2d &v) const;
    Vector2d  operator -  ()                  const;        // unary minus
    Vector2d &operator -= (const Vector2d &v);
    Vector2d  operator /  (double divisor)    const;
    Vector2d  operator /  (const Vector2d &v) const;        // element-wise
    Vector2d &operator /= (double divisor);
    Vector2d &operator /= (const Vector2d &v);              // element-wise
    Vector2d &operator =  (const Vector2d &v);
    bool      operator == (const Vector2d &v) const;

    double       &operator [] (int index)       { return (&x)[index]; }
    const double &operator [] (int index) const { return (&x)[index]; }

    operator double * () { return &x; }                     // convert the vector into an array

    void      copyTo(double *v, int offset = 1) const;      // copy the elements to an array
    double    dot(const Vector2d &v)            const;      // dot product
    Vector2d &normalize();                                  // normalize the vector in place
    Vector2d  normalized()                      const;      // normalized unit vector
    double    length()                          const;      // length of the vector from the origin
    double    lengthSquared()                   const;      // squared length of the vector from the origin
};

class Vector3i
{
public:
    int x, y, z;

public:
    Vector3i()                             : x(0),    y(0),    z(0)    {}
    Vector3i(int xpos, int ypos, int zpos) : x(xpos), y(ypos), z(zpos) {}
    Vector3i(const Vector2i &v, int zpos)  : x(v.x),  y(v.y),  z(zpos) {}
    Vector3i(const Vector3i &v)            : x(v.x),  y(v.y),  z(v.z)  {}
    Vector3i(const int *v, int offset = 1) : x(v[0]), y(v[offset]), z(v[offset * 2]) {}

    bool      operator != (const Vector3i &v) const;
    Vector3i  operator *  (int factor)        const;
    Vector3i  operator *  (const Vector3i &v) const;        // element-wise
    Vector3i &operator *= (int factor);
    Vector3i &operator *= (const Vector3i &v);              // element-wise
    Vector3i  operator +  (const Vector3i &v) const;
    Vector3i &operator += (const Vector3i &v);
    Vector3i  operator -  (const Vector3i &v) const;
    Vector3i  operator -  ()                  const;        // unary minus
    Vector3i &operator -= (const Vector3i &v);
    Vector3i  operator /  (int divisor)       const;        // integer division
    Vector3i  operator /  (const Vector3i &v) const;        // element-wise integer division
    Vector3i &operator /= (int divisor);                    // integer division
    Vector3i &operator /= (const Vector3i &v);              // element-wise integer division
    Vector3i &operator =  (const Vector3i &v);
    bool      operator == (const Vector3i &v) const;

    int       &operator [] (int index)       { return (&x)[index]; }
    const int &operator [] (int index) const { return (&x)[index]; }

    operator int * () { return &x; }                        // convert the vector into an array

    void      copyTo(int *v, int offset = 1) const;         // copy the elements to an array

    ////
    // max
    // min
};

class Vector3f
{
public:
    float x, y, z;

public:
    Vector3f()                                   : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3f(float xpos, float ypos, float zpos) : x(xpos), y(ypos), z(zpos) {}
    Vector3f(const Vector2f &v, float zpos)      : x(v.x),  y(v.y),  z(zpos) {}
    Vector3f(const Vector3f &v)                  : x(v.x),  y(v.y),  z(v.z)  {}
    Vector3f(const float *v, int offset = 1)     : x(v[0]), y(v[offset]), z(v[offset * 2]) {}

    // explicit?
    Vector3f(const Vector3i &v)                  : x((float)v.x), y((float)v.y), z((float)v.z) {}
    Vector3f(const Vector3d &v);

    bool      operator != (const Vector3f &v) const;
    Vector3f  operator *  (float factor)      const;
    Vector3f  operator *  (const Vector3f &v) const;        // element-wise
    Vector3f &operator *= (float factor);
    Vector3f &operator *= (const Vector3f &v);              // element-wise
    Vector3f  operator +  (const Vector3f &v) const;
    Vector3f &operator += (const Vector3f &v);
    Vector3f  operator -  (const Vector3f &v) const;
    Vector3f  operator -  ()                  const;        // unary minus
    Vector3f &operator -= (const Vector3f &v);
    Vector3f  operator /  (float divisor)     const;
    Vector3f  operator /  (const Vector3f &v) const;        // element-wise
    Vector3f &operator /= (float divisor);
    Vector3f &operator /= (const Vector3f &v);              // element-wise
    Vector3f &operator =  (const Vector3f &v);
    bool      operator == (const Vector3f &v) const;

    float       &operator [] (int index)       { return (&x)[index]; }
    const float &operator [] (int index) const { return (&x)[index]; }

    operator float * () { return &x; }                      // convert the vector into an array

    void      copyTo(float *v, int offset = 1) const;       // copy the elements to an array
    Vector3f  cross(const Vector3f &v)         const;       // cross product
    float     dot(const Vector3f &v)           const;       // dot product
    Vector3f &normalize();                                  // normalize the vector in place
    Vector3f  normalized()                     const;       // normalized unit vector
    float     length()                         const;       // length of the vector from the origin
    float     lengthSquared()                  const;       // squared length of the vector from the origin
};

//// float * Vector3

class Vector3d
{
public:
    double x, y, z;

public:
    Vector3d()                                      : x(0.0),  y(0.0),  z(0.0)  {}
    Vector3d(double xpos, double ypos, double zpos) : x(xpos), y(ypos), z(zpos) {}
    Vector3d(const Vector2d &v, double zpos)        : x(v.x),  y(v.y),  z(zpos) {}
    Vector3d(const Vector3d &v)                     : x(v.x),  y(v.y),  z(v.z)  {}
    Vector3d(const float *v, int offset = 1)        : x((double)v[0]), y((double)v[offset]), z((double)v[offset * 2]) {}
    Vector3d(const double *v, int offset = 1)       : x(        v[0]), y(        v[offset]), z(        v[offset * 2]) {}

    Vector3d(const Vector3f &v)                     : x((double)v.x),  y((double)v.y),       z((double)v.z)           {}

    bool      operator != (const Vector3d &v) const;
    Vector3d  operator *  (double factor)     const;
    Vector3d  operator *  (const Vector3d &v) const;        // element-wise
    Vector3d &operator *= (double factor);
    Vector3d &operator *= (const Vector3d &v);              // element-wise
    Vector3d  operator +  (const Vector3d &v) const;
    Vector3d &operator += (const Vector3d &v);
    Vector3d  operator -  (const Vector3d &v) const;
    Vector3d  operator -  ()                  const;        // unary minus
    Vector3d &operator -= (const Vector3d &v);
    Vector3d  operator /  (double divisor)    const;
    Vector3d  operator /  (const Vector3d &v) const;        // element-wise
    Vector3d &operator /= (double divisor);
    Vector3d &operator /= (const Vector3d &v);              // element-wise
    Vector3d &operator =  (const Vector3d &v);
    bool      operator == (const Vector3d &v) const;

    double       &operator [] (int index)       { return (&x)[index]; }
    const double &operator [] (int index) const { return (&x)[index]; }

    operator double * () { return &x; }                     // convert the vector into an array

    void      copyTo(double *v, int offset = 1) const;      // copy the elements to an array
    Vector3d  cross(const Vector3d &v)          const;      // cross product
    double    dot(const Vector3d &v)            const;      // dot product
    Vector3d &normalize();                                  // normalize the vector in place
    Vector3d  normalized()                      const;      // normalized unit vector
    double    length()                          const;      // length of the vector from the origin
    double    lengthSquared()                   const;      // squared length of the vector from the origin
};

class Vector4i
{
public:
    int x, y, z, w;

public:
    Vector4i()                                       : x(0),    y(0),    z(0),    w(0)    {}
    Vector4i(int xpos, int ypos, int zpos, int wpos) : x(xpos), y(ypos), z(zpos), w(wpos) {}
    Vector4i(const Vector2i &v, int zpos, int wpos)  : x(v.x),  y(v.y),  z(zpos), w(wpos) {}
    Vector4i(const Vector3i &v, int wpos)            : x(v.x),  y(v.y),  z(v.z),  w(wpos) {}
    Vector4i(const Vector4i &v)                      : x(v.x),  y(v.y),  z(v.z),  w(v.w)  {}
    Vector4i(const int *v, int offset = 1)           : x(v[0]), y(v[offset]), z(v[offset * 2]), w(v[offset * 3]) {}

    bool      operator != (const Vector4i &v) const;
    Vector4i  operator *  (int factor)        const;
    Vector4i  operator *  (const Vector4i &v) const;        // element-wise
    Vector4i &operator *= (int factor);
    Vector4i &operator *= (const Vector4i &v);              // element-wise
    Vector4i  operator +  (const Vector4i &v) const;
    Vector4i &operator += (const Vector4i &v);
    Vector4i  operator -  (const Vector4i &v) const;
    Vector4i  operator -  ()                  const;        // unary minus
    Vector4i &operator -= (const Vector4i &v);
    Vector4i  operator /  (int divisor)       const;        // integer division
    Vector4i  operator /  (const Vector4i &v) const;        // element-wise integer division
    Vector4i &operator /= (int divisor);                    // integer division
    Vector4i &operator /= (const Vector4i &v);              // element-wise integer division
    Vector4i &operator =  (const Vector4i &v);
    bool      operator == (const Vector4i &v) const;

    int       &operator [] (int index)       { return (&x)[index]; }
    const int &operator [] (int index) const { return (&x)[index]; }

    operator int * () { return &x; }                        // convert the vector into an array

    void      copyTo(int *v, int offset = 1) const;         // copy the elements to an array
};

class Vector4f
{
public:
    float x, y, z, w;

public:
    Vector4f()                                               : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    Vector4f(float xpos, float ypos, float zpos, float wpos) : x(xpos), y(ypos), z(zpos), w(wpos) {}
    Vector4f(const Vector3f &v, float wpos = 1.0f)           : x(v.x),  y(v.y),  z(v.z),  w(wpos) {}
    Vector4f(const Vector4f &v)                              : x(v.x),  y(v.y),  z(v.z),  w(v.w)  {}
    Vector4f(const float *v, int offset = 1)                 : x(       v[0]), y(       v[offset]), z(       v[offset * 2]), w(       v[offset * 3]) {}
    Vector4f(const double *v, int offset = 1)                : x((float)v[0]), y((float)v[offset]), z((float)v[offset * 2]), w((float)v[offset * 3]) {}

    bool      operator != (const Vector4f &v) const;
    Vector4f  operator *  (float factor)      const;
    Vector4f  operator *  (const Vector4f &v) const;        // element-wise
    Vector4f &operator *= (float factor);
    Vector4f &operator *= (const Vector4f &v);              // element-wise
    Vector4f  operator +  (const Vector4f &v) const;
    Vector4f &operator += (const Vector4f &v);
    Vector4f  operator -  (const Vector4f &v) const;
    Vector4f  operator -  ()                  const;        // unary minus
    Vector4f &operator -= (const Vector4f &v);
    Vector4f  operator /  (float divisor)     const;
    Vector4f  operator /  (const Vector4f &v) const;        // element-wise
    Vector4f &operator /= (float divisor);
    Vector4f &operator /= (const Vector4f &v);              // element-wise
    Vector4f &operator =  (const Vector4f &v);
    bool      operator == (const Vector4f &v) const;

    float       &operator [] (int index)       { return (&x)[index]; }
    const float &operator [] (int index) const { return (&x)[index]; }

    operator float * () { return &x; }                      // convert the vector into an array

    void      copyTo(float *v, int offset = 1) const;       // copy the elements to an array
    float     dot(const Vector4f &v)           const;       // dot product
    Vector4f &normalize();                                  // normalize the vector in place
    Vector4f  normalized()                     const;       // normalized unit vector
    float     length()                         const;       // length of the vector from the origin
    float     lengthSquared()                  const;       // squared length of the vector from the origin
};

class Vector4d
{
public:
    double x, y, z, w;

public:
    Vector4d()                                                   : x(0.0),  y(0.0),  z(0.0),  w(0.0)  {}
    Vector4d(double xpos, double ypos, double zpos, double wpos) : x(xpos), y(ypos), z(zpos), w(wpos) {}
    Vector4d(const Vector3d &v, double wpos = 1.0)               : x(v.x),  y(v.y),  z(v.z),  w(wpos) {}
    Vector4d(const Vector4d &v)                                  : x(v.x),  y(v.y),  z(v.z),  w(v.w)  {}
    Vector4d(const float *v, int offset = 1)                     : x((double)v[0]), y((double)v[offset]), z((double)v[offset * 2]), w((double)v[offset * 3]) {}
    Vector4d(const double *v, int offset = 1)                    : x(        v[0]), y(        v[offset]), z(        v[offset * 2]), w(        v[offset * 3]) {}

    bool      operator != (const Vector4d &v) const;
    Vector4d  operator *  (double factor)     const;
    Vector4d  operator *  (const Vector4d &v) const;        // element-wise
    Vector4d &operator *= (double factor);
    Vector4d &operator *= (const Vector4d &v);              // element-wise
    Vector4d  operator +  (const Vector4d &v) const;
    Vector4d &operator += (const Vector4d &v);
    Vector4d  operator -  (const Vector4d &v) const;
    Vector4d  operator -  ()                  const;        // unary minus
    Vector4d &operator -= (const Vector4d &v);
    Vector4d  operator /  (double divisor)    const;
    Vector4d  operator /  (const Vector4d &v) const;        // element-wise
    Vector4d &operator /= (double divisor);
    Vector4d &operator /= (const Vector4d &v);              // element-wise
    Vector4d &operator =  (const Vector4d &v);
    bool      operator == (const Vector4d &v) const;

    double       &operator [] (int index)       { return (&x)[index]; }
    const double &operator [] (int index) const { return (&x)[index]; }

    operator double * () { return &x; }                     // convert the vector into an array

    void      copyTo(double *v, int offset = 1) const;      // copy the elements to an array
    double    dot(const Vector4d &v)            const;      // dot product
    Vector4d &normalize();                                  // normalize the vector in place
    Vector4d  normalized()                      const;      // normalized unit vector
    double    length()                          const;      // length of the vector from the origin
    double    lengthSquared()                   const;      // squared length of the vector from the origin
};

////////////////////////////////////////////////////////////////////////////////
//  Vector2i                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline bool Vector2i::operator != (const Vector2i &v) const
{
    return !(*this == v);
}

inline Vector2i Vector2i::operator * (int factor) const
{
    return Vector2i(x * factor, y * factor);
}

inline Vector2i Vector2i::operator * (const Vector2i &v) const
{
    return Vector2i(x * v.x, y * v.y);
}

inline Vector2i &Vector2i::operator *= (int factor)
{
    x *= factor;
    y *= factor;
    return *this;
}

inline Vector2i &Vector2i::operator *= (const Vector2i &v)
{
    x *= v.x;
    y *= v.y;
    return *this;
}

inline Vector2i Vector2i::operator + (const Vector2i &v) const
{
    return Vector2i(x + v.x, y + v.y);
}

inline Vector2i &Vector2i::operator += (const Vector2i &v)
{
    x += v.x;
    y += v.y;
    return *this;
}

inline Vector2i Vector2i::operator - (const Vector2i &v) const
{
    return Vector2i(x - v.x, y - v.y);
}

inline Vector2i Vector2i::operator - () const
{
    return Vector2i(-x, -y);
}

inline Vector2i &Vector2i::operator -= (const Vector2i &v)
{
    x -= v.x;
    y -= v.y;
    return *this;
}

inline Vector2i Vector2i::operator / (int divisor) const
{
    return Vector2i(x / divisor, y / divisor);
}

inline Vector2i Vector2i::operator / (const Vector2i &v) const
{
    return Vector2i(x / v.x, y / v.y);
}

inline Vector2i &Vector2i::operator /= (int divisor)
{
    x /= divisor;
    y /= divisor;
    return *this;
}

inline Vector2i &Vector2i::operator /= (const Vector2i &v)
{
    x /= v.x;
    y /= v.y;
    return *this;
}

inline Vector2i &Vector2i::operator = (const Vector2i &v)
{
    x = v.x;
    y = v.y;
    return *this;
}

inline bool Vector2i::operator == (const Vector2i &v) const
{
    return (x == v.x && y == v.y);
}

inline void Vector2i::copyTo(int *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
}

////////////////////////////////////////////////////////////////////////////////
//  Vector2f                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline Vector2f::Vector2f(const Vector2d &v)
    : x((float)v.x),
      y((float)v.y)
{
}

inline bool Vector2f::operator != (const Vector2f &v) const
{
    return !(*this == v);
}

inline Vector2f Vector2f::operator * (float factor) const
{
    return Vector2f(x * factor, y * factor);
}

// element-wise multiplication
inline Vector2f Vector2f::operator * (const Vector2f &v) const
{
    return Vector2f(x * v.x, y * v.y);
}

inline Vector2f &Vector2f::operator *= (float factor)
{
    x *= factor;
    y *= factor;
    return *this;
}

// element-wise multiplication
inline Vector2f &Vector2f::operator *= (const Vector2f &v)
{
    x *= v.x;
    y *= v.y;
    return *this;
}

inline Vector2f Vector2f::operator + (const Vector2f &v) const
{
    return Vector2f(x + v.x, y + v.y);
}

inline Vector2f &Vector2f::operator += (const Vector2f &v)
{
    x += v.x;
    y += v.y;
    return *this;
}

inline Vector2f Vector2f::operator - (const Vector2f &v) const
{
    return Vector2f(x - v.x, y - v.y);
}

inline Vector2f Vector2f::operator - () const
{
    return Vector2f(-x, -y);
}

inline Vector2f &Vector2f::operator -= (const Vector2f &v)
{
    x -= v.x;
    y -= v.y;
    return *this;
}

inline Vector2f Vector2f::operator / (float divisor) const
{
    return Vector2f(x / divisor, y / divisor);
}

// element-wise division
inline Vector2f Vector2f::operator / (const Vector2f &v) const
{
    return Vector2f(x / v.x, y / v.y);
}

inline Vector2f &Vector2f::operator /= (float divisor)
{
    x /= divisor;
    y /= divisor;
    return *this;
}

// element-wise division
inline Vector2f &Vector2f::operator /= (const Vector2f &v)
{
    x /= v.x;
    y /= v.y;
    return *this;
}

inline Vector2f &Vector2f::operator = (const Vector2f &v)
{
    x = v.x;
    y = v.y;
    return *this;
}

inline bool Vector2f::operator == (const Vector2f &v) const
{
    return (x == v.x && y == v.y);
}

inline void Vector2f::copyTo(float *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
}

// dot product
inline float Vector2f::dot(const Vector2f &v) const
{
    return (x * v.x + y * v.y);
}

inline Vector2f &Vector2f::normalize()
{
    float invLength = 1.0f / length();
    x *= invLength;
    y *= invLength;
    return *this;
}

inline Vector2f Vector2f::normalized() const
{
    float invLength = 1.0f / length();
    return Vector2f(x * invLength, y * invLength);
}

inline float Vector2f::length() const
{
    return sqrt(lengthSquared());
}

inline float Vector2f::lengthSquared() const
{
    return (x * x + y * y);
}

////////////////////////////////////////////////////////////////////////////////
//  Vector2d                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline Vector2d::Vector2d(const Vector2f &v)
    : x((double)v.x),
      y((double)v.y)
{
}

inline bool Vector2d::operator != (const Vector2d &v) const
{
    return !(*this == v);
}

inline Vector2d Vector2d::operator * (double factor) const
{
    return Vector2d(x * factor, y * factor);
}

inline Vector2d Vector2d::operator * (const Vector2d &v) const
{
    return Vector2d(x * v.x, y * v.y);
}

inline Vector2d &Vector2d::operator *= (double factor)
{
    x *= factor;
    y *= factor;
    return *this;
}

inline Vector2d &Vector2d::operator *= (const Vector2d &v)
{
    x *= v.x;
    y *= v.y;
    return *this;
}

inline Vector2d Vector2d::operator + (const Vector2d &v) const
{
    return Vector2d(x + v.x, y + v.y);
}

inline Vector2d &Vector2d::operator += (const Vector2d &v)
{
    x += v.x;
    y += v.y;
    return *this;
}

inline Vector2d Vector2d::operator - (const Vector2d &v) const
{
    return Vector2d(x - v.x, y - v.y);
}

inline Vector2d Vector2d::operator - () const
{
    return Vector2d(-x, -y);
}

inline Vector2d &Vector2d::operator -= (const Vector2d &v)
{
    x -= v.x;
    y -= v.y;
    return *this;
}

inline Vector2d Vector2d::operator / (double divisor) const
{
    return Vector2d(x / divisor, y / divisor);
}

inline Vector2d Vector2d::operator / (const Vector2d &v) const
{
    return Vector2d(x / v.x, y / v.y);
}

inline Vector2d &Vector2d::operator /= (double divisor)
{
    x /= divisor;
    y /= divisor;
    return *this;
}

inline Vector2d &Vector2d::operator /= (const Vector2d &v)
{
    x /= v.x;
    y /= v.y;
    return *this;
}

inline Vector2d &Vector2d::operator = (const Vector2d &v)
{
    x = v.x;
    y = v.y;
    return *this;
}

inline bool Vector2d::operator == (const Vector2d &v) const
{
    return (x == v.x && y == v.y);
}

inline void Vector2d::copyTo(double *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
}

inline double Vector2d::dot(const Vector2d &v) const
{
    return (x * v.x + y * v.y);
}

inline Vector2d &Vector2d::normalize()
{
    double invLength = 1.0 / length();
    x *= invLength;
    y *= invLength;
    return *this;
}

inline Vector2d Vector2d::normalized() const
{
    double invLength = 1.0 / length();
    return Vector2d(x * invLength, y * invLength);
}

inline double Vector2d::length() const
{
    return sqrt(lengthSquared());
}

inline double Vector2d::lengthSquared() const
{
    return (x * x + y * y);
}

////////////////////////////////////////////////////////////////////////////////
//  Vector3i                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline bool Vector3i::operator != (const Vector3i &v) const
{
    return !(*this == v);
}

inline Vector3i Vector3i::operator * (int factor) const
{
    return Vector3i(x * factor, y * factor, z * factor);
}

inline Vector3i Vector3i::operator * (const Vector3i &v) const
{
    return Vector3i(x * v.x, y * v.y, z * v.z);
}

inline Vector3i &Vector3i::operator *= (int factor)
{
    x *= factor;
    y *= factor;
    z *= factor;
    return *this;
}

inline Vector3i &Vector3i::operator *= (const Vector3i &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

inline Vector3i Vector3i::operator + (const Vector3i &v) const
{
    return Vector3i(x + v.x, y + v.y, z + v.z);
}

inline Vector3i &Vector3i::operator += (const Vector3i &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

inline Vector3i Vector3i::operator - (const Vector3i &v) const
{
    return Vector3i(x - v.x, y - v.y, z - v.z);
}

inline Vector3i Vector3i::operator - () const
{
    return Vector3i(-x, -y, -z);
}

inline Vector3i &Vector3i::operator -= (const Vector3i &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

inline Vector3i Vector3i::operator / (int divisor) const
{
    return Vector3i(x / divisor, y / divisor, z / divisor);
}

inline Vector3i Vector3i::operator / (const Vector3i &v) const
{
    return Vector3i(x / v.x, y / v.y, z / v.z);
}

inline Vector3i &Vector3i::operator /= (int divisor)
{
    x /= divisor;
    y /= divisor;
    z /= divisor;
    return *this;
}

inline Vector3i &Vector3i::operator /= (const Vector3i &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

inline Vector3i &Vector3i::operator = (const Vector3i &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
}

inline bool Vector3i::operator == (const Vector3i &v) const
{
    return (x == v.x && y == v.y && z == v.z);
}

inline void Vector3i::copyTo(int *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
    v[offset * 2] = z;
}

////////////////////////////////////////////////////////////////////////////////
//  Vector3f                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline Vector3f::Vector3f(const Vector3d &v)
    : x((float)v.x),
      y((float)v.y),
      z((float)v.z)
{
}

inline bool Vector3f::operator != (const Vector3f &v) const
{
    return !(*this == v);
}

inline Vector3f Vector3f::operator * (float factor) const
{
    return Vector3f(x * factor, y * factor, z * factor);
}

// element-wise multiplication
inline Vector3f Vector3f::operator * (const Vector3f &v) const
{
    return Vector3f(x * v.x, y * v.y, z * v.z);
}

inline Vector3f &Vector3f::operator *= (float factor)
{
    x *= factor;
    y *= factor;
    z *= factor;
    return *this;
}

// element-wise multiplication
inline Vector3f &Vector3f::operator *= (const Vector3f &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

inline Vector3f Vector3f::operator + (const Vector3f &v) const
{
    return Vector3f(x + v.x, y + v.y, z + v.z);
}

inline Vector3f &Vector3f::operator += (const Vector3f &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

inline Vector3f Vector3f::operator - (const Vector3f &v) const
{
    return Vector3f(x - v.x, y - v.y, z - v.z);
}

inline Vector3f Vector3f::operator - () const
{
    return Vector3f(-x, -y, -z);
}

inline Vector3f &Vector3f::operator -= (const Vector3f &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

inline Vector3f Vector3f::operator / (float divisor) const
{
    return Vector3f(x / divisor, y / divisor, z / divisor);
}

// element-wise division
inline Vector3f Vector3f::operator / (const Vector3f &v) const
{
    return Vector3f(x / v.x, y / v.y, z / v.z);
}

inline Vector3f &Vector3f::operator /= (float divisor)
{
    x /= divisor;
    y /= divisor;
    z /= divisor;
    return *this;
}

// element-wise division
inline Vector3f &Vector3f::operator /= (const Vector3f &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

inline Vector3f &Vector3f::operator = (const Vector3f &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
}

inline bool Vector3f::operator == (const Vector3f &v) const
{
    return (x == v.x && y == v.y && z == v.z);
}

/*inline float &Vector3f::operator [] (int index)
{
    return (&x)[index];
}*/

/*inline const float &Vector3f::operator [] (int index) const
{
    return (&x)[index];
}*/

/*inline Vector3f::operator float * ()
{
    return &x;
}*/

inline void Vector3f::copyTo(float *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
    v[offset * 2] = z;
}

// cross product
inline Vector3f Vector3f::cross(const Vector3f &v) const
{
    return Vector3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}

// dot product
inline float Vector3f::dot(const Vector3f &v) const
{
    return (x * v.x + y * v.y + z * v.z);
}

inline Vector3f &Vector3f::normalize()
{
    float invLength = 1.0f / length();
    x *= invLength;
    y *= invLength;
    z *= invLength;
    return *this;
}

inline Vector3f Vector3f::normalized() const
{
    float invLength = 1.0f / length();
    return Vector3f(x * invLength, y * invLength, z * invLength);
}

inline float Vector3f::length() const
{
    return sqrt(lengthSquared());
}

inline float Vector3f::lengthSquared() const
{
    return (x * x + y * y + z * z);
}

////////////////////////////////////////////////////////////////////////////////
//  Vector3d                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline bool Vector3d::operator != (const Vector3d &v) const
{
    return !(*this == v);
}

inline Vector3d Vector3d::operator * (double factor) const
{
    return Vector3d(x * factor, y * factor, z * factor);
}

inline Vector3d Vector3d::operator * (const Vector3d &v) const
{
    return Vector3d(x * v.x, y * v.y, z * v.z);
}

inline Vector3d &Vector3d::operator *= (double factor)
{
    x *= factor;
    y *= factor;
    z *= factor;
    return *this;
}

inline Vector3d &Vector3d::operator *= (const Vector3d &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

inline Vector3d Vector3d::operator + (const Vector3d &v) const
{
    return Vector3d(x + v.x, y + v.y, z + v.z);
}

inline Vector3d &Vector3d::operator += (const Vector3d &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

inline Vector3d Vector3d::operator - (const Vector3d &v) const
{
    return Vector3d(x - v.x, y - v.y, z - v.z);
}

inline Vector3d Vector3d::operator - () const
{
    return Vector3d(-x, -y, -z);
}

inline Vector3d &Vector3d::operator -= (const Vector3d &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

inline Vector3d Vector3d::operator / (double divisor) const
{
    return Vector3d(x / divisor, y / divisor, z / divisor);
}

inline Vector3d Vector3d::operator / (const Vector3d &v) const
{
    return Vector3d(x / v.x, y / v.y, z / v.z);
}

inline Vector3d &Vector3d::operator /= (double divisor)
{
    x /= divisor;
    y /= divisor;
    z /= divisor;
    return *this;
}

inline Vector3d &Vector3d::operator /= (const Vector3d &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

inline Vector3d &Vector3d::operator = (const Vector3d &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
}

inline bool Vector3d::operator == (const Vector3d &v) const
{
    return (x == v.x && y == v.y && z == v.z);
}

inline void Vector3d::copyTo(double *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
    v[offset * 2] = z;
}

inline Vector3d Vector3d::cross(const Vector3d &v) const
{
    return Vector3d(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}

inline double Vector3d::dot(const Vector3d &v) const
{
    return (x * v.x + y * v.y + z * v.z);
}

inline Vector3d &Vector3d::normalize()
{
    double invLength = 1.0 / length();
    x *= invLength;
    y *= invLength;
    z *= invLength;
    return *this;
}

inline Vector3d Vector3d::normalized() const
{
    double invLength = 1.0 / length();
    return Vector3d(x * invLength, y * invLength, z * invLength);
}

inline double Vector3d::length() const
{
    return sqrt(lengthSquared());
}

inline double Vector3d::lengthSquared() const
{
    return (x * x + y * y + z * z);
}

////////////////////////////////////////////////////////////////////////////////
//  Vector4i                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline bool Vector4i::operator != (const Vector4i &v) const
{
    return !(*this == v);
}

inline Vector4i Vector4i::operator * (int factor) const
{
    return Vector4i(x * factor, y * factor, z * factor, w * factor);
}

inline Vector4i Vector4i::operator * (const Vector4i &v) const
{
    return Vector4i(x * v.x, y * v.y, z * v.z, w * v.w);
}

inline Vector4i &Vector4i::operator *= (int factor)
{
    x *= factor;
    y *= factor;
    z *= factor;
    w *= factor;
    return *this;
}

inline Vector4i &Vector4i::operator *= (const Vector4i &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;
    return *this;
}

inline Vector4i Vector4i::operator + (const Vector4i &v) const
{
    return Vector4i(x + v.x, y + v.y, z + v.z, w + v.w);
}

inline Vector4i &Vector4i::operator += (const Vector4i &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
}

inline Vector4i Vector4i::operator - (const Vector4i &v) const
{
    return Vector4i(x - v.x, y - v.y, z - v.z, w - v.w);
}

inline Vector4i Vector4i::operator - () const
{
    return Vector4i(-x, -y, -z, -w);
}

inline Vector4i &Vector4i::operator -= (const Vector4i &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
}

inline Vector4i Vector4i::operator / (int divisor) const
{
    return Vector4i(x / divisor, y / divisor, z / divisor, w / divisor);
}

inline Vector4i Vector4i::operator / (const Vector4i &v) const
{
    return Vector4i(x / v.x, y / v.y, z / v.z, w / v.w);
}

inline Vector4i &Vector4i::operator /= (int divisor)
{
    x /= divisor;
    y /= divisor;
    z /= divisor;
    w /= divisor;
    return *this;
}

inline Vector4i &Vector4i::operator /= (const Vector4i &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;
    return *this;
}

inline Vector4i &Vector4i::operator = (const Vector4i &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    w = v.w;
    return *this;
}

inline bool Vector4i::operator == (const Vector4i &v) const
{
    return (x == v.x && y == v.y && z == v.z && w == v.w);
}

inline void Vector4i::copyTo(int *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
    v[offset * 2] = z;
    v[offset * 3] = w;
}

////////////////////////////////////////////////////////////////////////////////
//  Vector4f                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline bool Vector4f::operator != (const Vector4f &v) const
{
    return !(*this == v);
}

inline Vector4f Vector4f::operator * (float factor) const
{
    return Vector4f(x * factor, y * factor, z * factor, w * factor);
}

inline Vector4f Vector4f::operator * (const Vector4f &v) const
{
    return Vector4f(x * v.x, y * v.y, z * v.z, w * v.w);
}

inline Vector4f &Vector4f::operator *= (float factor)
{
    x *= factor;
    y *= factor;
    z *= factor;
    w *= factor;
    return *this;
}

inline Vector4f &Vector4f::operator *= (const Vector4f &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;
    return *this;
}

inline Vector4f Vector4f::operator + (const Vector4f &v) const
{
    return Vector4f(x + v.x, y + v.y, z + v.z, w + v.w);
}

inline Vector4f &Vector4f::operator += (const Vector4f &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
}

inline Vector4f Vector4f::operator - (const Vector4f &v) const
{
    return Vector4f(x - v.x, y - v.y, z - v.z, w - v.w);
}

inline Vector4f Vector4f::operator - () const
{
    return Vector4f(-x, -y, -z, -w);
}

inline Vector4f &Vector4f::operator -= (const Vector4f &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
}

inline Vector4f Vector4f::operator / (float divisor) const
{
    return Vector4f(x / divisor, y / divisor, z / divisor, w / divisor);
}

inline Vector4f Vector4f::operator / (const Vector4f &v) const
{
    return Vector4f(x / v.x, y / v.y, z / v.z, w / v.w);
}

inline Vector4f &Vector4f::operator /= (float divisor)
{
    x /= divisor;
    y /= divisor;
    z /= divisor;
    w /= divisor;
    return *this;
}

inline Vector4f &Vector4f::operator /= (const Vector4f &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;
    return *this;
}

inline Vector4f &Vector4f::operator = (const Vector4f &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    w = v.w;
    return *this;
}

inline bool Vector4f::operator == (const Vector4f &v) const
{
    return (x == v.x && y == v.y && z == v.z && w == v.w);
}

inline void Vector4f::copyTo(float *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
    v[offset * 2] = z;
    v[offset * 3] = w;
}

inline float Vector4f::dot(const Vector4f &v) const
{
    return (x * v.x + y * v.y + z * v.z + w * v.w);
}

inline Vector4f &Vector4f::normalize()
{
    float invLength = 1.0f / length();
    x *= invLength;
    y *= invLength;
    z *= invLength;
    w *= invLength;
    return *this;
}

inline Vector4f Vector4f::normalized() const
{
    float invLength = 1.0f / length();
    return Vector4f(x * invLength, y * invLength, z * invLength, w * invLength);
}

inline float Vector4f::length() const
{
    return sqrt(lengthSquared());
}

inline float Vector4f::lengthSquared() const
{
    return (x * x + y * y + z * z + w * w);
}

////////////////////////////////////////////////////////////////////////////////
//  Vector4d                                                                  //
////////////////////////////////////////////////////////////////////////////////

inline bool Vector4d::operator != (const Vector4d &v) const
{
    return !(*this == v);
}

inline Vector4d Vector4d::operator * (double factor) const
{
    return Vector4d(x * factor, y * factor, z * factor, w * factor);
}

inline Vector4d Vector4d::operator * (const Vector4d &v) const
{
    return Vector4d(x * v.x, y * v.y, z * v.z, w * v.w);
}

inline Vector4d &Vector4d::operator *= (double factor)
{
    x *= factor;
    y *= factor;
    z *= factor;
    w *= factor;
    return *this;
}

inline Vector4d &Vector4d::operator *= (const Vector4d &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;
    return *this;
}

inline Vector4d Vector4d::operator + (const Vector4d &v) const
{
    return Vector4d(x + v.x, y + v.y, z + v.z, w + v.w);
}

inline Vector4d &Vector4d::operator += (const Vector4d &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
}

inline Vector4d Vector4d::operator - (const Vector4d &v) const
{
    return Vector4d(x - v.x, y - v.y, z - v.z, w - v.w);
}

inline Vector4d Vector4d::operator - () const
{
    return Vector4d(-x, -y, -z, -w);
}

inline Vector4d &Vector4d::operator -= (const Vector4d &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
}

inline Vector4d Vector4d::operator / (double divisor) const
{
    return Vector4d(x / divisor, y / divisor, z / divisor, w / divisor);
}

inline Vector4d Vector4d::operator / (const Vector4d &v) const
{
    return Vector4d(x / v.x, y / v.y, z / v.z, w / v.w);
}

inline Vector4d &Vector4d::operator /= (double divisor)
{
    x /= divisor;
    y /= divisor;
    z /= divisor;
    w /= divisor;
    return *this;
}

inline Vector4d &Vector4d::operator /= (const Vector4d &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;
    return *this;
}

inline Vector4d &Vector4d::operator = (const Vector4d &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    w = v.w;
    return *this;
}

inline bool Vector4d::operator == (const Vector4d &v) const
{
    return (x == v.x && y == v.y && z == v.z && w == v.w);
}

inline void Vector4d::copyTo(double *v, int offset) const
{
    v[0] = x;
    v[offset] = y;
    v[offset * 2] = z;
    v[offset * 3] = w;
}

inline double Vector4d::dot(const Vector4d &v) const
{
    return (x * v.x + y * v.y + z * v.z + w * v.w);
}

inline Vector4d &Vector4d::normalize()
{
    double invLength = 1.0 / length();
    x *= invLength;
    y *= invLength;
    z *= invLength;
    w *= invLength;
    return *this;
}

inline Vector4d Vector4d::normalized() const
{
    double invLength = 1.0 / length();
    return Vector4d(x * invLength, y * invLength, z * invLength, w * invLength);
}

inline double Vector4d::length() const
{
    return sqrt(lengthSquared());
}

inline double Vector4d::lengthSquared() const
{
    return (x * x + y * y + z * z + w * w);
}

#endif // VECTORS_H
