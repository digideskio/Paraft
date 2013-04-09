#ifndef _VECTORS_H_
#define _VECTORS_H_


#ifdef QT_CORE_LIB
#include <QIODevice>
#include <QTextStream>
#endif

struct Matrix4x4;
struct Vector4;

#define unpack2(v) (v).x(), (v).y()
#define unpack3(v) (v).x(), (v).y(), (v).z()
#define unpack4(v) (v).x(), (v).y(), (v).z(), (v).d()

struct Vector2  {
	double elements[2];
	Vector2() {
		elements[0] = 0;
		elements[1] = 0;
	}
	Vector2(const double& x, const double &y) {
		elements[0] = x;
		elements[1] = y;
	}
	Vector2(const Vector2 &other) {
		elements[0] = other.elements[0];
		elements[1] = other.elements[1];
	}
	
	double& x() { return elements[0]; }
	double& y() { return elements[1]; }
	double x() const { return elements[0]; }
	double y() const { return elements[1]; }
	
	Vector2 operator+(const Vector2& rhs) const;
	Vector2& operator+=(const Vector2& rhs);
	
	Vector2 operator-(const Vector2& rhs) const;
	Vector2& operator-=(const Vector2& rhs);
	
	Vector2 operator*(const double& rhs) const;
	Vector2 operator/(const double& rhs) const;
	
	Vector2& operator*=(const double& rhs);
	Vector2& operator/=(const double& rhs);
	
	Vector2 operator-() const {
		return Vector2(-elements[0], -elements[1]);
	}

	bool operator==(const Vector2& other) const {
		return (this->x() == other.x()) &&
				(this->y() == other.y());
	}
	
	Vector2& operator=(const Vector2& other) {
		elements[0] = other.elements[0];
		elements[1] = other.elements[1];
		
		return *this;
	}
	
	Vector2& normalize();
	
	double dot(const Vector2& rhs) const;
	double& operator[](const int& index);
	operator double*() {
		return elements;
	}
	
	double lengthSquared() const {
		return elements[0]*elements[0] + elements[1]*elements[1];
	}
	
#ifdef QT_CORE_LIB
	friend QDataStream& operator<<(QDataStream& lhs, const Vector2& rhs);
	friend QDataStream& operator>>(QDataStream& lhs, Vector2& rhs);
	void save(QIODevice& io);
	void load(QIODevice& io);
#endif
	
	double length() const ;

	Vector2 mul(const Vector2& other) const {
		return Vector2(elements[0]*other.elements[0], elements[1]*other.elements[1]);
	}

	void set(float* p, int num=2) const;
	void set(double* p, int num=2) const;
	void setValues(double v1, double v2);
};



struct Vector3 {
	double elements[3];
	Vector3() {
		elements[0] = 0;
		elements[1] = 0;
		elements[2] = 0;
	}
	Vector3(const double &x, const double &y, const double &z) {
		elements[0] = x;
		elements[1] = y;
		elements[2] = z;
	}
	Vector3(const Vector2& other, double z=1) {
		elements[0] = other.elements[0];
		elements[1] = other.elements[1];
		elements[2] = z;
	}
	Vector3(const double* v, int offset=1) {
		elements[0] = v[0];
		elements[1] = v[offset];
		elements[2] = v[2*offset];
	}
	Vector3(const Vector3& other) {
		*this = other;
	}
	
	double& x() { return elements[0]; }
	double& y() { return elements[1]; }
	double& z() { return elements[2]; }
	double x() const { return elements[0]; }
	double y() const { return elements[1]; }
	double z() const { return elements[2]; }
	
	Vector3 operator+(const Vector3& rhs) const;
	Vector3& operator+=(const Vector3& rhs);
	
	Vector3 operator-(const Vector3& rhs) const;
	Vector3& operator-=(const Vector3& rhs);
	
	Vector3 operator*(const Vector3& rhs) const; //cross
	Vector3 operator*(const double& rhs) const;
	
	Vector3& operator*=(const Vector3& rhs); //cross
	Vector3& operator*=(const double& rhs);
	
	Vector3 operator/(const double& rhs) const;
	Vector3& operator/=(const double& rhs);
	
	Vector3& operator=(const Vector3 &other) {
		elements[0] = other.elements[0];
		elements[1] = other.elements[1];
		elements[2] = other.elements[2];
		
		return *this;
	}
	bool operator==(const Vector3& other) const {
		return (this->x() == other.x()) &&
				(this->y() == other.y()) &&
				(this->z() == other.z());
	}
	Vector3& normalize();

	Vector3 reflect(const Vector3& other) const;
	
	double dot(const Vector3& rhs) const;
	const double& operator[](const int& index) const;
	double& operator[](const int& index);
	operator double*() {
		return elements;
	}
	operator Vector2() const {
		return Vector2(elements[0], elements[1]);
	}
	
#ifdef QT_CORE_LIB
	void save(QIODevice& io);
	void load(QIODevice& io);
#endif
	
	Vector3 operator*(const Matrix4x4& rhs) const;
	Vector3& operator*=(const Matrix4x4& rhs);
	double lengthSquared() const {
		return elements[0]*elements[0] + elements[1]*elements[1] + elements[2]*elements[2];
	}
	double length() const;
	
	Vector3 toInts() const { return Vector3((int)elements[0], (int)elements[1], (int)elements[2]); }
	
	Vector3 operator-() const {
		return Vector3(-elements[0], -elements[1], -elements[2]);
	}
	Vector3 mul(const Vector3& other) const {
		return Vector3(elements[0]*other.elements[0], elements[1]*other.elements[1], elements[2]*other.elements[2]);
	}
	
#ifdef QT_CORE_LIB
	friend QTextStream& operator>>(QTextStream& lhs, Vector3& rhs);
	friend QTextStream& operator<<(QTextStream& lhs, const Vector3& rhs);

	friend QDataStream& operator<<(QDataStream& lhs, const Vector3& rhs);
	friend QDataStream& operator>>(QDataStream& lhs, Vector3& rhs);
#endif
	friend Vector3 operator*(double lhs, const Vector3& rhs);
	
#ifdef QT_CORE_LIB
	void writeFloats(QIODevice* d) const;
#endif
	const static Vector3 xAxis;
	const static Vector3 yAxis;
	const static Vector3 zAxis;
	const static Vector3 Zero;

	void set(float* p, int num=3) const;
	void set(double* p, int num=3) const;
	Vector3 refract(const Vector3& normal, double eta) const;
	void setValues(double v1, double v2, double v3);

	Vector3(const Vector4& v);
};

struct Vector4 {
	double elements[4];
	Vector4() {
		elements[0] = 0;
		elements[1] = 0;
		elements[2] = 0;
		elements[3] = 0;
	}
	Vector4(const double &x, const double &y, const double &z, const double &d) {
		elements[0] = x;
		elements[1] = y;
		elements[2] = z;
		elements[3] = d;
	}
	Vector4(const Vector2& other, double z=0, double d=0) {
		elements[0] = other.elements[0];
		elements[1] = other.elements[1];
		elements[2] = z;
		elements[3] = d;
	}
	Vector4(const Vector3& other, double d=1) {
		elements[0] = other.elements[0];
		elements[1] = other.elements[1];
		elements[2] = other.elements[2];
		elements[3] = d;
	}
	Vector4(const Vector4& other) {
		*this = other;
	}
	
	Vector4(const double* v, int stride=1) {
		elements[0] = v[0];
		elements[1] = v[stride];
		elements[2] = v[stride*2];
		elements[3] = v[stride*3];
	}
	
	double& x() { return elements[0]; }
	double& y() { return elements[1]; }
	double& z() { return elements[2]; }
	double& d() { return elements[3]; }

	double x() const { return elements[0]; }
	double y() const { return elements[1]; }
	double z() const { return elements[2]; }
	double d() const { return elements[3]; }

	Vector4 operator+(const Vector4& rhs) const;
	Vector4& operator+=(const Vector4& rhs);
	
	Vector4 operator-(const Vector4& rhs) const;
	Vector4& operator-=(const Vector4& rhs);
	
	Vector4 operator*(const double& rhs) const;
	Vector4 operator*(const Matrix4x4& rhs) const;
	Vector4& operator*=(const Matrix4x4& rhs);
	
	Vector4& operator*=(const double& rhs);
	
	Vector4 operator/(const double& rhs) const;
	Vector4& operator/=(const double& rhs);
	
	Vector4& operator=(const Vector4 &other) {
		elements[0] = other.elements[0];
		elements[1] = other.elements[1];
		elements[2] = other.elements[2];
		elements[3] = other.elements[3];
		
		return *this;
	}

	bool operator==(const Vector4& other) const {
		return (this->x() == other.x()) &&
				(this->y() == other.y()) &&
				(this->z() == other.z()) &&
				(this->d() == other.d());
	}
	Vector4& normalize();
	Vector4& homogenize();
	
	double dot(const Vector4& rhs) const;
	double& operator[](const int& index);
	operator double*() {
		return elements;
	}
	
	operator Vector3() const {
		return Vector3(elements[0], elements[1], elements[2]);
	}
	
#ifdef QT_CORE_LIB
	friend QDataStream& operator<<(QDataStream& lhs, const Vector4& rhs);
	friend QDataStream& operator>>(QDataStream& lhs, Vector4& rhs);
	friend QTextStream& operator>>(QTextStream& lhs, Vector4& rhs);
	friend QTextStream& operator<<(QTextStream& lhs, const Vector4& rhs);

	void save(QIODevice& io);
	void load(QIODevice& io);
#endif
	double lengthSquared() const {
		return elements[0]*elements[0] + elements[1]*elements[1] + elements[2]*elements[2] + elements[3]*elements[3];
	}
	double length() const;	
	
	Vector4 operator-() const {
		return Vector4(-elements[0], -elements[1], -elements[2], -elements[3]);
	}

	Vector4 mul(const Vector4& other) const {
		return Vector4(elements[0]*other.elements[0], elements[1]*other.elements[1], elements[2]*other.elements[2], elements[3]*other.elements[3]);
	}

	static Vector4 CatmullRom(double t) {
		return Vector4(1, t, t*t, t*t*t);
	}
	
	void set(float* p, int num=4) const;
	void set(double* p, int num=4) const;
	void setValues(double v1, double v2, double v3, double v4);
};

//these are all dangerous, but they're convenient!


#endif
