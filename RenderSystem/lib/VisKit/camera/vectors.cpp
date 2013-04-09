#include "vectors.h"
#include "matrices.h"

#ifdef QT_CORE_LIB
#include <QDataStream>
#endif

#define X elements[0]
#define Y elements[1]
#define Z elements[2]
#define D elements[3]

#include <cmath>
#define EPSILON 0.000000000001

const Vector3 Vector3::xAxis(1, 0, 0);
const Vector3 Vector3::yAxis(0, 1, 0);
const Vector3 Vector3::zAxis(0, 0, 1);
const Vector3 Vector3::Zero;

Vector2 Vector2::operator+(const Vector2 &rhs) const {
	return Vector2(X + rhs.X, Y + rhs.Y);
}

Vector2& Vector2::operator+=(const Vector2 &rhs) {
	X += rhs.X;
	Y += rhs.Y;
	return *this;
}

Vector2 Vector2::operator-(const Vector2& rhs) const {
	return Vector2(X - rhs.X, Y - rhs.Y);
}

Vector2& Vector2::operator-=(const Vector2& rhs) {
	X -= rhs.X;
	Y -= rhs.Y;
	return *this;
}

Vector2 Vector2::operator*(const double &rhs) const {
	return Vector2(X*rhs, Y*rhs);
}

Vector2& Vector2::operator*=(const double &rhs) {
	X *= rhs;
	Y *= rhs;
	return *this;
}

Vector2 Vector2::operator/(const double &rhs) const {
	return *this * (1./rhs);
}

Vector2& Vector2::operator/=(const double &rhs) {
	return (*this *= (1./rhs));
}

Vector2& Vector2::normalize() {
	double l = 1./sqrt(X*X + Y*Y);
	return ((*this) *= l);
}

double Vector2::dot(const Vector2 &rhs) const {
	return X*rhs.X + Y*rhs.Y;
}

double& Vector2::operator[](const int &index) {
	if(index == 1)
		return Y;
	return X; //if the index is out of bounds, just return X.
}


double Vector2::length() const {
	return sqrt(lengthSquared());
}


#ifdef QT_CORE_LIB
void Vector2::save(QIODevice& io) {
	io.write((char*)elements, 16);
}
void Vector2::load(QIODevice& io) {
	io.read((char*)elements, 16);
}
#endif

Vector3 Vector3::operator+(const Vector3 &rhs) const {
	return Vector3(X + rhs.X, Y + rhs.Y, Z + rhs.Z);
}

Vector3& Vector3::operator+=(const Vector3 &rhs) {
	X += rhs.X;
	Y += rhs.Y;
	Z += rhs.Z;
	return *this;
}

Vector3 Vector3::operator-(const Vector3& rhs) const {
	return Vector3(X - rhs.X, Y - rhs.Y, Z - rhs.Z);
}

Vector3& Vector3::operator-=(const Vector3& rhs) {
	X -= rhs.X;
	Y -= rhs.Y;
	Z -= rhs.Z;
	return *this;
}

Vector3 Vector3::operator*(const double &rhs) const {
	return Vector3(X*rhs, Y*rhs, Z*rhs);
}

Vector3& Vector3::operator*=(const double &rhs) {
	X *= rhs;
	Y *= rhs;
	Z *= rhs;
	return *this;
}

Vector3 Vector3::operator*(const Vector3 &rhs) const {
	return Vector3( Y*rhs.Z - Z*rhs.Y, Z*rhs.X - X*rhs.Z, X*rhs.Y - Y*rhs.X );
}

Vector3& Vector3::operator*=(const Vector3 &rhs) {
	double x, y, z;
	x = Y*rhs.Z - Z*rhs.Y;
	y = Z*rhs.X - X*rhs.Z;
	z = X*rhs.Y - Y*rhs.X;
	X = x;
	Y = y;
	Z = z;
	return *this;
}


Vector3 Vector3::operator/(const double &rhs) const {
	return *this * (1./rhs);
}

Vector3& Vector3::operator/=(const double &rhs) {
	return (*this *= (1./rhs));
}

Vector3& Vector3::normalize() {
	if(lengthSquared() > EPSILON) {
		double l = 1./sqrt(X*X + Y*Y + Z*Z);
		return ((*this) *= l);
	}
	return *this;
}

double Vector3::dot(const Vector3 &rhs) const {
	return X*rhs.X + Y*rhs.Y + Z*rhs.Z;
}

double& Vector3::operator[](const int &index) {
	if(index == 1)
		return Y;
	else if(index == 2)
		return Z;
	return X; //if the index is out of bounds, just return X.
}

const double& Vector3::operator[](const int &index) const {
	if(index == 1)
		return Y;
	else if(index == 2)
		return Z;
	return X; //if the index is out of bounds, just return X.
}

Vector3 Vector3::operator*(const Matrix4x4& rhs) const {
	Vector4 x(*this, 1.0);
	Vector3 ret;
	ret.x() = x.dot(rhs.col(0));
	ret.y() = x.dot(rhs.col(1));
	ret.z() = x.dot(rhs.col(2));
	return ret;
}

Vector3& Vector3::operator*=(const Matrix4x4& rhs) {
	*this = (*this)*rhs;
	return *this;
}

double Vector3::length() const {
	return sqrt(lengthSquared());
}


#ifdef QT_CORE_LIB
void Vector3::save(QIODevice& io) {
	io.write((char*)elements, 24);
}
void Vector3::load(QIODevice& io) {
	io.read((char*)elements, 24);
}
#endif

Vector4 Vector4::operator+(const Vector4 &rhs) const {
	return Vector4(X + rhs.X, Y + rhs.Y, Z + rhs.Z, D + rhs.D);
}

Vector4& Vector4::operator+=(const Vector4 &rhs) {
	X += rhs.X;
	Y += rhs.Y;
	Z += rhs.Z;
	D += rhs.D;
	return *this;
}

Vector4 Vector4::operator-(const Vector4& rhs) const {
	return Vector4(X - rhs.X, Y - rhs.Y, Z - rhs.Z, D - rhs.D);
}

Vector4& Vector4::operator-=(const Vector4& rhs) {
	X -= rhs.X;
	Y -= rhs.Y;
	Z -= rhs.Z;
	D -= rhs.D;
	return *this;
}

Vector4 Vector4::operator*(const double &rhs) const {
	return Vector4(X*rhs, Y*rhs, Z*rhs, D*rhs);
}

Vector4& Vector4::operator*=(const double &rhs) {
	X *= rhs;
	Y *= rhs;
	Z *= rhs;
	D *= rhs;
	return *this;
}

Vector4 Vector4::operator/(const double &rhs) const {
	return *this * (1./rhs);
}

Vector4& Vector4::operator/=(const double &rhs) {
	return (*this *= (1./rhs));
}

Vector4& Vector4::normalize() {
	double l = 1./sqrt(X*X + Y*Y + Z*Z + D*D);
	return ((*this) *= l);
}


Vector4& Vector4::homogenize() {
	double l = 1./D;
	return ((*this) *= l);
}

double Vector4::dot(const Vector4 &rhs) const {
	return X*rhs.X + Y*rhs.Y + Z*rhs.Z + D*rhs.D;
}

double& Vector4::operator[](const int &index) {
	if(index < 1 || index > 3) {
		return X;
	}
	return elements[index];
}

Vector4 Vector4::operator*(const Matrix4x4& rhs) const {
	Vector4 ret;
	ret.x() = dot(rhs.col(0));
	ret.y() = dot(rhs.col(1));
	ret.z() = dot(rhs.col(2));
	ret.d() = dot(rhs.col(3));
	return ret;
}

Vector4& Vector4::operator*=(const Matrix4x4& rhs) {
	*this = (*this)*rhs;
	return *this;
}

double Vector4::length() const {
	return sqrt(lengthSquared());
}

#ifdef QT_CORE_LIB
void Vector4::save(QIODevice& io) {
	io.write((char*)elements, 32);
}
void Vector4::load(QIODevice& io) {
	io.read((char*)elements, 32);
}
QTextStream& operator>>(QTextStream& lhs, Vector3& rhs) {
	lhs >> rhs.elements[0] >> rhs.elements[1] >> rhs.elements[2];
	return lhs;
}
QTextStream& operator<<(QTextStream& lhs, const Vector3& rhs) {
	lhs << rhs.elements[0] << " " << rhs.elements[1] << " " << rhs.elements[2];
	return lhs;
}

QTextStream& operator>>(QTextStream& lhs, Vector4& rhs) {
	lhs >> rhs.elements[0] >> rhs.elements[1] >> rhs.elements[2] >> rhs.elements[3];
	return lhs;
}
QTextStream& operator<<(QTextStream& lhs, const Vector4& rhs) {
	lhs << rhs.elements[0] << " " << rhs.elements[1] << " " << rhs.elements[2] << " " << rhs.elements[3];
	return lhs;
}

QDataStream& operator<<(QDataStream& lhs, const Vector2& rhs) {
	lhs.device()->write((char*)rhs.elements, 16);
	return lhs;
}

QDataStream& operator>>(QDataStream& lhs, Vector2& rhs) {
	lhs.device()->read((char*)rhs.elements, 16);
	return lhs;
}
QDataStream& operator<<(QDataStream& lhs, const Vector3& rhs) {
	lhs.device()->write((char*)rhs.elements, 24);
	return lhs;
}

QDataStream& operator>>(QDataStream& lhs, Vector3& rhs) {
	lhs.device()->read((char*)rhs.elements, 24);
	return lhs;
}
QDataStream& operator<<(QDataStream& lhs, const Vector4& rhs) {
	lhs.device()->write((char*)rhs.elements, 32);
	return lhs;
}

QDataStream& operator>>(QDataStream& lhs, Vector4& rhs) {
	lhs.device()->read((char*)rhs.elements, 32);
	return lhs;
}
#endif
Vector3 operator*(double lhs, const Vector3& rhs) {
	return rhs*lhs;
}

#ifdef QT_CORE_LIB
void Vector3::writeFloats(QIODevice* d) const {
	float f[3];
	f[0] = (float)X;
	f[1] = (float)Y;
	f[2] = (float)Z;
	d->write((char*)f, 12);
}
#endif

void Vector2::set(double *p, int num) const {
	for(int i = 0; i < (2 < num ? 2 : num); i++) {
		p[i] = elements[i];
	}
}

void Vector3::set(double *p, int num) const {
	for(int i = 0; i < (3 < num ? 3 : num); i++) {
		p[i] = elements[i];
	}
}

void Vector4::set(double *p, int num) const {
	for(int i = 0; i < (4 < num ? 4 : num); i++) {
		p[i] = elements[i];
	}
}

void Vector2::set(float *p, int num) const {
	for(int i = 0; i < (2 < num ? 2 : num); i++) {
		p[i] = (float)elements[i];
	}
}

void Vector3::set(float *p, int num) const {
	for(int i = 0; i < (3 < num ? 3 : num); i++) {
		p[i] = (float)elements[i];
	}
}

void Vector4::set(float *p, int num) const {
	for(int i = 0; i < (4 < num ? 4 : num); i++) {
		p[i] = (float)elements[i];
	}
}

Vector3 Vector3::reflect(const Vector3 &other) const {
	return (*this) - other*2*dot(other);
}

Vector3 Vector3::refract(const Vector3& normal, double eta) const {
	double k = 1.0 - eta*eta*(1.0 - dot(normal)*dot(normal));
	if(k < 0)
		return Vector3();
	return eta*(*this) - (eta*dot(normal) + sqrt(k))*normal;
}
void Vector2::setValues(double v1, double v2) {
	elements[0] = v1;
	elements[1] = v2;
}
void Vector3::setValues(double v1, double v2, double v3) {
	elements[0] = v1;
	elements[1] = v2;
	elements[2] = v3;
}
void Vector4::setValues(double v1, double v2, double v3, double v4) {
	elements[0] = v1;
	elements[1] = v2;
	elements[2] = v3;
	elements[3] = v4;
}

Vector3::Vector3(const Vector4& vec) {
	setValues(unpack3(vec));
}

