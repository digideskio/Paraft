#include "quaternion.h"
#include <cmath>

#define TOLERANCE 0.00000001

#define X elements[0]
#define Y elements[1]
#define Z elements[2]
#define W elements[3]

Quaternion::Quaternion() {
	X = 0;
	Y = 0;
	Z = 0;
	W = 1;
}

Quaternion::Quaternion(double x, double y, double z, double w) {
	X = x;
	Y = y;
	Z = z;
	W = w;
}

Quaternion::Quaternion(const Quaternion& other) {
	X = other.X;
	Y = other.Y;
	Z = other.Z;
	W = other.W;
}


Quaternion::Quaternion(const Vector3& vec, double angle) {
	angle *= 0.5;
	Vector3 temp(vec);
	temp.normalize();
	
	temp *= sin(angle);
	X = temp.x();
	Y = temp.y();
	Z = temp.z();
	W = cos(angle);
	
}

Quaternion& Quaternion::operator*=(const double &rhs) {
	elements[0] *= rhs;
	elements[1] *= rhs;
	elements[2] *= rhs;
	elements[3] *= rhs;
	
	return *this;
}


void Quaternion::fromAxis(double dx, double dy, double dz, double angle) {
	angle *= 0.5;

	double mag = 1 / length();
	dx *= mag;
	dy *= mag;
	dz *= mag;

	double sAngle = sin(angle);

	X = dx * sAngle;
	Y = dy * sAngle;
	Z = dz * sAngle;
	W = cos(angle);

}

Quaternion Quaternion::getConjugate() const {
	return Quaternion( -X, -Y, -Z, W );
}

Quaternion Quaternion::operator*(const Quaternion& rhs) const {
	return Quaternion(W * rhs.X + X * rhs.W + Y * rhs.Z - Z * rhs.Y,
		              W * rhs.Y + Y * rhs.W + Z * rhs.X - X * rhs.Z,
		              W * rhs.Z + Z * rhs.W + X * rhs.Y - Y * rhs.X,
		              W * rhs.W - X * rhs.X - Y * rhs.Y - Z * rhs.Z);

}

Quaternion& Quaternion::operator*=(const Quaternion& rhs) {
	*this = *this * rhs;
	return *this;
}

void Quaternion::normalize() {
	double mag = lengthSquared();
	//don't try to normalize if the length is 0, bad things happen
	if(fabs(mag - 1.0) > TOLERANCE) {
		mag = 1./sqrt(mag);
		*this *= mag;
	}
}

void Quaternion::rotateVector(Vector3& vector) {
	Quaternion quat(vector.x(), vector.y(), vector.z(), 0);
	quat *= getConjugate();
	quat = *this * quat;
	//quat = getConjugate() * quat;
	vector.x() = quat.X;
	vector.y() = quat.Y;
	vector.z() = quat.Z;
}

void Quaternion::rotateVector(double &vx, double &vy, double &vz) {
	Quaternion quat(vx, vy, vz, 0.0);
	quat *= getConjugate();
	quat = *this * quat;

	vx = quat.X;
	vy = quat.Y;
	vz = quat.Z;
}

Quaternion& Quaternion::operator=(const Quaternion& rhs) {
	X = rhs.X;
	Y = rhs.Y;
	Z = rhs.Z;
	W = rhs.W;

	return *this;
}

double Quaternion::length() {
	return sqrt(lengthSquared());
}

Quaternion::operator Matrix4x4() const {
	Matrix4x4 m;
	double wx, wy, wz, xx, yy, zz, xy, xz, yz, x2, y2, z2;	

	x2 = X + X; y2 = Y + Y;
	z2 = Z + Z;
	xx = X * x2; xy = X * y2; xz = X * z2;
	yy = Y * y2; yz = Y * z2; zz = Z * z2;
	wx = W * x2; wy = W * y2; wz = W * z2;


	m[0][0] = 1.0 - (yy + zz); m[1][0] = xy - wz;
	m[2][0] = xz + wy; m[3][0] = 0.0;

	m[0][1] = xy + wz; m[1][1] = 1.0 - (xx + zz);
	m[2][1] = yz - wx; m[3][1] = 0.0;


	m[0][2] = xz - wy; m[1][2] = yz + wx;
	m[2][2] = 1.0 - (xx + yy); m[3][2] = 0.0;


	m[0][3] = 0; m[1][3] = 0;
	m[2][3] = 0; m[3][3] = 1;

	return m;
}

Quaternion::operator double*() const {
	return (double*)(Matrix4x4)(*this);
}
Quaternion Quaternion::operator*(double rhs) const {
	return Quaternion(X*rhs, Y*rhs, Z*rhs, W*rhs);
}

Vector3 Quaternion::slerp(const Vector3& first, const Vector3& second, double time) {
	Vector3 axis = first*second;
	double angle = first.dot(second);
	if(fabs(1 - angle) < TOLERANCE) {
		return first;
	}
	angle = acos(angle);
	Quaternion q1;
	Quaternion q2(axis, angle);
	q2.normalize();
	angle *= 0.5;
	Quaternion qm = (q1 * sin(angle*(1. - time)) + q2 * sin(angle*time))/sin(angle);
	qm.normalize();
	Vector3 ret(first);
	qm.rotateVector(ret);
	return ret;
}

Quaternion Quaternion::slerp(const Quaternion& q1, const Quaternion& q2, double t) {
	Quaternion ret = q1*(1.0 - t) + q2*t;
	ret.normalize();
	return ret;
}


Quaternion Quaternion::operator/(double rhs) const {
	return (*this)*(1./rhs);
}

Quaternion Quaternion::operator+(const Quaternion& rhs) const {
	return Quaternion(X + rhs.X, Y + rhs.Y, Z + rhs.Z, W + rhs.W);
}

