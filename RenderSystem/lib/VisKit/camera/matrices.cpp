#include "matrices.h"
#include "quaternion.h"
#include <cmath>

const Matrix4x4 Matrix4x4::identity = Matrix4x4();

Matrix4x4 Matrix4x4::operator+(const Matrix4x4 &rhs) const {
	Matrix4x4 ret(*this);
	ret += rhs;
	return ret;
}

Matrix4x4& Matrix4x4::operator+=(const Matrix4x4 &rhs) {
	for(int i = 0; i < 16; i++) {
		elements[i] += rhs.elements[i];
	}
	return *this;
}

Matrix4x4 Matrix4x4::operator-(const Matrix4x4 &rhs) const {
	Matrix4x4 ret(*this);
	ret -= rhs;
	return ret;
}

Matrix4x4& Matrix4x4::operator-=(const Matrix4x4 &rhs) {
	for(int i = 0; i < 16; i++) {
		elements[i] -= rhs.elements[i];
	}
	return *this;
}

Vector4 Matrix4x4::row(const int& index) const {
	return Vector4(elements + (index >= 0 && index <= 3 ? index : 0), 4);
}

Vector4 Matrix4x4::col(const int& index) const {
	return Vector4(elements + ((index >= 0 && index <= 3) ? index*4 : 0));
}


Matrix4x4& Matrix4x4::operator*=(const Matrix4x4 &rhs) {
	Vector4 rows[4];
	Vector4 cols[4];
	for(int i = 0; i < 4; i++) {
		rows[i] = row(i);
		cols[i] = rhs.col(i);
	}
	for(int c = 0; c < 4; c++) {
		for(int r = 0; r < 4; r++) {
			(*this)[c][r] = rows[r].dot(cols[c]);
		}
	}
	return (*this);
}

Matrix4x4 Matrix4x4::operator*(const Matrix4x4 &rhs) const {
	Matrix4x4 ret(*this);
	return (ret *= rhs);
}

Matrix4x4 Matrix4x4::operator*(const double &rhs) const {
	Matrix4x4 ret(*this);
	return (ret *= rhs);
}

Matrix4x4& Matrix4x4::operator*=(const double &rhs) {
	for(int i = 0; i < 16; i++) {
		elements[i] *= rhs;
	}
	return *this;
}


Matrix4x4 Matrix4x4::operator/(const double &rhs) const {
	return (*this * (1./rhs));
}

Matrix4x4& Matrix4x4::operator/=(const double &rhs) {
	return (*this *= (1./rhs));
}


Matrix4x4& Matrix4x4::operator=(const Matrix4x4 &rhs) {
	for(int i = 0; i < 16; i++) {
		elements[i] = rhs.elements[i];
	}
	return *this;
}

Matrix4x4::operator double*() {
	return elements;
}


#ifdef QT_CORE_LIB
void Matrix4x4::save(QIODevice& io) {
	io.write((char*)elements, 128);
}
void Matrix4x4::load(QIODevice& io) {
	io.read((char*)elements, 128);
}
#endif

Vector3 Matrix4x4::transform(const Vector3& vec) const {
	return (*this)*vec;
}

Vector3 Matrix4x4::operator *(const Vector3& rhs) const {
	Vector4 vec(rhs, 1.0);
	vec = (*this)*vec;
	vec /= vec.d(); //when in doubt, read the OpenGL spec (sigh)
	return Vector3(vec.x(), vec.y(), vec.z());
}

Matrix4x4 Matrix4x4::fromLookAt(const Vector3& camera, const Vector3& focus, const Vector3& up) {
	Matrix4x4 ret;
	Vector3 f = focus - camera;
	Vector3 u = up;
	f.normalize();
	u.normalize();
	Vector3 s = f*u;
	u = s*f;
	for(int c = 0; c < 3; c++) {
		ret[c][0] = s[c];
		ret[c][1] = u[c];
		ret[c][2] = -f[c];
	}
	ret *= Matrix4x4::fromTranslation(-camera);
	return ret;
}

Vector4 Matrix4x4::operator*(const Vector4& rhs) const {
	Vector4 ret;
	ret.x() = row(0).dot(rhs);
	ret.y() = row(1).dot(rhs);
	ret.z() = row(2).dot(rhs);
	ret.d() = row(3).dot(rhs);
	return ret;
}

Vector2 Matrix2x2::operator *(const Vector2 &rhs) {
	return Vector2(Vector2(elements[0], elements[2]).dot(rhs), Vector2(elements[1], elements[3]).dot(rhs));
}

void Matrix2x2::inverse() {
	Matrix2x2 temp(*this);
	double inv = 1./(elements[0]*elements[3] - elements[1]*elements[2]);
	elements[0] = temp.elements[3]*inv;
	elements[1] = -temp.elements[2]*inv;
	elements[2] = -temp.elements[1]*inv;
	elements[3] = temp.elements[0]*inv;
}

Matrix3x3 Matrix3x3::operator+(const Matrix3x3 &rhs) const {
	Matrix3x3 ret(*this);
	ret += rhs;
	return ret;
}

Matrix3x3& Matrix3x3::operator+=(const Matrix3x3 &rhs) {
	for(int i = 0; i < 9; i++) {
		elements[i] += rhs.elements[i];
	}
	return *this;
}

Matrix3x3 Matrix3x3::operator-(const Matrix3x3 &rhs) const {
	Matrix3x3 ret(*this);
	ret -= rhs;
	return ret;
}

Matrix3x3& Matrix3x3::operator-=(const Matrix3x3 &rhs) {
	for(int i = 0; i < 9; i++) {
		elements[i] -= rhs.elements[i];
	}
	return *this;
}

Matrix3x3 Matrix3x3::operator*(const double &rhs) const {
	Matrix3x3 ret(*this);
	return (ret *= rhs);
}

Matrix3x3 Matrix3x3::operator*(const Matrix3x3& rhs) const {
	Matrix3x3 ret;
	ret *= rhs;
	return ret;
}

Matrix3x3& Matrix3x3::operator*=(const Matrix3x3& rhs) {
	Vector3 rows[3];
	Vector3 cols[3];
	for(int i = 0; i < 3; i++) {
		rows[i] = row(i);
		cols[i] = rhs.col(i);
	}
	for(int c = 0; c < 3; c++) {
		for(int r = 0; r < 3; r++) {
			elements[c*3 + r] = rows[r].dot(cols[c]);
		}
	}
	return *this;
}

Vector3 Matrix3x3::operator*(const Vector3& rhs) const {
	Vector3 ret;
	ret.x() = row(0).dot(rhs);
	ret.y() = row(1).dot(rhs);
	ret.z() = row(2).dot(rhs);
	return ret;
}

Vector3 Matrix3x3::row(int index) const {
	return Vector3(elements + ((index >= 0 && index <= 2) ? index : 0), 3);
}

Vector3 Matrix3x3::col(int index) const {
	return Vector3(elements + ((index >= 0 && index <= 2) ? index*3 : 0));
}

Matrix3x3 Matrix3x3::fromRotation(double radians, const Vector2& point) {
	Matrix3x3 trans1 = fromTranslation(-point);

	Matrix3x3 rot;
	rot[0][0] = cos(radians);
	rot[0][1] = sin(radians);
	rot[1][0] = -rot[0][1];
	rot[1][1] = rot[0][0];

	Matrix3x3 trans2 = fromTranslation(point);

	trans2 *= rot;
	trans2 *= trans1;

	return trans2;

}

Matrix3x3 Matrix3x3::fromTranslation(const Vector2& offset) {
	Matrix3x3 ret;
	ret[2][0] = offset.x();
	ret[2][1] = offset.y();
	return ret;
}

Matrix3x3 Matrix3x3::fromScale(double scale, const Vector2& point) {
	Matrix3x3 t1 = fromTranslation(-point);
	Matrix3x3 s;
	for(int i = 0; i < 2; i++) {
		s[i][i] = scale;
	}
	Matrix3x3 t2 = fromTranslation(point);

	t2 *= s;
	t2 *= t1;
	return t2;
}

Matrix4x4 Matrix4x4::fromTranslation(const Vector3& offset) {
	Matrix4x4 ret;
	ret[3][0] = offset.x();
	ret[3][1] = offset.y();
	ret[3][2] = offset.z();
	return ret;
}

Matrix4x4 Matrix4x4::fromRotation(const Vector3& axis, double radians, const Vector3& offset) {
	Matrix4x4 ret = Matrix4x4::fromTranslation(offset);
	Quaternion quat(axis, radians);
	ret *= (Matrix4x4)quat;
	ret *= Matrix4x4::fromTranslation(-offset);
	return ret;
}

Matrix4x4 Matrix4x4::fromScale(double scale, const Vector3& point) {
	Matrix4x4 ret = Matrix4x4::fromTranslation(point);
	Matrix4x4 s;
	s[0][0] = scale;
	s[1][1] = scale;
	s[2][2] = scale;
	ret *= s;
	ret *= Matrix4x4::fromTranslation(-point);
	return ret;
}

Matrix4x4 Matrix4x4::fromScale(const Vector3& scale, const Vector3& point) {
	Matrix4x4 ret = Matrix4x4::fromTranslation(point);
	Matrix4x4 s;
	s[0][0] = scale.x();
	s[1][1] = scale.y();
	s[2][2] = scale.z();
	ret *= s;
	ret *= Matrix4x4::fromTranslation(-point);
	return ret;
}

Matrix4x4 Matrix4x4::fromOrtho(double left, double right, double bottom, double top, double near, double far) {
	Matrix4x4 ret;

	if(right == left || bottom == top || near == far) {
#ifdef QT_CORE_LIB
		qDebug("Invalid ortho parameters left: %f right: %f bottom: %f top: %f near: %f far: %f", left, right, bottom, top, near, far);
#endif
		return ret;
	}
	ret[0][0] = 2./(right - left);
	ret[1][1] = 2./(top - bottom);
	ret[2][2] = -2./(far - near);
	ret[3][0] = -(right + left)/(right - left);
	ret[3][1] = -(top + bottom)/(top - bottom);
	ret[3][2] = -(far + near)/(far - near);
	return ret;
}

Matrix4x4 Matrix4x4::fromPerspective(double fov, double aspect, double zNear, double zFar) {
	Matrix4x4 ret;
	fov = 1./(tan(fov/2.));
	ret[0][0] = fov/aspect;
	ret[1][1] = fov;
	ret[2][2] = (zFar + zNear)/(zNear - zFar);
	ret[3][3] = 0;
	ret[3][2] = (2*zFar*zNear)/(zNear - zFar);
	ret[2][3] = -1;
	return ret;
}

Matrix4x4 Matrix4x4::fromScreen(double xMin, double xMax, double yMin, double yMax) {
	//qDebug("%f %f %f %f", xMin, xMax, yMin, yMax);
	Matrix4x4 ret;
	ret[0][0] = (xMax - xMin)/2.;
	ret[1][1] = (yMax - yMin)/2.;
	ret[2][2] = 0.5;
	ret[3][0] = (xMax + xMin)/2.;
	ret[3][1] = (yMax + yMin)/2.;
	ret[3][2] = 0.5;
	return ret;
}

void Matrix4x4::print() const {

#ifdef QT_CORE_LIB
	qDebug("Matrix print:");
	for(int i = 0; i < 4; ++i) {
		qDebug("%f %f %f %f", elements[0 + i], elements[4 + i], elements[8 + i], elements[12 + i]);
	}
#endif
}
//http://www.dr-lex.be/random/matrix_inv.html
Matrix3x3& Matrix3x3::inverse() {
	double determinant = det();
	if(fabs(det()) < 0.000000000001) //matrix has no inverse
		return *this;

	determinant = 1./determinant;
	Matrix3x3 copy(*this);
	elements[0] = copy[2][2]*copy[1][1] - copy[1][2]*copy[2][1];
	elements[1] = copy[2][1]*copy[0][2] - copy[2][2]*copy[0][1];
	elements[2] = copy[0][1]*copy[1][2] - copy[0][2]*copy[1][1];

	elements[3] = copy[2][0]*copy[1][2] - copy[2][2]*copy[1][0];
	elements[4] = copy[0][0]*copy[2][2] - copy[0][2]*copy[2][0];
	elements[5] = copy[1][0]*copy[0][2] - copy[1][2]*copy[0][0];

	elements[6] = copy[1][0]*copy[2][1] - copy[1][1]*copy[2][0];
	elements[7] = copy[2][0]*copy[0][1] - copy[2][1]*copy[0][0];
	elements[8] = copy[0][0]*copy[1][1] - copy[0][1]*copy[1][0];

	*this *= determinant;
	return *this;
}

double Matrix4x4::det() const {
#define a(i,j) (elements[(j-1)*4 + (i-1)])
	return	a(1,1)*a(2,2)*a(3,3)*a(4,4) + a(1,1)*a(2,3)*a(3,4)*a(4,2) + a(1,1)*a(2,4)*a(3,2)*a(4,3) +
			a(1,2)*a(2,1)*a(3,4)*a(4,3) + a(1,2)*a(2,3)*a(3,1)*a(4,4) + a(1,2)*a(2,4)*a(3,3)*a(4,1) +
			a(1,3)*a(2,1)*a(3,2)*a(4,4) + a(1,3)*a(2,2)*a(3,4)*a(4,1) + a(1,3)*a(2,4)*a(3,1)*a(4,2) +
			a(1,4)*a(2,1)*a(3,3)*a(4,2) + a(1,4)*a(2,2)*a(3,1)*a(4,3) + a(1,4)*a(2,3)*a(3,2)*a(4,1) -
			a(1,1)*a(2,2)*a(3,4)*a(4,3) - a(1,1)*a(2,3)*a(3,2)*a(4,4) - a(1,1)*a(2,4)*a(3,3)*a(4,2) -
			a(1,2)*a(2,1)*a(3,3)*a(4,4) - a(1,2)*a(2,3)*a(3,4)*a(4,1) - a(1,2)*a(2,4)*a(3,1)*a(4,3) -
			a(1,3)*a(2,1)*a(3,4)*a(4,2) - a(1,3)*a(2,2)*a(3,1)*a(4,4) - a(1,3)*a(2,4)*a(3,2)*a(4,1) -
			a(1,4)*a(2,1)*a(3,2)*a(4,3) - a(1,4)*a(2,2)*a(3,3)*a(4,1) - a(1,4)*a(2,3)*a(3,1)*a(4,2);
#undef a

//		elements[0]*elements[5]*elements[10]*elements[15] + elements[0]*elements[9]*elements[14]*elements[7]  + elements[0]*elements[13]*elements[6]*elements[11] +
//		elements[4]*elements[1]*elements[14]*elements[11] + elements[4]*elements[9]*elements[2]*elements[15]  + elements[4]*elements[13]*elements[10]*elements[3] +
//		elements[8]*elements[1]*elements[6]*elements[15]  + elements[8]*elements[5]*elements[14]*elements[3]  + elements[8]*elements[13]*elements[2]*elements[7]  +
//		elements[12]*elements[1]*elements[10]*elements[7] + elements[12]*elements[5]*elements[2]*elements[11] + elements[12]*elements[9]*elements[6]*elements[3]  -
//		elements[0]*elements[5]*elements[14]*elements[11] - elements[0]*elements[9]*elements[6]*elements[15]  - elements[0]*elements[13]*elements[10]*elements[7] -
//		elements[4]*elements[1]*elements[10]*elements[15] - elements[4]*elements[9]*elements[14]*elements[3]  - elements[4]*elements[13]*elements[2]*elements[11] -
//		elements[8]*elements[1]*elements[14]*elements[7]  - elements[8]*elements[5]*elements[2]*elements[15]  - elements[8]*elements[13]*elements[6]*elements[3]  -
//		elements[12]*elements[1]*elements[6]*elements[11] - elements[12]*elements[5]*elements[10]*elements[3] - elements[12]*elements[9]*elements[2]*elements[7];
}

Matrix4x4& Matrix4x4::inverse() {
	double determinant = det();
	if(fabs(det()) < 0.000000000001) //matrix has no inverse
		return *this;

	determinant = 1./determinant;
	Matrix4x4 copy(*this);

#define a(i,j) (copy[j-1][i-1])
#define b(i,j) (elements[(j-1)*4 + (i-1)])

	b(1,1) =
	(a(2,2)*a(3,3)*a(4,4) + a(2,3)*a(3,4)*a(4,2) + a(2,4)*a(3,2)*a(4,3)
	 - a(2,2)*a(3,4)*a(4,3) - a(2,3)*a(3,2)*a(4,4) -a(2,4)*a(3,3)*a(4,2))*determinant;

	b(1,2) =
	(a(1,2)*a(3,4)*a(4,3) + a(1,3)*a(3,2)*a(4,4) + a(1,4)*a(3,3)*a(4,2)
	 - a(1,2)*a(3,3)*a(4,4) - a(1,3)*a(3,4)*a(4,2) - a(1,4)*a(3,2)*a(4,3))*determinant;

	b(1,3) =
	(a(1,2)*a(2,3)*a(4,4) + a(1,3)*a(2,4)*a(4,2) + a(1,4)*a(2,2)*a(4,3)
	 - a(1,2)*a(2,4)*a(4,3) - a(1,3)*a(2,2)*a(4,4) - a(1,4)*a(2,3)*a(4,2))*determinant;

	b(1,4) =
	(a(1,2)*a(2,4)*a(3,3) + a(1,3)*a(2,2)*a(3,4) + a(1,4)*a(2,3)*a(3,2)
	 - a(1,2)*a(2,3)*a(3,4) - a(1,3)*a(2,4)*a(3,2) - a(1,4)*a(2,2)*a(3,3))*determinant;

	b(2,1) =
	(a(2,1)*a(3,4)*a(4,3) + a(2,3)*a(3,1)*a(4,4) + a(2,4)*a(3,3)*a(4,1)
	 - a(2,1)*a(3,3)*a(4,4) - a(2,3)*a(3,4)*a(4,1) - a(2,4)*a(3,1)*a(4,3))*determinant;

	b(2,2) =
	(a(1,1)*a(3,3)*a(4,4) + a(1,3)*a(3,4)*a(4,1) + a(1,4)*a(3,1)*a(4,3)
	 - a(1,1)*a(3,4)*a(4,3) - a(1,3)*a(3,1)*a(4,4) - a(1,4)*a(3,3)*a(4,1))*determinant;

	b(2,3) =
	(a(1,1)*a(2,4)*a(4,3) + a(1,3)*a(2,1)*a(4,4) + a(1,4)*a(2,3)*a(4,1)
	 - a(1,1)*a(2,3)*a(4,4) - a(1,3)*a(2,4)*a(4,1) - a(1,4)*a(2,1)*a(4,3))*determinant;

	b(2,4) =
	(a(1,1)*a(2,3)*a(3,4) + a(1,3)*a(2,4)*a(3,1) + a(1,4)*a(2,1)*a(3,3)
	 - a(1,1)*a(2,4)*a(3,3) - a(1,3)*a(2,1)*a(3,4) - a(1,4)*a(2,3)*a(3,1))*determinant;

	b(3,1) =
	(a(2,1)*a(3,2)*a(4,4) + a(2,2)*a(3,4)*a(4,1) + a(2,4)*a(3,1)*a(4,2)
	 - a(2,1)*a(3,4)*a(4,2) - a(2,2)*a(3,1)*a(4,4) - a(2,4)*a(3,2)*a(4,1))*determinant;

	b(3,2) =
	(a(1,1)*a(3,4)*a(4,2) + a(1,2)*a(3,1)*a(4,4) + a(1,4)*a(3,2)*a(4,1)
	 - a(1,1)*a(3,2)*a(4,4) - a(1,2)*a(3,4)*a(4,1) - a(1,4)*a(3,1)*a(4,2))*determinant;

	b(3,3) =
	(a(1,1)*a(2,2)*a(4,4) + a(1,2)*a(2,4)*a(4,1) + a(1,4)*a(2,1)*a(4,2)
	 - a(1,1)*a(2,4)*a(4,2) - a(1,2)*a(2,1)*a(4,4) - a(1,4)*a(2,2)*a(4,1))*determinant;

	b(3,4) =
	(a(1,1)*a(2,4)*a(3,2) + a(1,2)*a(2,1)*a(3,4) + a(1,4)*a(2,2)*a(3,1)
	 - a(1,1)*a(2,2)*a(3,4) - a(1,2)*a(2,4)*a(3,1) - a(1,4)*a(2,1)*a(3,2))*determinant;

	b(4,1) =
	(a(2,1)*a(3,3)*a(4,2) + a(2,2)*a(3,1)*a(4,3) + a(2,3)*a(3,2)*a(4,1)
	 - a(2,1)*a(3,2)*a(4,3) - a(2,2)*a(3,3)*a(4,1) - a(2,3)*a(3,1)*a(4,2))*determinant;

	b(4,2) =
	(a(1,1)*a(3,2)*a(4,3) + a(1,2)*a(3,3)*a(4,1) + a(1,3)*a(3,1)*a(4,2)
	 - a(1,1)*a(3,3)*a(4,2) - a(1,2)*a(3,1)*a(4,3) - a(1,3)*a(3,2)*a(4,1))*determinant;

	b(4,3) =
	(a(1,1)*a(2,3)*a(4,2) + a(1,2)*a(2,1)*a(4,3) + a(1,3)*a(2,2)*a(4,1)
	 - a(1,1)*a(2,2)*a(4,3) - a(1,2)*a(2,3)*a(4,1) - a(1,3)*a(2,1)*a(4,2))*determinant;

	b(4,4) =
	(a(1,1)*a(2,2)*a(3,3) + a(1,2)*a(2,3)*a(3,1) + a(1,3)*a(2,1)*a(3,2)
	 - a(1,1)*a(2,3)*a(3,2) - a(1,2)*a(2,1)*a(3,3) - a(1,3)*a(2,2)*a(3,1))*determinant;
#undef a
#undef b
	/*
	elements[0]  = c[1][1]*c[2][2]*c[3][3]+c[1][2]*c[2][3]*c[3][1]+c[1][3]*c[2][1]*c[3][2]-
				   c[1][1]*c[2][3]*c[3][2]-c[1][2]*c[2][1]*c[3][3]-c[1][3]*c[2][2]*c[3][1];
	elements[4]  = c[0][1]*c[2][3]*c[3][2]+c[0][2]*c[2][1]*c[3][3]+c[0][3]*c[2][2]*c[3][1]-
				   c[0][1]*c[2][2]*c[3][3]-c[0][2]*c[2][3]*c[3][1]-c[0][3]*c[2][1]*c[3][2];
	elements[8]  = c[0][1]*c[1][2]*c[3][3]+c[0][2]*c[1][3]*c[3][1]+c[0][3]*c[1][1]*c[3][2]-
				   c[0][1]*c[1][3]*c[3][2]-c[0][2]*c[1][1]*c[3][3]-c[0][3]*c[1][2]*c[3][1];
	elements[12] = c[0][1]*c[1][3]*c[2][2]+c[0][2]*c[1][1]*c[2][3]+c[0][3]*c[1][2]*c[2][1]-
				   c[0][1]*c[1][2]*c[2][3]-c[0][2]*c[1][3]*c[2][1]-c[0][3]*c[1][1]*c[2][2];

	elements[1]  = c[1][0]*c[2][3]*c[3][2]+c[1][2]*c[2][0]*c[3][3]+c[1][3]*c[2][2]*c[3][0]-
				   c[1][0]*c[2][2]*c[3][3]-c[1][2]*c[2][3]*c[3][0]-c[1][3]*c[2][0]*c[3][2];
	elements[5]  = c[0][0]*c[2][2]*c[3][3]+c[0][2]*c[2][3]*c[3][0]+c[0][3]*c[2][0]*c[3][2]-
				   c[0][0]*c[2][3]*c[3][2]-c[0][2]*c[2][0]*c[3][3]-c[0][3]*c[2][2]*c[3][0];
	elements[9]  = c[0][0]*c[1][3]*c[3][2]+c[0][2]*c[1][0]*c[3][3]+c[0][3]*c[1][2]*c[3][0]-
				   c[0][0]*c[1][2]*c[3][3]-c[0][2]*c[1][3]*c[3][3]-c[0][3]*c[1][0]*c[3][2];
	elements[13] = c[0][0]*c[1][2]*c[2][3]+c[0][2]*c[1][3]*c[2][0]+c[0][3]*c[1][0]*c[2][2]-
				   c[0][0]*c[1][3]*c[2][2]-c[0][2]*c[1][0]*c[2][3]-c[0][3]*c[1][2]*c[2][0];

	elements[2]  = c[1][0]*c[2][1]*c[3][3]+c[1][1]*c[2][3]*c[3][0]+c[1][3]*c[2][0]*c[3][1]-
				   c[1][0]*c[2][3]*c[3][1]-c[1][1]*c[2][0]*c[3][3]-c[1][3]*c[2][1]*c[3][0];
	elements[6]  = c[0][0]*c[2][3]*c[3][1]+c[0][1]*c[2][0]*c[3][3]+c[0][3]*c[2][1]*c[3][0]-
				   c[0][0]*c[2][1]*c[3][3]-c[0][1]*c[2][3]*c[3][0]-c[0][3]*c[2][0]*c[3][1];
	elements[10] = c[0][0]*c[1][1]*c[3][3]+c[0][1]*c[1][3]*c[3][0]+c[0][3]*c[1][0]*c[3][1]-
				   c[0][0]*c[1][3]*c[3][1]-c[0][1]*c[1][0]*c[3][3]-c[0][3]*c[1][1]*c[3][0];
	elements[14] = c[0][0]*c[1][3]*c[2][1]+c[0][1]*c[1][0]*c[2][3]+c[0][3]*c[1][1]*c[2][0]-
				   c[0][0]*c[1][1]*c[2][3]-c[0][1]*c[1][3]*c[2][0]-c[0][3]*c[1][0]*c[2][1];

	elements[3]  = c[1][0]*c[2][2]*c[3][1]+c[1][1]*c[2][0]*c[3][2]+c[1][2]*c[2][1]*c[3][0]-
				   c[1][0]*c[2][1]*c[3][2]-c[1][1]*c[2][2]*c[3][0]-c[1][2]*c[2][0]*c[3][1];
	elements[7]  = c[0][0]*c[2][1]*c[3][2]+c[0][1]*c[2][2]*c[3][0]+c[0][2]*c[2][0]*c[3][1]-
				   c[0][0]*c[2][2]*c[3][1]-c[0][1]*c[2][0]*c[3][2]-c[0][2]*c[2][1]*c[3][0];
	elements[11] = c[0][0]*c[1][2]*c[3][1]+c[0][1]*c[1][0]*c[3][2]+c[0][2]*c[1][1]*c[3][0]-
				   c[0][0]*c[1][1]*c[3][2]-c[0][1]*c[1][2]*c[3][0]-c[0][2]*c[1][0]*c[3][1];
	elements[15] = c[0][0]*c[1][1]*c[2][2]+c[0][1]*c[1][2]*c[2][0]+c[0][2]*c[1][0]*c[2][1]-
				   c[0][0]*c[1][2]*c[2][1]-c[0][1]*c[1][0]*c[2][2]-c[0][2]*c[1][1]*c[2][0];
	 */
	//*this *= determinant;
	return *this;
}

Matrix4x4 Matrix4x4::CatmullRom(const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& p4, double tau) {
	Matrix4x4 ret;
	p1.set(ret.elements);
	p2.set(ret.elements + 4);
	p3.set(ret.elements + 8);
	p4.set(ret.elements + 12);
	ret[0][3] = 1;
	ret[1][3] = 1;
	ret[2][3] = 1;

	Matrix4x4 t;
	t[0][0] = 0;
	t[0][1] = 1;

	t[1][0] = -tau;
	t[1][1] = 0;
	t[1][2] = tau;

	t[2][0] = 2*tau;
	t[2][1] = tau - 3;
	t[2][2] = 3 - 2*tau;
	t[2][3] = -tau;

	t[3][0] = -tau;
	t[3][1] = 2 - tau;
	t[3][2] = tau - 2;
	t[3][3] = tau;

	return ret*t;
}


void Matrix2x2::set(float* v, int cols, int rows) const {
	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < cols; c++) {
			if(r < 2 && c < 2) {
				v[c*rows + r] = elements[c*2 + r];
			}
		}
	}
}


void Matrix2x2::set(double* v, int cols, int rows) const {
	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < cols; c++) {
			if(r < 2 && c < 2) {
				v[c*rows + r] = elements[c*2 + r];
			}
		}
	}
}



void Matrix3x3::set(float* v, int cols, int rows) const {
	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < cols; c++) {
			if(r < 3 && c < 3) {
				v[c*rows + r] = elements[c*3 + r];
			}
		}
	}
}

void Matrix3x3::set(double* v, int cols, int rows) const {
	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < cols; c++) {
			if(r < 3 && c < 3) {
				v[c*rows + r] = elements[c*3 + r];
			}
		}
	}
}

void Matrix4x4::set(float* v, int cols, int rows) const {
	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < cols; c++) {
			if(r < 4 && c < 4) {
				v[c*rows + r] = elements[c*4 + r];
			}
		}
	}
}

void Matrix4x4::set(double* v, int cols, int rows) const {
	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < cols; c++) {
			if(r < 4 && c < 4) {
				v[c*rows + r] = elements[c*4 + r];
			}
		}
	}
}

Matrix4x4 Matrix4x4::fromFrustrum(double left, double right, double bottom, double top, double near, double far) {
	Matrix4x4 ret;
	ret[0][0] = 2.*near/(right - left);
	ret[1][1] = 2.*near/(top - bottom);
	ret[2][2] = (far + near)/(near - far);
	ret[3][3] = 0;
	ret[2][3] = -1;
	ret[2][0] = (right + left)/(right - left);
	ret[2][1] = (top + bottom)/(top - bottom);
	ret[3][2] = 2*far*near/(near - far);

	return ret;
}

inline void swap(double& t1, double& t2) {
	if(t1 == t2)
		return;
	*reinterpret_cast<long long*>(&t1) ^= *reinterpret_cast<long long*>(&t2);
	*reinterpret_cast<long long*>(&t2) ^= *reinterpret_cast<long long*>(&t1);
	*reinterpret_cast<long long*>(&t1) ^= *reinterpret_cast<long long*>(&t2);
}

void Matrix4x4::transpose() {
	swap(elements[1], elements[4]);
	swap(elements[2], elements[8]);
	swap(elements[3], elements[12]);
	swap(elements[6], elements[9]);
	swap(elements[7], elements[13]);
	swap(elements[11], elements[14]);
}
