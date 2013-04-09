#ifndef _MATRICES_H_
#define _MATRICES_H_


#include "vectors.h"


//! A 2x2 Matrix class
/*! This class was more for baricentric coordinates in 2D triangles */
struct Matrix2x2 {

	//! Elements
	/*! The elements of the matrix, stored in column major order */
	double elements[4];

	//! Constructor
	/*! Initializes to the identity */
	Matrix2x2() {
		for(int i = 0; i < 4; i++) {
			elements[i] = i % 2 ? 0 : 1;
		}
	}

	//! Copy constructor
	Matrix2x2(const Matrix2x2& other) {
		for(int i = 0; i < 4; i++) {
			elements[i] = other.elements[i];
		}
	}

	//! [] Operator
	/*! returns a double* to the column specified, if column is out of range it will return column 0 */
	double* operator[](int index) {
		if(index < 0 || index >= 2) {
			return elements;
		}
		return elements + index*2;
	}

	//! Inverse
	/*! Inverse the current matrix */
	void inverse();

	//! Multiply
	/*! Returns a M*V where V is a vector of 2 elements */
	Vector2 operator*(const Vector2& rhs);

	void set(float* v, int cols=2, int rows=2) const;
	void set(double* v, int cols=2, int rows=2) const;
};

//! 3x3 Matrix
/*! A 3x3 Matrix to be used with 2D Transformations */
struct Matrix3x3 {
	//! Elements
	/*! Column major elements */
	double elements[9];

	//! Constructor
	//! Initializes to Identity
	Matrix3x3() {
		for(int r = 0; r < 3; r++) {
			for(int c = 0; c < 3; c++) {
				elements[c*3 + r] = (c != r) ? 0 : 1;
			}
		}
	}

	//! Row
	/*! Returns a Vector3 of the row specified
		\return The specified Row
		*/
	Vector3 row(int r) const;

	//! Column
	/*! Returns a Vector3 of the column specified
		\return The specified Column
		*/
	Vector3 col(int c) const;

	//! From Rotation
	/*! Produces a Matrix3x3 from the rotation specified
		\param raidans Double with the raidans of the rotation
		\param point Vector2 of the point to rotate around, default is the origin
		\return The 3x3 Matrix representing the specified rotation
		*/
	static Matrix3x3 fromRotation(double radians, const Vector2& point=Vector2());

	//! From Translation
	/*! Produces a Matrix3x3 that translates a Vector2
		\param offset Vector2 of the offset you wish to translate by
		\return The 3x3 Matrix representing the specified translation
		*/
	static Matrix3x3 fromTranslation(const Vector2& offset);

	//! From Scale
	/*! Produces a Matrix3x3 from a scale factor and a point
		\param scale Double of the scale factor
		\param point Vector2 of the point to scale from, default is the origin
		\return The matrix representing the scale transformation
		*/
	static Matrix3x3 fromScale(double scale, const Vector2& point=Vector2());

	Matrix3x3 operator+(const Matrix3x3 &rhs) const;
	Matrix3x3& operator+=(const Matrix3x3 &rhs);
	Matrix3x3 operator-(const Matrix3x3 &rhs) const;
	Matrix3x3& operator-=(const Matrix3x3 &rhs);
	Matrix3x3 operator*(const double &rhs) const;

	//! Multiply with a Vector3
	/*! Multiples a Vector3 (column major)
		\return The resulting Vector3
		*/
	Vector3 operator*(const Vector3& rhs) const;

	//! Multiply with another Matrix3x3
	/*! Multiplies with another Matrix3x3
		\return The resulting Matrix3x3
		*/
	Matrix3x3 operator*(const Matrix3x3& rhs) const;

	//! Multiply and save
	/*! Mutiplies this with another Matrix3x3 and sets the result to this
		\return This matrix after multiplication
		*/
	Matrix3x3& operator*=(const Matrix3x3& rhs);

	//! Assignment
	/*! Copies the values from another Matrix3x3
		\return This matrix after the copy
		*/
	Matrix3x3& operator=(const Matrix3x3& rhs) {
		for(int i = 0; i < 9; i++) {
			elements[i] = rhs.elements[i];
		}
		return *this;
	}

	//! index Operator
	/*! returns a double* to the column specified
		\param index Which column you want
		*/
	double* operator[](int index) {
		return elements + (index >= 0 && index <=2 ? index*3 : 0);
	}

	//! determinant
	/*! returns the determinant of the matrix */
	double det() const {
		return elements[0]*elements[4]*elements[8] -
			elements[0]*elements[7]*elements[5] -
			elements[3]*elements[1]*elements[8] +
			elements[3]*elements[7]*elements[2] +
			elements[6]*elements[1]*elements[5] -
			elements[6]*elements[4]*elements[2];
	}

	Matrix3x3& operator*=(const double& rhs) {
		for(int i = 0; i < 9; i++) {
			elements[i] *= rhs;
		}
		return *this;
	}

	//! inverse
	/*! inverses the matrix */
	Matrix3x3& inverse();

	void set(float* v, int cols=3, int rows=3) const;
	void set(double* v, int cols=3, int rows=3) const;
};

//! 4x4 Matrix
/*! A 4x4 Matrix used for 3D Transformations */
struct Matrix4x4 {

	//! Elements
	/*! The elements of the matrix stored in column-major order */
	double elements[16];

	//! Constructor
	/*! Initializes to the identity */
	Matrix4x4() {
		for(int r = 0; r < 4; r++) {
			for(int c = 0; c < 4; c++) {
				elements[r*4 + c] = (c != r) ? 0 : 1;
			}
		}
	}

	//! Copy constructor
	Matrix4x4(const Matrix4x4 &other) {
		for(int i = 0; i < 16; i++) {
			elements[i] = other.elements[i];
		}
	}

	//! index operator
	/*! Returns a double* to the column specified
		\param index integer index for the column
		\return double* to the column specified
		*/
	double* operator[](int index) { //pretend its a double[4][4]
		if(index < 0 || index >= 4)
			return elements;
		return elements + (index*4);
	}

	//! Addition Operator
	Matrix4x4 operator+(const Matrix4x4 &rhs) const;

	//! Addition Equals
	Matrix4x4& operator+=(const Matrix4x4 &rhs);

	//! Subtraction Operator
	Matrix4x4 operator-(const Matrix4x4 &rhs) const;

	//! Subtraction Equals
	Matrix4x4& operator-=(const Matrix4x4 &rhs);

	//! Multiply Operator
	Matrix4x4 operator*(const Matrix4x4 &rhs) const;

	//! Multiply Equals
	Matrix4x4& operator*=(const Matrix4x4 &rhs);

	//! Multiply
	Matrix4x4 operator*(const double &rhs) const;

	//! Multiply Equals
	Matrix4x4& operator*=(const double &rhs);

	//! Multiply With Vector3
	/*! This casts the Vector3 to a Vector4 with d = 1
		\return Vector3 after the multiplication
		*/
	Vector3 operator*(const Vector3& rhs) const;

	//! Divide
	Matrix4x4 operator/(const double &rhs) const;

	//! Divide Equals
	Matrix4x4& operator/=(const double &rhs);

	//! Assignment
	Matrix4x4& operator=(const Matrix4x4 &rhs);

	//! Does the multiply in the other direction, Vector V * Matrix M
	Vector3 transform(const Vector3& vec) const;

	//! Multiply
	Vector4 operator*(const Vector4& rhs) const;

	//! Row
	/*! Gets the row
		\param index The row you want
		\return Vector4 with the row info
		*/
	Vector4 row(const int& index) const;

	//! Column
	/*! Gets the column
		\param index The column you want
		\return Vector4 with the column info
		*/
	Vector4 col(const int& index) const;

	//! Returns a Translation Matrix
	/*! Returns a Matrix4x4 that when multiplied with a Vector3 will produce a Vector3 that's been translated by the offset
		\param translation Vector3 with the offset
		\return A Matrix4x4 of the translation
		*/
	static Matrix4x4 fromTranslation(const Vector3& translation);

	//! Returns a Rotation Matrix
	/*! Takes an axis, offset and radians for a translation
		\param axis Vector3 of the axis of which to rotate around
		\param raidans double of the radians that you want to rotate by
		\param offset Vector3 of the offset of the axis, default is the origin
		\return A Matrix4x4 that when multiplied by a Vector3 V (M*V) will return a Vector3 rotated about the axis specified
		*/
	static Matrix4x4 fromRotation(const Vector3& axis, double radians, const Vector3& offset=Vector3());

	//! Returns a Scale Matrix
	/*! Takes a scale factor and an offset to scale from
		\param scale double containing the scale factor
		\param point Vector3 with the point you want to relatively scale to
		\return A Matrix4x4 that when multiplied with a Vector3 V (M*V) will return a Vector3 scaled relative to the point
		*/
	static Matrix4x4 fromScale(double scale, const Vector3& point=Vector3());

	//! Returns a Scale Matrix
	/*! Takes a scale factor and an offset to scale from
		\param scale vector containing the scale factor for each axis
		\param point Vector3 with the point you want to relatively scale to
		\return A Matrix4x4 that when multiplied with a Vector3 V (M*V) will return a Vector3 scaled relative to the point
		*/
	static Matrix4x4 fromScale(const Vector3& scale, const Vector3& point=Vector3());

	//! Returns an Orthographic Projection Matrix
	/*! Used to create an orthographic projection
		\param left double with the minX
		\param right double with the maxX
		\param bottom double with the minY
		\param top double with the maxY
		\param near double with the minZ
		\param far double with the maxZ
		\return A Matrix4x4 that when multiplied with a vector returns a new point relative to 0,0 and between (-1,1) for all values if they're in the cube
		*/
	static Matrix4x4 fromOrtho(double left, double right, double bottom, double top, double near, double far);

	//! Returns a Perspective Projection Matrix
	/*! Used to create a Perspective projection
		\param fov Double containing the radians of the field of view in the y direction
		\param aspect Double containing the aspect ratio (x/y) of the screen
		\param zNear Double containing the distance to the near plane
		\param zFar Double containing the distance to the far plane
		\return A Matrix4x4 that does something, not entirely sure this one works properly
		*/
	static Matrix4x4 fromPerspective(double fov, double aspect, double zNear, double zFar);

	//! Creates a Modelview Matrix
	/*! Makes a modelview matrix based on camera position, focus postion, and the camera's up vector
		\param camera Vector3 with the camera position
		\param focus Vector3 of the focus position
		\param up Vector3 of the up vector
		\return A Matrix4x4 that when multiplied with a point returns that point in eye coordinates
		*/
	static Matrix4x4 fromLookAt(const Vector3& camera, const Vector3& focus, const Vector3& up);
	static Matrix4x4 fromScreen(double xMin, double xMax, double yMin, double yMax);
	static Matrix4x4 CatmullRom(const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector3& p4, double tau);
	static Matrix4x4 fromFrustrum(double left, double right, double bottom, double top, double near, double far);

	void transpose();

	//! Print Matrix Values to stderr
	void print() const;

	//! determinant
	/*! returns the determinant of the matrix */
	//http://www.cvl.iis.u-tokyo.ac.jp/~miyazaki/tech/teche23.html
	double det() const;

	//! inverse
	/*! inverses the matrix */
	Matrix4x4& inverse();

	//! Returns the elemnents
	operator double*();

#ifdef QT_CORE_LIB
	//! NYI
	QDataStream& operator<<(QDataStream& lhs);

	//! NYI
	QDataStream& operator>>(QDataStream& lhs);

	//! saves to an io device
	void save(QIODevice& io);
	void load(QIODevice& io);
#endif


	void set(float* v, int cols=4, int rows=4) const;
	void set(double* v, int cols=4, int rows=4) const;

	const static Matrix4x4 identity;
};



#endif
