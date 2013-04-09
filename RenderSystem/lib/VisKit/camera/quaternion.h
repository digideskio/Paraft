#ifndef QUATERNION_H_
#define QUATERNION_H_

#include "vectors.h"
#include "matrices.h"

/*! basic quaternion class
 * very simple quaternion class, the bare minimum needed for rotating vectors
 */
class Quaternion {

	double elements[4];

	//! normalizes the quaternion length to 1
	void normalize();

	//! get's the quaternion's conjugate
	Quaternion getConjugate() const;

	public:

	//! constructor
	Quaternion();
	
	Quaternion(const Vector3& vec, double angle);
	Quaternion(const Quaternion& other);

	/*! constructor
	 * \param x the x component
	 * \param y the y component
	 * \param z the z component
	 * \param w the w component
	 */
	Quaternion(double x, double y, double z, double w);

	/*! quaternion multiplication
	 * \param rhs the right hand side, this quaternion being the left
	 */
	Quaternion operator*(const Quaternion & rhs) const;

	/*! build a quaternion from an xyz axis and an angle
	 * \param dx the x component of the axis
	 * \param dy the y component of the axis
	 * \param dz the z component of the axis
	 * \param angle the angle on which to rotate stuff by
	 */
	void fromAxis(double x, double y, double z, double angle);
	void fromAxis(const Vector3& axis, double angle);

	/*! rotates a 3d vector by the quaternion
	 * \param vx the vector's x component
	 * \param vy the vector's y component
	 * \param vz the vector's z component
	 */
	void rotateVector(double &vx, double &vy, double &vz);
	void rotateVector(Vector3& vector);

	/*! assignment operator
	 * \param rhs the right hand side
	 */
	Quaternion& operator=(const Quaternion & rhs);
	
	Quaternion& operator*=(const double& rhs);
	Quaternion& operator*=(const Quaternion & rhs);
	
	double lengthSquared() { return elements[0]*elements[0] + 
		elements[1]*elements[1] + 
		elements[2]*elements[2] + 
		elements[3]*elements[3]; }
	Quaternion operator+(const Quaternion& rhs) const;
	Quaternion operator*(double rhs) const;
	Quaternion operator/(double rhs) const;
	double length();
	
	operator Matrix4x4() const;
	operator double*() const;

	static Vector3 slerp(const Vector3& first, const Vector3& second, double time);
	static Quaternion slerp(const Quaternion& q1, const Quaternion& q2, double t);
	//GLfloat* operator*(const GLfloat* rhs) const;
};

#endif /*QUATERNION_H_*/
