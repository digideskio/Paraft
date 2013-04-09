#ifndef CAMERA_H_
#define CAMERA_H_
#ifdef QT_CORE_LIB
#include <QObject>
#include <QFile>
#include <QPoint>
#include <QStack>
#else
#include <stack>
#include <iostream>
#endif
#include "quaternion.h"
#include "vectors.h"
#include "matrices.h"
#include "frustrum.h"

struct CameraOptions;
class QMouseEvent;
class Camera

#ifdef QT_CORE_LIB
	: public QObject {
#else
		{
#endif

#ifdef QT_CORE_LIB
	Q_OBJECT
#endif
public:
	//! Projection Type Enums
	enum ProjectionType { Perspective, Ortho };
	enum CameraMode { Fixed, Dolly, Free };
	void set_prev(int i, int j) { prev = Vector2(i,j); }; // tj
protected:

	//! Focus Vector
	/*! The focus of the camera */
	Vector3 l;

	//! Camera Vector
	/*! The direction from the focus the camera is positioned */
	Vector3 c;

	//! Up Vector
	/*! The camera's up vector */
	Vector3 u;

	//! Right Vector
	/*! The vector that's the result of the c*u */
	Vector3 o;

	//! The view matrix
	/*! Used internally for repositioning the camera, not the actual modelview matrix */
	Matrix4x4 view;

	//Frustrum frustrum;

	//! Free Camera eye
	//Vector3 freeeye;
	//Vector3 freelook;
	//Vector3 freeup;


	//! Scale
	/*! Internal scale for the trackball */
	double scale;

	//! Width
	/*! Window width/2 given during setPerspective() */
	double w;

	//! Height
	/*! Window height/2 given during setPerspective() */
	double h;

	//! Field of ViewY (degrees)
	/*! The field of view given in degrees in the Y direction */
	double fov;

	//! tan(fov)
	/*! Tangent of the fov, used internally */
	double tanfov;

	//! Distance
	/*! Distance from the camera to the focus point */
	double dist;

	//! Near Clip
	/*! The distance to the near clipping range */
	double nearclip;

	//! Far Clip
	/*! the distance to the far clipping range */
	double farclip;

	//! Moved
	/*! Whether or not the camera has moved since the last updateCamera() */
	bool movd;

	//! Max Dist
	/*! The maximum distance the camera can be from its focus */
	double maxDist;

	//! Default Step
	//! The default step size
	double defaultStep;

	//! Options Stack
	/*! Used for storing and popping previous options onto the stack, like OpenGL can do with its matrices */

#ifdef QT_CORE_LIB
	QStack<CameraOptions> m_stack;
#else
	std::stack<CameraOptions> m_stack;
#endif



	//Quaternion getTransform(QMouseEvent*, bool=false, bool=false );

	//! Get Transform
	/*! Get's a transformation based on the new mouse position e
	  \param e a Vector2 with the current mouse position
	  \param lock whether or not to lock the trackball to the edges
	  \param flip whether or not we're doing trackball vs. free look
	  */
	Quaternion getTransform(const Vector2& e, bool lock=false, bool flip=false );

	//! Previous Position
	Vector2 prev;

	//! Projection
	/*! Projection type, either ortho or perspective */
	ProjectionType projection;

	//! Mode
	/*! Mode of the camera, as if its fixed or on a dolly */
	CameraMode mode;

	//! UpLock
	/*! Lock the Up Vector */
	bool uplock;


#ifdef QT_CORE_LIB
signals:
		void camUpdate(double, double, double, double, double, double, double, double, double);
		void updated();
#endif

public:

	//! constructor
#ifdef QT_CORE_LIB
	Camera(QObject* parent=0);
#else
	Camera();
#endif

		//! Load Camera Options
	/*! internally loads camera options
		\param o A CameraOptions struct to load options from
	*/
	void loadOptions(const CameraOptions& o);

	//! Save Camera Options
	/*! interally saves camera options
		\param o A CameraOptions struct to save the options to
	*/
	void saveOptions(CameraOptions& o) const;

	//! Track
	/*! Qt Convenience function
	  \param e QMouseEvent* for the current position
	  \param lock Lock the edges of the trackball
	  */

	void track(QMouseEvent* e, bool lock=true);
	//! Track
	/*! Trackball based movement based on the new current position e
	  \param e Vector2 of the current position
	  \param lock Lock the edges of the trackball
	  */
	void track(const Vector2 &e, bool lock=true);

	//! Start
	/*! Qt convenience function
	  \param e QMouseEvent* from mousePress to start tracking/looking
	  */
	void start(QMouseEvent* e);

	//! Start
	/*! Starts tracking/looking
	  \param e Vector2 from mousePress to start tracking/looking
	  */
	void start(const Vector2& e);

	//! Set Projection
	/*! Sets the type of projection to use in updateCamera()
	  \param p ProjectionType (either Camera::Ortho or Camera::Perspective)
	  */
	void setProjection(ProjectionType p);

	//! whether or not the camera has moved since it was last checked
	bool moved() const { return movd; }

	//! done setting up stuff after moving, set moved to false
	void done() { movd = false; }

	//! Mouse Look
	/*! Convenience Qt function
	  \param e QMouseEvent* for the current event
	  */
	void look(QMouseEvent* e);

	//! Mouse Look
	/*! Move based on the new position
	  \param e Vector2 of the new position
	  */
	void look(const Vector2& e);

	//bool toggleZLock(bool);
	//bool toggleZLock();

	//! sets up the camera using gluLookAt()
	void updateCamera();

	/*! sets the distance of the camera
	 * \param delta how far the mousewheel has scrolled
	 */
	void zoom(int, bool fovzoom=false);

	/*! sets the look at point of this camera
	 * \param x the x coordinate of where we're looking
	 * \param y the y coordinate of where we're looking
	 * \param x the z coordinate of where we're looking
	 */
	void setFocus(double x, double y, double z);

	/*! control freeeye movement
	 */
	void freeForward(double t=0);
	void freeBackward(double t=0);
	void freeStrafeLeft(double t=0);
	void freeStrafeRight(double t=0);
	void freeTiltLeft(double t=1);
	void freeTiltRight(double t=1);
	void freeUp(double t=0);
	void freeDown(double t=0);
	Quaternion getFreeXTransform(const Vector2& e, bool lock=false, bool flip=false );
	Quaternion getFreeYTransform(const Vector2& e, bool lock=false, bool flip=false );
	Vector3 getFreeEye() const {
		return getCamPosition();
	}

	void setDefaultStep(double t) {
		defaultStep = t;
	}

	void setWH(int, int);

	/*! sets the perspective info
	 * \param f the new fov
	 * \param width the new width
	 * \param height the new height
	 */
	void setPerspective(double, int, int);

	//! sets up all the internal viewport stuff
	void setPerspective();

	/*! sets the clip distance
	 * \param c the clip distance
	 */
	void setNearclip(double);

	//! returns the up vector x value
	double UpX() const { return u.x(); }

	//! returns the up vector y value
	double UpY() const { return u.y(); }

	//! returns the up vector z value
	double UpZ() const { return u.z(); }

	//! returns the reverse look vector x value
	double X() const { return c.x(); }

	//! returns the reverse look vector y value
	double Y() const { return c.y(); }

	//! returns the reverse look vector z value
	double Z() const { return c.z(); }

	//! returns the right vector x value
	double oX() const { return o.x(); }

	//! returns the right value y value
	double oY() const { return o.y(); }

	//! returns the right vector z value
	double oZ() const { return o.z(); }

	//! returns the point we're looking at's x value
	double lX() const { return l.x(); }

	//! returns the point we're looking at's y value
	double lY() const { return l.y(); }

	//! returns the point we're looking at's z value
	double lZ() const { return l.z(); }

	//! returns the distance from the camera to the focus point
	double Dist() const { return dist; }

	//! returns the distance between the clipping plane and the focus point
	double nearClip() const { return nearclip; }

	double farClip() const { return farclip; }

	//! returns the location of the camera's x value
	double CamX() const { return c.x()*dist + l.x(); }

	//! returns the location of the camera's y value
	double CamY() const { return c.y()*dist + l.y(); }

	//! returns the location of the camera's z value
	double CamZ() const { return c.z()*dist + l.z(); }

	double Width() const { return w; }
	double Height() const { return h; }

	/*! returns the distance from the point x,y,z to the camera
	 * \param x the x coordinate
	 * \param y the y coordinate
	 * \param z the z coordinate
	 */
	double distance(double x, double y, double z) const;

	//! returns the fov
	double Fov() const { return fov; }

	void setFov(double f);

	/*! sets the camera offset from the focus
	 * \param x the x offset
	 * \param y the y offset
	 * \param z the z offset
	 */
	void setCamera(double,double,double);

	/*! sets sets the focus point
	 * \param x the x value
	 * \param y the y value
	 * \param z the z value
	 */
	void setLook(double,double,double);

	/*! sets the camera's up vector
	 * \param x the x value
	 * \param y the y value
	 * \param z the z value
	 */

	void setUp(double,double,double);

	void setLookAt(const Vector3& focus, const Vector3& campos, const Vector3& up);

	/*! sets the camera's right vector
	 * \param x the x value
	 * \param y the y value
	 * \param z the z value
	 */
	void setRight(double,double,double);

	/*! sets teh camera's distance
	 * \param d the distance
	 */
	void setDist(double);

	/*! sets the farclip distance
	 * \param fclip the farclip distance
	 */
	void setFarclip(double);

	//! Set Max Distance
	/*! Sets the maximum distance the camera can be from the focus
	  \param distance a type double containing the max distance
	  */
	void setMaxDist(double);

	//! Get Focus Vector
	Vector3& getLook() {
		return l;
	}

	//! Get Camera Offset Vector
	Vector3& getCam() {
		return c;
	}

	//! Get Right Vector
	Vector3& getRight() {
		return o;
	}

	//! Get Up Vector
	Vector3& getUp() {
		return u;
	}

	const Vector3& getLook() const {
		return l;
	}
	Vector3 getCam() const {
		//if (mode == Free) return freelook * -1;
		return c;
	}
	Vector3 getRight() const {
		//if (mode == Free) return freelook * freeup;
		return o;
	}
	Vector3 getUp() const {
		//if (mode == Free) return freeup;
		return u;
	}

	//! Get the Look Vector
	Vector3 getLookVector() const {
		return l - c*dist;
	}

	//! Get the Camera Position
	Vector3 getCamPosition() const {
		//if (mode == Free) return freeeye;
		return l + c*dist;
	}

	//! Get the modelview matrix
	Matrix4x4 getTransform() const {
		return Matrix4x4::fromLookAt(getCamPosition(), getLook(), getUp());
	}

	Matrix4x4 getProjectionMatrix() const {
		if(projection == Ortho) {
			return Matrix4x4::fromOrtho(-dist * w / (w < h ? w : h),dist  * w / (w < h ? w : h),
				-dist  * h / (w < h ? w : h),dist * h / (w < h ? w : h),nearclip,farclip);
			//if(mode == Fixed)
			//	return Matrix4x4::fromOrtho(-dist * w / (w < h ? w : h),dist  * w / (w < h ? w : h),
			//		-dist  * h / (w < h ? w : h),dist * h / (w < h ? w : h),-1,1);
			//else
			//	return Matrix4x4::fromOrtho(l.x() -dist * w / (w < h ? w : h), l.x() + dist  * w / (w < h ? w : h),
			//		l.y() -dist  * h / (w < h ? w : h), l.y() + dist * h / (w < h ? w : h),-1,1);
		}
		return Matrix4x4::fromPerspective(fov*0.0174532925, w/h, nearclip, farclip);
	}

	//! Change the Focus Point
	/*! Changes the camera's focus to the new point
	  \param x double containing the x coordinate
	  \param y double containing the y coordinate
	  \param z double containing the z coordinate
	  */
	void changeFocus(double x, double y, double z); //changes the focus without 'moving' the camera

	//! Save settings
	/*! Save the settings to a file
	  \param file QFile to save to
	  */
#ifdef QT_CORE_LIB
	QIODevice& saveSettings(QIODevice&);
#else
	std::ostream& saveSettings(std::ostream&);
#endif

	//! Load settings
	/*! Load the settings from a file
	  \param file QFile to load from
	  */
#ifdef QT_CORE_LIB
	QIODevice& loadSettings(QIODevice&);
#else
	std::istream& loadSettings(std::istream&);
#endif

	//! Push Settings
	/*! Pushes current settings onto the settings stack */
	void push();

	//! Pop Settings
	/*! Pops the previous settings off the stack and them to the current settings */
	void pop();


	//bool inside(const Vector3& point) const;


	//! Get Projection
	/*! Gets the current projection type */
	ProjectionType getProjection() const { return projection; }

	//! Get Ray
	/*! Gets a ray based on a screen coordinate
		In Ortho mode, gets the world coordinate at the focus distance */
	Vector3 getRay(const Vector2& e) const;


	//! Get Ray
	/*! Qt convenience function */
	Vector3 getRay(QMouseEvent* e) const;

	//! Get Mode
	/*! Gets the camera mode */
	CameraMode getMode() { return mode; }

	//! Set Mode
	/*! Sets the camera mode */
	void setMode(CameraMode m);


	//! Set Up Lock
	/*! Sets whether or not the up vector is locked */
	void setUpLock(bool l) { uplock = l; }

	friend class CameraAnimator;
};
struct CameraOptions {
	CameraOptions() {
		l = Vector3(0.5, 0.5, 0.5);
		c = Vector3(0.0, 0.0, 1.0);
		u = Vector3(0.0, 1.0, 0.0);
		o = Vector3(1.0, 0.0, 0.0);
		dist = 3.0;
		nearclip = 0.01;
		farclip = 10.0;
		fov = 45.0;
		maxDist = -1.0;
		mode = Camera::Fixed;
		proj = Camera::Perspective;
	}
	CameraOptions(const CameraOptions & co) {
		(*this) = co;
	}
	CameraOptions & operator=(const CameraOptions & src) {
		this->l = src.l;
		this->c = src.c;
		this->o = src.o;
		this->u = src.u;
		this->scale = src.scale;
		this->w = src.w;
		this->h = src.h;
		this->fov = src.fov;
		this->tanfov = src.tanfov;
		this->dist = src.dist;
		this->hoffset = src.hoffset;
		this->woffset = src.woffset;
		this->nearclip = src.nearclip;
		this->farclip = src.farclip;
		this->ofov = src.ofov;
		this->movd = src.movd;
		this->maxDist = src.maxDist;
		this->mode = src.mode;
		this->proj = src.proj;
		return *this;
	}
	bool operator==(const CameraOptions & src) const {
		if (l == src.l &&
		    c == src.c &&
			o == src.o &&
			u == src.u &&
			dist == src.dist &&
			nearclip == src.nearclip &&
			farclip == src.farclip &&
			fov == src.fov &&
			maxDist == src.maxDist &&
			mode == src.mode &&
			proj == src.proj) return true;
		return false;
	}
	Vector3 l;
	Vector3 c;
	Vector3 o;
	Vector3 u;
	/*
	double l_x, l_y, l_z;
	double c_x, c_y, c_z;
	double u_x, u_y, u_z;
	double o_x, o_y, o_z;
	*/
	double scale;
	double w;
	double h;
	double fov; //degrees
	double tanfov;
	double dist;
	double hoffset, woffset;
	double nearclip;
	double farclip;
	double ofov;
	bool movd;
	//bool zlock;
	double maxDist;
	Camera::CameraMode mode;
	Camera::ProjectionType proj;
};

#ifdef QT_CORE_LIB
class QDataStream;
QDataStream & operator<<(QDataStream &, const CameraOptions &);
QDataStream & operator>>(QDataStream &, CameraOptions &);
#endif

#endif /*CAMERA_H_*/
