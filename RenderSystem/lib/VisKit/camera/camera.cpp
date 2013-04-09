#ifdef _OPENGL_CAMERA
#include <GL/glew.h>
#endif
#ifdef _WIN32
#include <float.h>
#endif
#include "camera.h"
#include "quaternion.h"

#ifdef QT_CORE_LIB
#include <QMouseEvent>
#include <QDataStream>
#endif
#include <cmath>


#define PIOVER180 0.017453293
#define ONE80OVERPI 57.295779506

#ifdef QT_CORE_LIB
Camera::Camera(QObject* parent):QObject(parent),
#else
Camera::Camera():
#endif
	l(0.5, 0.5, 0.5),
	c(0, 0, 1),
	u(0, 1, 0),
	o(1, 0, 0),
	//freeeye(0, 0, 0),
	//freelook(0, 0, -1),
	//freeup(0, 1, 0),
	//frustrum(this),
	projection(Perspective),
	mode(Fixed),
	uplock(false) {
	dist = 1;
	w = 1;
	h = 1;
	defaultStep = 0.001;
	nearclip = 0.001;
	farclip = 2;
	fov = 45;
	movd = true;
	//zlock = false;
	maxDist = -1;
	view = Matrix4x4::fromLookAt(Vector3(), -c, u);
}

void Camera::setWH(int width, int height) {
	w = width/2.;
	h = height/2.;
}

void Camera::setFov(double f) {
	fov = f;
}

void Camera::setPerspective(double f, int width, int height) {
	setWH(width, height);
	setFov(f);
	setPerspective();
}

void Camera::changeFocus(double x, double y, double z) {
	Vector3 campos = l + c*dist;
	l.x() = x;
	l.y() = y;
	l.z() = z;
	c = campos - l;
	dist = c.length();
	c.normalize();

#ifdef QT_CORE_LIB
	emit updated();
#endif
}

void Camera::setPerspective() {
	tanfov = tan(fov*PIOVER180*0.5);
	scale = 2*tanfov/h;
	//hoffset = tanfov*nearclip;
	//woffset = hoffset*w/h;
}
#ifdef QT_CORE_LIB
void Camera::start(QMouseEvent* e) {
	start(Vector2(e->x(), e->y()));
}
#endif
void Camera::start(const Vector2 &e) {
	prev = e;
}

/*
Quaternion Camera::getTransform(QMouseEvent* e, bool lock, bool flip) {
	return getTransform(Vector2(e->x(), e->y()), lock, flip);
}
*/

Quaternion Camera::getTransform(const Vector2& e, bool lock, bool flip) {
	Vector3 f;
	if(!uplock)
		f = Vector3(prev.x() - w, h - prev.y(), 0);
	else
		f = Vector3(0, 0, 0);
	Vector3 t;
	Vector3 x;
	if(!uplock)
		x = Vector3(e.x() - w, h - e.y(), 0);
	else
		x = Vector3(e.x() - prev.x(), e.y() - prev.y(), 0);
	double len;

	f *= scale;


	int z = 1;
	if(!lock) {
		while(f.lengthSquared() > 1) {
			len = 2./f.length();
			t = f*len;
			f = t - f;
			z *= -1;
		}
		f.z() = z*sqrt(1 - f.lengthSquared());
	} else {
		if(f.lengthSquared() > 1) {
			f.normalize();
		} else {
			f.z() = sqrt(1 - f.lengthSquared());
		}
	}


	x *= scale;
	z = 1;


	if(!lock) {
		while(x.lengthSquared() > 1) {
			len = 2./x.length();
			t = x*len;
			x = t - x;
			z *= -1;
		}
		x.z() = z*sqrt(1 - x.lengthSquared());
	} else {
		if(x.lengthSquared() > 1) {
			x.normalize();
		} else {
			x.z() = sqrt(1 - x.lengthSquared());
		}
	}

	//qDebug("%f, %f, %f - %f, %f, %f", f.x(), f.y(), f.z(), x.x(), x.y(), x.z());

	double dot = f.dot(x);

	f *= x;
	f *= view;
	//f = o*f.x() + u*f.y() + c*f.z();

	if(dot == 1) return Quaternion(); //initializes to identity

	return Quaternion(flip ? -f : f, acos(dot));
}

Quaternion Camera::getFreeXTransform(const Vector2& e, bool, bool) {
	// rotate along freeup vector if moving right
	// rotate along -freeup vector if moving left
	double angle = -(e.x()-prev.x()) *0.1* PIOVER180;/// 2 / 180.0 * 3.14159265;
	//double residual = sqrt(1 - cos(angle) * cos(angle));
	//double direction = e.x() - prev.x() > 0 ? 1.0 : -1.0;
	//double x = direction * freeup.x() * residual;
	//double y = direction * freeup.y() * residual;
	//double z = direction * freeup.z() * residual;
	return Quaternion(u, angle);
}

Quaternion Camera::getFreeYTransform(const Vector2& e, bool, bool) {
	// rotate along freeright vector if moving up
	// rotate along -freeright vector if moving down
	//Vector3 freeright = freelook * freeup;
	//double angle = (e.y()-prev.y()) / 2 / 180.0 * 3.14159265;
	double angle = -(e.y()-prev.y()) *0.1* PIOVER180;/// 2 / 180.0 * 3.14159265;
	//double residual = sqrt(1 - cos(angle) * cos(angle));
	//double direction = e.y() - prev.y() > 0 ? 1.0 : -1.0;
	//double x = direction * freeright.x() * residual;
	//double y = direction * freeright.y() * residual;
	//double z = direction * freeright.z() * residual;
	return Quaternion(o, angle);
}

void Camera::setProjection(ProjectionType p) {
	projection = p;
}

#ifdef QT_CORE_LIB
void Camera::track(QMouseEvent* e, bool lock) {
	track(Vector2(e->x(), e->y()), lock);
}
#endif

void Camera::track(const Vector2& e, bool lock) {
	if(e == prev)
		return;

	//qDebug("%f %f", e.x(), e.y());
	if(mode == Fixed) {
		view = Matrix4x4::fromLookAt(Vector3::Zero, -c, u);

		Quaternion q = getTransform(e, lock, true);


		q.rotateVector(c);
		//if(!uplock)
			q.rotateVector(u);

		o = u*c;
		view = Matrix4x4::fromLookAt(Vector3::Zero, -c, u);
		movd = true;
		prev = e;

#ifdef QT_CORE_LIB
		emit updated();
#endif
	} else if (mode == Dolly) {
		double dx = (prev.x() - e.x())*dist/((w < h ? w : h));
		double dy = (e.y() - prev.y())*dist/((w < h ? w : h));
		prev = e;

		l += o*dx + u*dy;
	} else if (mode == Free) {
			Quaternion qx = getFreeXTransform(e);
			Quaternion qy = getFreeYTransform(e);
			Vector3 p = l + c*dist;
			Vector3 tc = -c;
			qx.rotateVector(tc);
			qy.rotateVector(u);
			qy.rotateVector(tc);
			tc.normalize();
			u.normalize();
			l = p + tc*dist;
			c = -tc;
			o = u*c;
			//qDebug("(%f, %f), (%f %f %f), (%f %f %f), (%f %f %f)",
			//	unpack2(e), unpack3(l), unpack3(c), unpack3(o));
			//freelook.normalize();
			//freeup.normalize();
			view = Matrix4x4::fromLookAt(Vector3::Zero, -c, u);
	/*	Quaternion qx = getFreeXTransform(e, lock, true);
		Quaternion qy = getFreeYTransform(e, lock, true);
		qx.rotateVector(freelook);
		qy.rotateVector(freeup);
		qy.rotateVector(freelook);
		freelook.normalize();
		freeup.normalize();
		view = Matrix4x4::fromLookAt(freeeye, freelook, freeup); */
		movd = true;
		prev = e;
	}
}

#ifdef QT_CORE_LIB
void Camera::look(QMouseEvent* e) {
	look(Vector2(e->x(), e->y()));
}
#endif
void Camera::look(const Vector2& e) {
	if(e == prev)
		return;

	//if(mode == Fixed) {
		/*if(!uplock) {
			Quaternion q = getTransform(e);
			Vector3 t = -c;

			q.rotateVector(t);

			o = c*dist + l;
			c = -t;

			q.rotateVector(u);

			l = t*dist + o;
			o = u*c;

			view = Matrix4x4::fromLookAt(Vector3::Zero, -c, u);
			prev = e;
		} else { */
			Quaternion qx = getFreeXTransform(e);
			Quaternion qy = getFreeYTransform(e);
			Vector3 p = l + c*dist;
			Vector3 tc = -c;
			qx.rotateVector(tc);
			qy.rotateVector(u);
			qy.rotateVector(tc);
			tc.normalize();
			u.normalize();
			l = p + tc*dist;
			c = -tc;
			o = u*c;
			//freelook.normalize();
			//freeup.normalize();
			view = Matrix4x4::fromLookAt(Vector3::Zero, -c, u);
			movd = true;
			prev = e;
		//}
#ifdef QT_CORE_LIB
		emit updated();
#endif
	//}
}

void Camera::updateCamera() {
#ifdef _OPENGL_CAMERA
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if(projection == Perspective)
		gluPerspective(fov, (double)w/(double)h, nearclip, farclip);
	else if(projection == Ortho) {
		//if(mode == Fixed)
			glOrtho(-dist * w / (w < h ? w : h),dist  * w / (w < h ? w : h),
				-dist  * h / (w < h ? w : h),dist * h / (w < h ? w : h),nearclip,farclip);
		//else
		//	glOrtho(l.x() -dist * w / (w < h ? w : h), l.x() + dist  * w / (w < h ? w : h),
		//		l.y() -dist  * h / (w < h ? w : h), l.y() + dist * h / (w < h ? w : h),nearclip,farclip);
	}
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
#ifdef _WIN32
	if(_isnan(c.x())) {
#else
	if(std::isnan(c.x())) {
#endif

		c.x() = 0; c.y() = 0; c.z() = 1;
		u.x() = 0; u.y() = 1; u.z() = 0;
		o.x() = 1; o.y() = 0; o.z() = 0;
	}
	//frustrum.update();
	Vector3 cam = c*dist + l;
	//if(mode == Fixed || mode == Free)
		gluLookAt(cam.x(), cam.y(), cam.z(), l.x(), l.y(), l.z(), u.x(), u.y(), u.z());
	//else if(mode == Free)
	//	gluLookAt(freeeye.x(), freeeye.y(), freeeye.z(), freeeye.x()+freelook.x(), freeeye.y()+freelook.y(), freeeye.z()+freelook.z(), freeup.x(), freeup.y(), freeup.z());
#else
	qDebug("NO OPENGL");
#endif
}

void Camera::zoom(int delta, bool fovzoom) {
	if(!fovzoom) {
		if(delta > 0) {
			dist *= 0.9;
			if(mode == Fixed && dist < nearclip) dist = nearclip;
		} else {
			dist /= 0.9;
			if(maxDist > 0)
				dist = dist > maxDist ? maxDist : dist;
		}
	} else {
		if(delta > 0) {
			fov *= 0.9;
			if(fov < 5.0) fov = 5.0;
		} else {
			fov /= 0.9;
			if(fov > 90) fov = 90;
		}
		setPerspective();
	}
	movd = true;

#ifdef QT_CORE_LIB
	emit updated();
#endif
	//updateCamera();
}

void Camera::setNearclip(double c) {
	nearclip = c;
	//hoffset = tanfov*nearclip;
	//woffset = hoffset*w/h;
}


void Camera::setCamera(double x, double y, double z) {
	c.x() = x;
	c.y() = y;
	c.z() = z;

}
void Camera::setLook(double x, double y, double z) {
	l.x() = x;
	l.y() = y;
	l.z() = z;
}
void Camera::setUp(double x, double y, double z) {
	u.x() = x;
	u.y() = y;
	u.z() = z;
}
void Camera::setRight(double x, double y, double z) {
	o.x() = x;
	o.y() = y;
	o.z() = z;
}

void Camera::setLookAt(const Vector3 &focus, const Vector3 &campos, const Vector3 &up) {
	l = focus;
	Vector3 offset = campos - focus;
	dist = offset.length();
	offset.normalize();
	c = offset;
	u = up;
	o = u*c;
}

double Camera::distance(double x, double y, double z) const {
	return (x - CamX())*(x - CamX()) +
			(y - CamY())*(y - CamY()) +
			(z - CamZ())*(z - CamZ());
}

void Camera::setDist(double d) {
	dist = d;
}

/*
bool Camera::toggleZLock() {
	zlock = !zlock;
	return zlock;
}
bool Camera::toggleZLock(bool b) {
	zlock = b;
	return zlock;
} */

void Camera::setFarclip(double fclip) {
	farclip  = fclip;
}



#ifdef QT_CORE_LIB
QIODevice& Camera::saveSettings(QIODevice& file) {
	if(!file.isWritable())
		return file;
#else
std::ostream& Camera::saveSettings(std::ostream& file) {
	if(!file.good())
		return file;
#endif
	CameraOptions o;
	saveOptions(o);

	file.write((char*)&o, sizeof(CameraOptions));

	return file;
}
#ifdef QT_CORE_LIB
QIODevice& Camera::loadSettings(QIODevice& file) {
	if(!file.isReadable())
		return file;
#else
std::istream& Camera::loadSettings(std::istream& file) {
	if(!file.good())
		return file;
#endif
	CameraOptions o;
	file.read((char*)&o, sizeof(CameraOptions));
	loadOptions(o);

	setPerspective();
	return file;
}

void Camera::setFocus(double x, double y, double z) {
	l.x() = x;
	l.y() = y;
	l.z() = z;
//	freeeye.x() = x;
//	freeeye.y() = y;
//	freeeye.z() = z * 2 + 1;
//printf("setFocus: %f %f %f\n", x, y, z);
}

void Camera::setMaxDist(double m) {
	maxDist = m;
}

void Camera::freeForward(double t) {
	l -= c*(t ? t : defaultStep);
	//freeeye += freelook*t;
}
void Camera::freeBackward(double t) {
	l += c*(t ? t : defaultStep);
	//freeeye -= freelook*0.001;
}
void Camera::freeStrafeLeft(double t) {
	//Vector3 freeright = freelook * freeup;
	//freeright.normalize();
	//freeeye -= freeright*0.001;
	l -= o*(t ? t : defaultStep);
}
void Camera::freeStrafeRight(double t) {
	//Vector3 freeright = freelook * freeup;
	//freeright.normalize();
	//freeeye += freeright*0.001;
	l += o*(t ? t : defaultStep);
}
void Camera::freeTiltLeft(double t) {
	//double angle = t*PIOVER180;//1.0 / 2.0 / 180.0 * 3.14159265;
	//double residual = sin(angle);//sqrt(1 - cos(angle) * cos(angle));
	//double direction = 1.0;
	//Vector3 axis = -c * sin(angle);
	Quaternion q = Quaternion(c, t*PIOVER180);
	//Quaternion q = Quaternion(axis.x(), axis.y(), axis.z(), cos(angle));
	q.rotateVector(u);
	u.normalize();
	o = (u * c).normalize();
	//freeup.normalize();
	view = Matrix4x4::fromLookAt(l + c*dist, l, u);
	movd = true;
}
void Camera::freeTiltRight(double t) {
	//double angle = 0.5*PIOVER180;//1.0 / 2.0 / 180.0 * 3.14159265;
	//double residual = sin(angle);//sqrt(1 - cos(angle) * cos(angle));
	//double direction = -1.0;
	//Vector3 axis = freelook * direction * residual;
	//Quaternion q = Quaternion(axis.x(), axis.y(), axis.z(), cos(angle));
	//q.rotateVector(freeup);
	//freeup.normalize();
	Quaternion q(-c, t*PIOVER180);
	q.rotateVector(u);
	u.normalize();
	o = (u * c).normalize();
	view = Matrix4x4::fromLookAt(l + c*dist, l, u);
	movd = true;
}
void Camera::freeUp(double t) {
//	freeeye += freeup * 0.001;
	l += u*(t ? t : defaultStep);
}
void Camera::freeDown(double t) {
	//freeeye -= freeup * 0.001;
	l -= u*(t ? t : defaultStep);
}

void Camera::saveOptions(CameraOptions& options) const {
	options.l = l;
	options.c = c;
	options.u = u;
	options.o = o;
	options.dist = dist;
	options.nearclip = nearclip;
	options.farclip = farclip;
	options.fov = fov;
	options.maxDist = maxDist;
	options.mode = mode;
	options.proj = projection;
}

void Camera::loadOptions(const CameraOptions& options) {
	l = options.l;
	c = options.c;
	u = options.u;
	o = options.o;
	dist = options.dist;
	nearclip = options.nearclip;
	farclip = options.farclip;
	fov = options.fov;
	maxDist = options.maxDist;
	mode = options.mode;
	projection = options.proj;
	setPerspective();
}

void Camera::push() {
	CameraOptions options;
	saveOptions(options);
	m_stack.push(options);
}

void Camera::pop() {
	CameraOptions options = m_stack.top();
	m_stack.pop();
	loadOptions(options);
	setPerspective();
}

Vector3 Camera::getRay(const Vector2& e) const {
	Vector3 p;
	if(projection == Perspective) {
		p.x() = 2*(e.x()/(2*w) - 0.5)*tanfov*w/h;
		p.y() = 2*((2*h - e.y())/(2*h) - 0.5)*tanfov;
		p.z() = 1;

		p = o*p.x() + u*p.y() - c*p.z();
		p.normalize();
	} else {
		double f = dist/(w < h ? w : h);
		p.x() = (e.x() - w)*f;
		p.y() = (h - e.y())*f;
		p = o*p.x() + u*p.y() + l;
	}
	return p;
}

void Camera::setMode(CameraMode m) {
	if (mode == Fixed && m == Free) {
/*		freelook = -c * dist;
		freelook.normalize();
		freeeye = l + c * dist;
		freeup = u;
		freeup.normalize();*/
	}
	mode = m;
}

#ifdef QT_CORE_LIB
Vector3 Camera::getRay(QMouseEvent* e) const {
	return getRay(Vector2(e->x(), e->y()));
}
#endif

#ifdef QT_CORE_LIB
QDataStream & operator<<(QDataStream & out, const CameraOptions & co) {
	out << co.l.x() << co.l.y() << co.l.z();
	out << co.c.x() << co.c.y() << co.c.z();
	out << co.u.x() << co.u.y() << co.u.z();
	out << co.o.x() << co.o.y() << co.o.z();
	out << co.dist << co.nearclip << co.farclip << co.fov << co.maxDist;
	switch (co.mode) {
	case Camera::Fixed: out << (quint8)0; break;
	case Camera::Dolly: out << (quint8)1; break;
	case Camera::Free: out << (quint8)2; break;
	}
	switch (co.proj) {
	case Camera::Perspective: out << (quint8)0; break;
	case Camera::Ortho: out << (quint8)1; break;
	}
	return out;
}
QDataStream & operator>>(QDataStream & in, CameraOptions & co) {
	in >> co.l.x() >> co.l.y() >> co.l.z();
	in >> co.c.x() >> co.c.y() >> co.c.z();
	in >> co.u.x() >> co.u.y() >> co.u.z();
	in >> co.o.x() >> co.o.y() >> co.o.z();
	in >> co.dist >> co.nearclip >> co.farclip >> co.fov >> co.maxDist;
	quint8 value;
	in >> value;
	switch (value) {
	case 0: co.mode = Camera::Fixed; break;
	case 1: co.mode = Camera::Dolly; break;
	case 2: co.mode = Camera::Free; break;
	}
	in >> value;
	switch (value) {
	case 0: co.proj = Camera::Perspective; break;
	case 1: co.proj = Camera::Ortho; break;
	}
	return in;
}
#endif
