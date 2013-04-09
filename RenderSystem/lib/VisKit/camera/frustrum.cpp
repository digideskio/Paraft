#include "frustrum.h"
#include <cmath>
#include "camera.h"

#define PIOVER180 0.017453293
inline double deg2rad(double deg) {
	return deg*PIOVER180;
}

void Frustrum::update() {
	setPlanes();
	doTransform();
}

void Frustrum::setPlanes() {
	n_point = Vector3(0,0,-cam->nearClip());
	n_normal = Vector3(0,0,-1);
	
	f_point = Vector3(0,0,-cam->farClip());
	f_normal = -n_normal;
	
	Vector3 temp(-1, 0, 0);
	
	t_point = Vector3(0, cam->Height(), -cam->farClip());
	t_normal = temp*t_point;
	t_normal.normalize();
	
	temp = Vector3(1,0,0);
	b_point = Vector3(0, -cam->Height(), -cam->farClip());
	b_normal = temp*b_point;
	b_normal.normalize();
	
	temp = Vector3(0,-1, 0);
	r_point = Vector3(cam->Width(), 0, -cam->farClip());
	r_normal = temp*r_point;
	r_normal.normalize();
	
	temp = Vector3(0,1,0);
	l_point = Vector3(-cam->Width(), 0, -cam->farClip());
	l_normal = temp*l_point;
	l_normal.normalize();
}

void Frustrum::doTransform() {
	n_point = n_point*cam->getTransform();
	n_normal = n_normal*cam->getTransform();
	
	t_point = t_point*cam->getTransform();
	t_normal = t_normal*cam->getTransform();
	
	b_point = b_point*cam->getTransform();
	b_normal = b_normal*cam->getTransform();
	
	f_point = f_point*cam->getTransform();
	f_normal = f_normal*cam->getTransform();
	
	r_point = r_point*cam->getTransform();
	r_normal = r_normal*cam->getTransform();
	
	l_point = l_point*cam->getTransform();
	l_normal = l_normal*cam->getTransform();
}

bool Frustrum::inside(const Vector3 &) const {
	return true; //not really working yet
	/*
	return (n_normal.dot(point - n_point) > 0) &&
			(f_normal.dot(point - f_point) > 0) &&
			(t_normal.dot(point - t_point) > 0) &&
			(b_normal.dot(point - b_point) > 0) &&
			(r_normal.dot(point - r_point) > 0) &&
			(l_normal.dot(point - l_point) > 0); */
}
