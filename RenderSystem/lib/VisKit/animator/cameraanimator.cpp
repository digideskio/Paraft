#include <GL/glew.h>
#include "cameraanimator.h"
#include "vectors.h"
#include "matrices.h"
#include "quaternion.h"
#include "cameraoptions.h"

#include <cmath>
#include <QIODevice>

#define campos(a) ((a)->options.l + (a)->options.c*(a)->options.dist)
#define clamp(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) : (a))

void CameraAnimator::push_back(const CameraOptions& options, double t) {
	if(t <= 0) {
		t = defaultt;
	}
	Animator::push_back(new CameraKeyFrame(this, options, t));
}

void CameraAnimator::push_back(const Camera& cam, double t) {
	if(t <= 0) {
		t = defaultt;
	}
	CameraOptions options;
	cam.saveOptions(options);
	push_back(options, t);
}

void CameraAnimator::push_back(double t) {
	if(t <= 0) {
		t = defaultt;
	}
	if(cam)
		push_back(*cam, t);
}

void CameraAnimator::insert(KeyFrame* next, double t) {
	if(t <= 0) {
		t = defaultt;
	}
	if(cam) {
		CameraOptions options;
		cam->saveOptions(options);
		insertKeyFrame(new CameraKeyFrame(this, options, t), next);
	}
}

void CameraAnimator::drawSpline() {
	glBegin(GL_POINTS);
	for(CameraKeyFrame* k = getHead(); k; k = k->getNext()) {
		if(k == last)
			glColor3f(1,0,0);
		else
			glColor3f(1,1,1);
		glVertex3d(unpack3(campos(k)));
	}
	glEnd();

	if(head == tail)
		return;

	double t;
	Vector4 p;
	Matrix4x4 mat;
	glBegin(GL_LINE_STRIP);
	CameraKeyFrame *k1, *k2, *k3, *k4;
	for(CameraKeyFrame* k = getHead(); k->getNext(); k = k->getNext()) {
		k1 = k->getPrev() ? k->getPrev() : k;
		k2 = k;
		k3 = k->getNext();
		k4 = k3->getNext() ? k3->getNext() : k3;
		mat = Matrix4x4::CatmullRom(campos(k1),
									campos(k2),
									campos(k3),
									campos(k4),
									tau);
		for(int i = 0; i < 50; i++) {
			t = i/49.;
			p = mat*Vector4::CatmullRom(t);
			glVertex3d(unpack3(p));
		}
	}
	glEnd();
	
	glColor3f(0.3f, 0.3f, 0.3f);
	glBegin(GL_LINE_STRIP);
	for(CameraKeyFrame* k = getHead(); k->getNext(); k = k->getNext()) {
		k1 = k->getPrev() ? k->getPrev() : k;
		k2 = k;
		k3 = k->getNext();
		k4 = k3->getNext() ? k3->getNext() : k3;
		mat = Matrix4x4::CatmullRom(k1->options.l,
									k2->options.l,
									k3->options.l,
									k4->options.l,
									tau);
		for(int i = 0; i < 50; i++) {
			t = i/49.;
			p = mat*Vector4::CatmullRom(t);
			glVertex3d(unpack3(p));
		}
	}
	glEnd();
}

void CameraAnimator::set(double t) {
	if(!cam)
		return;
	if(!head)
		return;
	if(head == tail) {
		cam->loadOptions(getHead()->options);
		return;
	}
	if(t > totaltime) {
		cam->loadOptions(getTail()->options);
		return;
	}
	CameraOptions options;
	CameraKeyFrame *k1, *k2, *k3, *k4;
	k2 = reinterpret_cast<CameraKeyFrame*>(findFrame(t));
	if(k2 == getTail()) {
		cam->loadOptions(getTail()->options);
		return;
		
	}
	t = (t - k2->getStartTime());
	if(t < k2->getPause()) {
		cam->loadOptions(k2->options);
		return;
	}
	t = (t - k2->getPause())/(k2->getTime());
	last = k2;
	k1 = k2->getPrev() ? k2->getPrev() : k2;
	k3 = k2->getNext();
	k4 = k3->getNext() ? k3->getNext() : k3;

	Matrix4x4 mat = Matrix4x4::CatmullRom(campos(k1),
								campos(k2),
								campos(k3),
								campos(k4),
								tau);

	Matrix4x4 mat2 = Matrix4x4::CatmullRom(k1->options.l,
								k2->options.l,
								k3->options.l,
								k4->options.l,
								tau);

	Vector3 campos = mat*Vector4::CatmullRom(t);
	Vector3 lpos = mat2*Vector4::CatmullRom(t);

	options = k2->options;

	options.l = lpos;
	options.dist = (campos - lpos).length();
	//options.dist = steps[j].options.dist + t*(steps[j + 1].options.dist - steps[j].options.dist);
	options.c = campos - lpos;
	//options.c = Quaternion::slerp(steps[j].options.c, steps[j + 1].options.c, t);
	options.c.normalize();
	options.u = Quaternion::slerp(k2->options.u, k3->options.u, t);
	options.o = options.u*options.c;
	
	options.fov = k2->options.fov + t*(k3->options.fov - k2->options.fov);

	cam->loadOptions(options);

}

void CameraAnimator::drawCamera() const {
	if(!cam)
		return;

	glBegin(GL_LINES);
	glColor3f(1,1,0);
	glVertex3d(unpack3(cam->l));
	glVertex3d(unpack3(cam->l + cam->c*cam->dist));

	glColor3f(1,0,0);
	glVertex3d(unpack3(cam->l));
	glVertex3d(unpack3(cam->l + cam->u));

	glColor3f(0,0,1);
	glVertex3d(unpack3(cam->l));
	glVertex3d(unpack3(cam->l + cam->o));
	glEnd();

	glBegin(GL_POINTS);
	glColor3f(0, 1, 0);
	glVertex3d(unpack3(cam->l));
	glColor3f(1, 1, 0);
	glVertex3d(unpack3(cam->l + cam->c*cam->dist));
	glEnd();

	glColor3f(1,1,1);
}

void CameraKeyFrame::save(QIODevice* d) const {
	KeyFrame::save(d);
	d->write((char*)&options, sizeof(CameraOptions));
}

void CameraKeyFrame::read(QIODevice* d) {
	KeyFrame::read(d);
	d->read((char*)&options, sizeof(CameraOptions));
}

void CameraAnimator::read(QIODevice *d) {
	if(head)
		delete head;
	head = 0;
	tail = 0;

	int n;
	d->read((char*)&n, 4);
	CameraKeyFrame* k;

	CameraOptions o;

	for(int i = 0; i < n; i++) {
		k = new CameraKeyFrame(this, o);
		k->read(d);
		Animator::push_back(k);
	}
}

void CameraAnimator::clear() {
	if(head)
		delete head;
	head = 0;
	tail = 0;
	cur = 0;
}

AnimatorWidget* CameraAnimator::getAnimatorWidget() {
	if(!optionswidget) {
		optionswidget = new CameraAnimatorWidget(this);
	}
	return optionswidget;
}


