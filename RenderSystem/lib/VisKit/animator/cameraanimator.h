#ifndef _CAMERAANIMATOR_H_
#define _CAMERAANIMATOR_H_

#include "camera.h"
#include "animator.h"

#include <QList>

class CameraAnimator;
class CameraKeyFrameListItem;
class CameraKeyFrame : public KeyFrame {
	CameraOptions options;

	CameraKeyFrame(Animator* parent, const CameraOptions& options, double t=1, CameraKeyFrame* prev=0, CameraKeyFrame* next=0)
		:KeyFrame(parent, t, prev, next), options(options) {}
	friend class CameraAnimator;
public:
	CameraKeyFrame* getNext() { return reinterpret_cast<CameraKeyFrame*>(next); }
	CameraKeyFrame* getPrev() { return reinterpret_cast<CameraKeyFrame*>(prev); }
	CameraOptions& getOptions() { return options; }
	void read(QIODevice* d);
	void save(QIODevice* d) const;
};

class CameraAnimatorWidget;
class CameraAnimator : public Animator {
	Q_OBJECT

	double tau;
	Camera* cam;
	KeyFrame* last;
	CameraAnimatorWidget* optionswidget;
public:
	CameraAnimator(double tau=0.2):tau(tau), cam(0), optionswidget(0) {} 
	void push_back(const CameraOptions& step, double t=0);
	void push_back(const Camera& cam, double t=0);
	void push_back(double t=0);
	void insert(KeyFrame* next, double t=0);
	void setCam(Camera* c) { cam = c; }
	void drawSpline();
	void drawCamera() const;
	void setTau(double t) { tau = t; }
	double getTau() const { return tau; }
	CameraKeyFrame* getHead() { return reinterpret_cast<CameraKeyFrame*>(head); }
	CameraKeyFrame* getTail() { return reinterpret_cast<CameraKeyFrame*>(tail); }
	AnimatorWidget* getAnimatorWidget();
	void clear();
	void set(double t);
	
	void read(QIODevice* d);
};


#endif
