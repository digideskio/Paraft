#ifndef _FLOATANIMATOR_H_
#define _FLOATANIMATOR_H_

#include "animator.h"

class FloatAnimator;
class FloatListItem;
class FloatKeyFrame : public KeyFrame {
	float v;

	FloatListItem* li;
	FloatKeyFrame(Animator* parent, float v, double t=1, FloatKeyFrame* prev=0, FloatKeyFrame* next=0)
		:KeyFrame(parent, t, prev, next), v(v), li(0) {}
	friend class FloatAnimator;
public:
	FloatKeyFrame* getNext() { return reinterpret_cast<FloatKeyFrame*>(next); }
	FloatKeyFrame* getPrev() { return reinterpret_cast<FloatKeyFrame*>(prev); }
	float getFloat() { return v; }
	void setFloat(float var) { v = var; }
	void read(QIODevice* d);
	void save(QIODevice* d) const;
	void setListWidgetItem(FloatListItem* i) {
		li = i;
	}
	//~FloatKeyFrame();
};

class FloatAnimatorWidget;
class FloatAnimator : public Animator {
	Q_OBJECT

	double tau;
	float* v;
	FloatAnimatorWidget* optionswidget;
public:
	FloatAnimator():v(0), optionswidget(0) {} 
	void push_back(float val, double t=0);
	void push_back(double t=0);
	void setFloat(float* var) { v = var; }
	FloatKeyFrame* getHead() { return reinterpret_cast<FloatKeyFrame*>(head); }
	FloatKeyFrame* getTail() { return reinterpret_cast<FloatKeyFrame*>(tail); }
	QWidget* getOptionsWidget();
	void clear();
	void set(double t);
	
	void read(QIODevice* d);
};


#endif
