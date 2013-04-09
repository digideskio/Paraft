#include "floatanimator.h"

void FloatAnimator::push_back(double t) {
	Animator::push_back(new FloatKeyFrame(this, *v, t));
}

void FloatAnimator::push_back(float val, double t) {
	Animator::push_back(new FloatKeyFrame(this, val, t));
}

void FloatAnimator::set(double t) {
	if(!v)
		return;
	if(t > totaltime) {
		*v = getTail()->v;
	}
	
	FloatKeyFrame* k = reinterpret_cast<FloatKeyFrame*>(findFrame(t));
	if(k->getNext()) {
		t -= k->getStartTime();
		if(t < k->getPause()) {
			*v = k->v;
		} else {
			t = (t - k->getPause())/k->getTime();
			*v = k->v + t*(k->getNext()->v - k->v);
		}
	} else {
		*v = k->v;
	}
}

void FloatKeyFrame::save(QIODevice* d) const {
	KeyFrame::save(d);
	d->write((char*)&v, 4);
}

void FloatKeyFrame::read(QIODevice* d) {
	KeyFrame::read(d);
	d->read((char*)&v, 4);
}

void FloatAnimator::read(QIODevice *d) {
	if(head)
		delete head;
	head = 0;
	tail = 0;

	int n;
	d->read((char*)&n, 4);
	FloatKeyFrame* k;

	for(int i = 0; i < n; i++) {
		k = new FloatKeyFrame(this, 0);
		k->read(d);
		Animator::push_back(k);
	}
}
