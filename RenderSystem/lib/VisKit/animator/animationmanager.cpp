#include "animationmanager.h"
#include "animator.h"

AnimationManager::AnimationManager(QObject *parent):QObject(parent) {
	connect(&timer, SIGNAL(timeout()), this, SLOT(step()));
	timer.setSingleShot(false);
	fps = 24;
	timer.setInterval(1000/fps);
	currentTime = 0;
	stepTime = 1./24.;
	animating = false;
	isWaiting = false;
	doneWaiting = false;
	connect(&timer, SIGNAL(timeout()), this, SLOT(step()));
}

void AnimationManager::addAnimator(const QString& name, Animator* animator) {
	if(!animators.contains(name)) {
		animators[name] = animator;
		if(animator->getAnimatorWidget()) {
			connect(animator->getAnimatorWidget(), SIGNAL(play(double,double)), this, SLOT(play(double,double)));
			connect(animator->getAnimatorWidget(), SIGNAL(stop()), this, SLOT(stop()));
		}
	}
}

void AnimationManager::set(double t) {
	for(QHash<QString, Animator*>::iterator it = animators.begin(); it != animators.end(); it++) {
		(*it)->set(t);
	}
	emit updated();
}

void AnimationManager::play(double s, double e) {
	currentTime = s;
	end = e;
	animating = true;
	//mutex.lock();

	timer.start();
}

void AnimationManager::step() {
	if(doneWaiting) {
		doneWaiting = false;
		mutex.unlock();
	}
	if(!animating) {
		timer.stop();
		return;
	}
	if(!playEachFrame) {
		set(currentTime);
		currentTime += stepTime;
		if(currentTime > end) {
			animating = false;
		}
	} else {
		if(mutex.tryLock()) {
			if(currentTime > end) {
				animating = false;
			}
			isWaiting = true;
			set(currentTime);
			currentTime += stepTime;
		}
	}
}

void AnimationManager::updateFinished() {
	if(isWaiting) {
		isWaiting = false;
		doneWaiting = true;
	}
}

void AnimationManager::stop() {
	if(isWaiting) {
		isWaiting = false;
		doneWaiting = true;
	}
	animating = false;
}
