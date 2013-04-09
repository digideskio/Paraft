#include "animator.h"
#include <QIODevice>
#include <QListWidgetItem>

void KeyFrame::updateTime() {
	if(!prev) {
		starttime = 0;
		frame = 0;
	} else {
		starttime = prev->starttime + prev->t + prev->pause;
		frame = prev->frame + 1;
	}

	if(next)
		next->updateTime();
	else
		parent->totaltime = starttime;

	setListWidgetText();
}

void KeyFrame::remove() {
	if(prev)
		prev->next = next;
	if(next) {
		next->setPrev(prev);
	}

	prev = 0;
	next = 0;
}

void KeyFrame::setPrev(KeyFrame* k) {
	prev = k;
	if(prev)
		prev->next = this;
	updateTime();
}

void KeyFrame::setNext(KeyFrame* k) {
	next = k;
	if(k)
		k->updateTime();
}

void KeyFrame::setTime(double tnew) {
	if(tnew <= 0)
		return;
	t = tnew;

	updateTime();
}

void KeyFrame::setPause(double p) {
	if(p < 0)
		return;
	pause = p;

	updateTime();
}

void KeyFrame::insert(KeyFrame* p, KeyFrame* n) {
	remove();
	next = n;
	if(next)
		next->prev = this;
	setPrev(p);
}

KeyFrame* KeyFrame::head() {
	if(!prev)
		return this;
	KeyFrame* h = prev;
	while(h->prev) {
		h = h->prev;
	}
	return h;
}

KeyFrame* KeyFrame::tail() {
	if(!next)
		return this;
	KeyFrame* h = next;
	while(h->next) {
		h = h->next;
	}
	return h;
}

KeyFrame::~KeyFrame() {
	if(next)
		delete next;
	if(li)
		delete li;
}

void Animator::push_back(KeyFrame* k) {
	if(!head) {
		head = k;
		tail = k;
	} else {
		k->setPrev(tail);
		tail = k->tail();
	}
	if(tail)
		totaltime = tail->starttime;

	emit newKeyFrame(k);
}

Animator::~Animator() {
	if(head)
		delete head;
}

//so, the reason to have this is to make the search fast for sequential renders of frames
KeyFrame* Animator::findFrame(double t) {
	if(!head)
		return 0;
	if(!cur)
		cur = head;

	if(t <= 0)
		return head;

	if(head == tail)
		return head;
	if(t >= cur->starttime) {
		while(cur->next && t > cur->next->starttime) {
			cur = cur->next;
		}
		return cur;
	}
	while(cur->prev && cur->prev->starttime >= t) {
		cur = cur->prev;
	}
	return cur->prev;
}

void Animator::insertKeyFrame(KeyFrame* k, KeyFrame* next) {
	if(!next) {
		push_back(k);
		return;
	}
	if(!k->li)
		emit newKeyFrame(k);
	k->insert(next->prev, next);
	if(next == head)
		head = k;
}

void Animator::remove(KeyFrame* k) {
	if(!k || !(k->parent == this)) //dont do anything if its not ours
		return;
	if(k == head) {
		head = k->next;
	}
	if(k == tail) {
		tail = k->prev;
	}
	k->remove();
	delete k;
	cur = head;

	if(tail)
		totaltime = tail->starttime;
}

QDataStream& operator<<(QDataStream& ds, const KeyFrame& k) {
	k.save(ds.device());
	return ds;
}

QDataStream& operator>>(QDataStream& ds, KeyFrame& k) {
	k.read(ds.device());
	return ds;
}

void KeyFrame::save(QIODevice* d) const {
	d->write((char*)&t, 8);
	d->write((char*)&pause, 8);
}

void KeyFrame::read(QIODevice* d) {
	d->read((char*)&t, 8);
	d->read((char*)&pause, 8);
}


QDataStream& operator<<(QDataStream& ds, const Animator& rhs) {
	rhs.save(ds.device());
	return ds;
}

QDataStream& operator>>(QDataStream& ds, Animator& rhs) {
	rhs.read(ds.device());
	return ds;
}

void Animator::save(QIODevice *d) const {
	if(!head)
		return;

	int n = tail->frame + 1;
	d->write((char*)&n, 4);

	for(KeyFrame* k = head; k; k = k->next) {
		k->save(d);
	}
}


