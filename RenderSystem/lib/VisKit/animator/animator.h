#ifndef _ANIMATOR_H_
#define _ANIMATOR_H_

#include <QObject>
#include <QDataStream>

#include <QListWidgetItem>
#include "animatorwidget.h"
class Animator;
class KeyFrame {
protected:
	Animator* parent;
	double t;
	double pause;
	double starttime;
	int frame;
	KeyFrame* prev;
	KeyFrame* next;
	QListWidgetItem* li;
	KeyFrame(Animator* parent, double t=1, KeyFrame* prev=0, KeyFrame* next=0, double pause=0):
			parent(parent), t(t), pause(pause), prev(prev), next(next), li(0) {
		updateTime();
	}
	void setPrev(KeyFrame* k);
	void setNext(KeyFrame* k);
	void insert(KeyFrame* prev, KeyFrame* next);
	void remove();
public:
	virtual ~KeyFrame();
	void setTime(double t);
	KeyFrame* head();
	KeyFrame* tail();
	virtual KeyFrame* getNext() { return next; }
	virtual KeyFrame* getPrev() { return prev; }
	virtual void updateTime();
	int getFrame() const { return frame; }
	double getTime() const { return t; }
	double getStartTime() const { return starttime; }
	double getPause() const { return pause; }
	void setPause(double p);
	void setListWidgetItem(QListWidgetItem* i) {
		li = i;
		setListWidgetText();
	}
	void setListWidgetText() {
		if(li) {
			li->setText(QString("%1 %2").arg(QString("%1").arg(frame, 5, 10, QChar('0')), QString::number(starttime, '0', 5)));
			
		}
	}
	friend QDataStream& operator<<(QDataStream& ds, const KeyFrame& k);
	friend QDataStream& operator>>(QDataStream& ds, KeyFrame& k);

	virtual void read(QIODevice* d);
	virtual void save(QIODevice* d) const;

	friend class Animator;
};

class Animator : public QObject {
	Q_OBJECT
protected:
	double totaltime;
	double currentime;
	double defaultt;
	
	KeyFrame* head;
	KeyFrame* tail;
	KeyFrame* cur;
	friend class KeyFrame;
public:
	
	Animator():totaltime(0), currentime(0), defaultt(1) , head(0), tail(0), cur(0) {}
	virtual ~Animator();
	virtual void set(double t)=0;
	
	virtual void push_back(KeyFrame*);
	virtual void remove(KeyFrame*);
	
	virtual KeyFrame* getHead() { return head; }
	virtual KeyFrame* getTail() { return tail; }
	virtual KeyFrame* findFrame(double t);
	virtual void insertKeyFrame(KeyFrame* k, KeyFrame* next);

	friend QDataStream& operator<<(QDataStream& ds, const Animator& rhs);
	friend QDataStream& operator>>(QDataStream& ds, Animator& rhs);

	virtual void read(QIODevice* d)=0;
	virtual void save(QIODevice* d) const;
	
	double getTotalTime() const { return totaltime; }
	double getDefaultT() const { return defaultt; }
	virtual AnimatorWidget* getAnimatorWidget() { return 0; }
public slots:
	void setDefaultT(double t) {
		defaultt = t;
	}
signals:
	void newKeyFrame(KeyFrame* key);
};


#endif
