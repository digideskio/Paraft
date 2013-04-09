#ifndef _ANIMATIONMANAGER_H_
#define _ANIMATIONMANAGER_H_

#include <QHash>
#include <QObject>
#include <QTimer>
#include <QTime>
#include <QMutex>

class Animator;
class AnimationManager : public QObject {
Q_OBJECT

	QHash<QString, Animator*> animators;
	QTimer timer;
	bool playEachFrame;
	double stepTime;
	int fps;
	double currentTime;
	bool isWaiting;
	QMutex mutex;
	QTime qtime;
	double end;
	bool animating;
	bool doneWaiting;
public:
	explicit AnimationManager(QObject* parent=0);
	void set(double t);
	void addAnimator(const QString& name, Animator* animator);
	Animator* operator[](const QString& index);
	void setPlayEachFrame(bool v) { playEachFrame = v; }
	int getCurrentFrame() const { return static_cast<int>(currentTime*fps); }
	bool isPlayingEachFrame() const { return playEachFrame; }
	void setFPS(int v) { fps = v; stepTime = 1./v; }
	int getFPS() const { return fps; }
	bool isAnimating() const { return animating; }
public slots:
	void play(double start, double end);
	void step();
	void updateFinished();
	void stop();
signals:
	void updated();
};

#endif
