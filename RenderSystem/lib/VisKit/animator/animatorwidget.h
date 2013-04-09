#ifndef ANIMATORWIDGET_H
#define ANIMATORWIDGET_H

#include <QWidget>
class QListWidgetItem;
class KeyFrame;
class AnimatorWidget : public QWidget
{
Q_OBJECT
public:
    explicit AnimatorWidget(QWidget *parent = 0);

public slots:
	virtual void itemClicked(QListWidgetItem* item)=0;
	virtual void itemActivated(QListWidgetItem* item)=0;
	virtual void applyChanges()=0;
	virtual void resetChanges()=0;
	virtual void deleteSelected()=0;
	virtual void newKeyFrame(KeyFrame* k)=0;
	virtual void insertKeyFrame()=0;
	virtual void moveUp()=0;
	virtual void moveDown()=0;
	virtual void playClicked()=0;
	virtual void playToClicked()=0;
	virtual void stopClicked()=0;
signals:
	void updated();
	void play(double start, double end);
	void stop();
};

#endif // ANIMATORWIDGET_H
