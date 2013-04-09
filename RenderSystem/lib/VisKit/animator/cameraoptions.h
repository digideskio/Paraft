#ifndef _CAMERAOPTIONS_H_
#define _CAMERAOPTIONS_H_

#include <QWidget>
#include "animatorwidget.h"
#include "ui_cameraanimator.h"
#include <QListWidgetItem>

class CameraKeyFrame;
class CameraAnimator;
class KeyFrame;
class CameraKeyFrameListItem : public QListWidgetItem {
	CameraKeyFrame* frame;
public:
	CameraKeyFrameListItem(CameraKeyFrame* k, QListWidget* parent);
	CameraKeyFrame* v() { return frame; }
};

class CameraAnimatorWidget : public AnimatorWidget, public Ui::CANWidget  {
	Q_OBJECT
	CameraAnimator* parent;
	CameraKeyFrame* selected;
public:
	CameraAnimatorWidget(CameraAnimator* parent);
public slots:
	void itemClicked(QListWidgetItem* item);
	void itemActivated(QListWidgetItem* item);
	void applyChanges();
	void resetChanges();
	void deleteSelected();
	void newKeyFrame(KeyFrame* k);
	void insertKeyFrame();
	void moveUp();
	void moveDown();
	void playClicked();
	void playToClicked();
	void stopClicked();
	void deselectSelection();
};


#endif

