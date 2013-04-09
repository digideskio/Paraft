#include "cameraoptions.h"
#include <QtDebug>
#include "cameraanimator.h"

CameraAnimatorWidget::CameraAnimatorWidget(CameraAnimator* parent):parent(parent), selected(0) {
	setupUi(this);
	/*
	connect(tspin, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(pspin, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(ux, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(uy, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(uz, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(lx, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(ly, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(lz, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(px, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(py, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double)));
	connect(pz, SIGNAL(valueChanged(double)), this, SLOT(stuffChanged(double))); */

	dtspin->setValue(parent->getDefaultT());
	connect(dtspin, SIGNAL(valueChanged(double)), parent, SLOT(setDefaultT(double)));

	connect(apply, SIGNAL(clicked()), this, SLOT(applyChanges()));
	connect(reset, SIGNAL(clicked()), this, SLOT(resetChanges()));

	connect(listWidget, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(itemClicked(QListWidgetItem*)));
	connect(listWidget, SIGNAL(itemActivated(QListWidgetItem*)), this, SLOT(itemActivated(QListWidgetItem*)));

	connect(parent, SIGNAL(newKeyFrame(KeyFrame*)), this, SLOT(newKeyFrame(KeyFrame*)));
	connect(deleteframe, SIGNAL(clicked()), this, SLOT(deleteSelected()));
	connect(insertbutton, SIGNAL(clicked()), this, SLOT(insertKeyFrame()));
	connect(moveup, SIGNAL(clicked()), this, SLOT(moveUp()));
	connect(movedown, SIGNAL(clicked()), this, SLOT(moveDown()));
	connect(playbutton, SIGNAL(clicked()), this, SLOT(playClicked()));
	connect(stopbutton, SIGNAL(clicked()), this, SLOT(stopClicked()));
	connect(playtoframe, SIGNAL(clicked()), this, SLOT(playToClicked()));
	connect(deselect, SIGNAL(clicked()), this, SLOT(deselectSelection()));

	listWidget->setAutoScroll(true);
	listWidget->setSortingEnabled(true);

	CameraKeyFrame* k = parent->getHead();
	while(k) {
		k->setListWidgetItem(new CameraKeyFrameListItem(k, listWidget));
		k = k->getNext();
	}
}

CameraKeyFrameListItem::CameraKeyFrameListItem(CameraKeyFrame* k, QListWidget* parent):QListWidgetItem("", parent), frame(k) {
}

void CameraAnimatorWidget::applyChanges() {
	if(!selected)
		return;

	selected->setTime(tspin->value());
	selected->setPause(pspin->value());

	selected->getOptions().c.x() = -lx->value();
	selected->getOptions().c.y() = -ly->value();
	selected->getOptions().c.z() = -lz->value();

	selected->getOptions().u.x() = ux->value();
	selected->getOptions().u.y() = uy->value();
	selected->getOptions().u.z() = uz->value();

	Vector3 cam(px->value(), py->value(), pz->value());
	cam += -selected->getOptions().c*selected->getOptions().dist;

	selected->getOptions().l = cam;
}

void CameraAnimatorWidget::itemClicked(QListWidgetItem* item) {
	selected = reinterpret_cast<CameraKeyFrameListItem*>(item)->v();
	resetChanges();
}

void CameraAnimatorWidget::itemActivated(QListWidgetItem* item) {
	parent->set(reinterpret_cast<CameraKeyFrameListItem*>(item)->v()->getStartTime());
	emit updated();
}

void CameraAnimatorWidget::resetChanges() {
	if(!selected)
		return;

	tspin->setValue(selected->getTime());
	pspin->setValue(selected->getPause());

	ux->setValue(selected->getOptions().u.x());
	uy->setValue(selected->getOptions().u.y());
	uz->setValue(selected->getOptions().u.z());

	lx->setValue(-selected->getOptions().c.x());
	ly->setValue(-selected->getOptions().c.y());
	lz->setValue(-selected->getOptions().c.z());

	Vector3 cam = selected->getOptions().l + selected->getOptions().dist*selected->getOptions().c;

	px->setValue(cam.x());
	py->setValue(cam.y());
	pz->setValue(cam.z());

}

void CameraAnimatorWidget::newKeyFrame(KeyFrame* k) {
	reinterpret_cast<CameraKeyFrame*>(k)->setListWidgetItem(new CameraKeyFrameListItem(reinterpret_cast<CameraKeyFrame*>(k), listWidget));
}

void CameraAnimatorWidget::deleteSelected() {
	if(!selected)
		return;
	parent->remove(selected);
	selected = 0;
}

void CameraAnimatorWidget::insertKeyFrame() {
	parent->insert(selected);
}

void CameraAnimatorWidget::moveUp() {
	if(selected && selected->getPrev()) {
		parent->insertKeyFrame(selected, selected->getPrev());
	}
}

void CameraAnimatorWidget::moveDown() {
	if(selected && selected->getNext()) {
		parent->insertKeyFrame(selected, selected->getNext()->getNext());
	}
}

void CameraAnimatorWidget::playClicked() {
	if(!selected)
		return;
	emit play(selected->getStartTime(), parent->getTotalTime());
}

void CameraAnimatorWidget::playToClicked() {
	KeyFrame* k;
	for(k = parent->getHead(); k && k->getFrame() != endframe->value(); k = k->getNext()) {}
	if(k)
		emit play(selected->getStartTime(), k->getStartTime());
}

void CameraAnimatorWidget::stopClicked() {
	emit stop();
}

void CameraAnimatorWidget::deselectSelection() {
	selected = 0;
	listWidget->setCurrentItem(0);

}
