#include "QAniEditor.h"
#include "QTFPanel.h"
#include <QMouseEvent>
#include <QFileDialog>
#include <cmath>
using namespace std;


#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

QAniEditor::QAniEditor(QAniInterface *aniInterface, QWidget *parent, const QGLWidget * shareWidget)
:QGLWidget(parent, shareWidget)
{
	Q_INIT_RESOURCE(QAniEditor);

	setWindowFlags(Qt::Tool);
	setWindowTitle(tr("Animation"));
	this->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
	setMinimumSize(400,300);
	resize(400,300);

	makeCurrent();
	setMouseTracking(true);

	renderer = aniInterface;
	m_updating = 0;
}

QAniEditor::~QAniEditor()
{
	makeCurrent();
}

void QAniEditor::initializeGL()
{
	glewInit();
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	m_timeline = new QAniTimeline(this);
	m_timeline->setRect(51, 0, width()-51, height()-30);
	clickables.push_back(m_timeline);

	m_timelineScaler = new QAniTimelineScaler(this);
	m_timelineScaler->setRect(51, height()-30, width()-51, 30);
	clickables.push_back(m_timelineScaler);
	m_timeline->setScaler(m_timelineScaler);

	m_toolBox = new QAniToolBox(this);
	m_toolBox->setRect(0,0,50,100);
	clickables.push_back(m_toolBox);

	m_templateBox = new QAniTemplateBox(this);
	m_templateBox->setRect(0,100,50,height()-100);
	m_templateBox->setScaler(m_timelineScaler);
	clickables.push_back(m_templateBox);
	m_timeline->setTemplateBox(m_templateBox);

	QAniTemplate *spacialOverview = new QAniTemplate(QAniTemplate::SpacialOverview, this);
	m_templateBox->add(spacialOverview);
	QAniTemplate *temporalOverview = new QAniTemplate(QAniTemplate::TemporalOverview, this);
	m_templateBox->add(temporalOverview);
	QAniTemplate *transferFunctionOverview = new QAniTemplate(QAniTemplate::TransferFunctionOverview, this);
	m_templateBox->add(transferFunctionOverview);

	m_graph = new QAniGraph(this, this);
	m_graph->resize(400, frameGeometry().height());
	m_graph->move(frameGeometry().x() + frameGeometry().width() + 2,frameGeometry().y());
//	m_graph->show();
	makeCurrent();

	m_menu = new QMenu(this);
	m_openGraph = new QAction(tr("Image Graph View"),this);
	m_saveFileAct = new QAction(tr("Save Animation"),this);
	m_openFileAct = new QAction(tr("Load Animation"),this);
	connect(m_openGraph,   SIGNAL(triggered()), this, SLOT(showGraph()));
	connect(m_saveFileAct, SIGNAL(triggered()), this, SLOT(saveAnimation()));
	connect(m_openFileAct, SIGNAL(triggered()), this, SLOT(openAnimation()));
	m_menu->addAction(m_openGraph);
	m_menu->addSeparator();
	m_menu->addAction(m_saveFileAct);
	m_menu->addAction(m_openFileAct);


	connect(m_toolBox,SIGNAL(playButtonHit()),m_timeline,SLOT(playButtonHit()));
	connect(m_toolBox,SIGNAL(recordButtonHit()),m_timeline,SLOT(recordButtonHit()));
	connect(m_toolBox,SIGNAL(menuButtonHit()),this,SLOT(popMenu()));
	connect(spacialOverview,SIGNAL(setTemplate(QAniTemplate::Type,float,float,GLint)),this,SLOT(setTemplate(QAniTemplate::Type,float,float,GLint)));
	connect(temporalOverview,SIGNAL(setTemplate(QAniTemplate::Type,float,float,GLint)),this,SLOT(setTemplate(QAniTemplate::Type,float,float,GLint)));
	connect(transferFunctionOverview,SIGNAL(setTemplate(QAniTemplate::Type,float,float,GLint)),this,SLOT(setTemplate(QAniTemplate::Type,float,float,GLint)));
	connect(m_timeline,SIGNAL(setKeyframe(float, QAniKeyframe::Type)),this,SLOT(setKeyframe(float, QAniKeyframe::Type)));
	connect(m_timeline,SIGNAL(currentTimeChange()),this,SLOT(currentTimeChange()), Qt::QueuedConnection);
	connect(m_graph,SIGNAL(pickedTimelineFromGraph(QList<QAniKeyframe*>*)),this,SLOT(updateTimelineFromGrpah(QList<QAniKeyframe*>*)));
}

void QAniEditor::resizeGL(int width, int height)
{
	glViewport(0,0,width,height);

	for (unsigned int i=0; i<clickables.size();++i) {
		clickables[i]->resize();
	}

	update();
}

void QAniEditor::paintGL()
{
	glClearColor(1.0,1.0,1.0,1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//	for (unsigned int i=0; i<clickables.size();++i) {
//		clickables[i]->draw();
//	}
	m_timeline->draw();
	m_timelineScaler->draw();
	m_toolBox->draw();
	m_templateBox->draw();
}

// Control
void QAniEditor::mousePressEvent(QMouseEvent* e)
{
	for (unsigned int i=0; i<clickables.size();++i) {
		if (clickables[i]->encloses(e->x(), e->y())) clickables[i]->press(e->x(), e->y());
	}
}

void QAniEditor::mouseMoveEvent(QMouseEvent* e)
{
	if (m_templateBox->isDragging()) {
		m_templateBox->getTemplateDrag()->drag(e->x(), e->y());
		updateGL();
		return;
	}
	for (unsigned int i=0; i<clickables.size();++i) {
		if (clickables[i]->dragging) clickables[i]->drag(e->x(), e->y());
		else if (clickables[i]->encloses(e->x(), e->y())) clickables[i]->hover(e->x(), e->y());
	}
}

void QAniEditor::mouseReleaseEvent(QMouseEvent* e)
{
	for (unsigned int i=0; i<clickables.size();++i) {
		if (clickables[i]->encloses(e->x(), e->y()) || clickables[i]->dragging) clickables[i]->release(e->x(), e->y());
	}
}

void QAniEditor::wheelEvent(QWheelEvent * e)
{
	for (unsigned int i=0; i<clickables.size();++i) {
		if (clickables[i]->encloses(e->x(), e->y())) clickables[i]->wheel(e->x(), e->y(), e->delta());
	}
}

void QAniEditor::keyPressEvent(QKeyEvent* e)
{
	switch(e->key()) {
	case Qt::Key_Escape:
		if (m_templateBox->isDragging()) {
			m_templateBox->clearDragging();
			releaseMouse();
			updateGL();
		}
		break;
	}
	e->accept();
}

void QAniEditor::keyReleaseEvent(QKeyEvent* e)
{
	e->accept();
}

void QAniEditor::setKeyframe(float t, QAniKeyframe::Type type)
{
	//renderer->makeCurrent();
	//renderer->renderToTexture();
	QImage img = renderer->getTexture();
	makeCurrent();

/* This function should
 * 1) determine which type of keyframe to create according to the change that users have already made.
 * 2) create corresponding instances
 */
	if (type == QAniKeyframe::Camera || type == QAniKeyframe::ALL) {
		// Create a camera keyframe
		QAniCameraKeyframe * ckf = new QAniCameraKeyframe(t);
		renderer->getCamera().saveOptions(ckf->getOptions());
		if (!(*((QAniCameraKeyframe*)kfm(QAniKeyframe::Camera)) == *ckf) || kfm[QAniKeyframe::Camera].size() == 0) {
			*((QAniCameraKeyframe*)kfm(QAniKeyframe::Camera)) = *ckf;
			ckf->setTexture(img);
			kfm.addKeyframe(ckf);

			// Dealing with camera instances
			// case 1) new keyframe on some instances -> remove intersecting instances
			// case 2) new keyframe in the blank space -> create corresponding instances
			itm.removeInstancesAt(QAniInstance::Camera, t);
			QAniKeyframe * pkf = kfm.findPrevKeyframe(t, QAniKeyframe::Camera);
			QAniKeyframe * nkf = kfm.findNextKeyframe(t, QAniKeyframe::Camera);
			if (pkf) itm.generateInstancesAccordingTo(pkf, ckf);
			if (nkf) itm.generateInstancesAccordingTo(ckf, nkf);
			updateGL();
		}
		else {
			QAniKeyframe * hkf = kfm.findKeyframeAt(t, QAniKeyframe::Camera);
			QAniKeyframe * pkf = kfm.findPrevKeyframe(t, QAniKeyframe::Camera);
			QAniKeyframe * nkf = kfm.findNextKeyframe(t, QAniKeyframe::Camera);
			if (!hkf && pkf && !nkf) {
				if (itm.contains(QAniInstance::Camera, pkf->getTime())) {
					*((QAniCameraKeyframe*)kfm(QAniKeyframe::Camera)) = *ckf;
					ckf->setTexture(img);
					kfm.addKeyframe(ckf);
				}
			}
			else if (!hkf && !pkf && nkf) {
				if (itm.contains(QAniInstance::Camera, nkf->getTime())) {
					*((QAniCameraKeyframe*)kfm(QAniKeyframe::Camera)) = *ckf;
					ckf->setTexture(img);
					kfm.addKeyframe(ckf);
				}
			}
			else ckf;
		}
	}

	if (type == QAniKeyframe::Temporal || type == QAniKeyframe::ALL) {
		// Create a temporal keyframe
		QAniTemporalKeyframe * tkf = new QAniTemporalKeyframe(t, renderer->getCurrentStep());
		if (!(*((QAniTemporalKeyframe*)kfm(QAniKeyframe::Temporal)) == *tkf) || kfm[QAniKeyframe::Temporal].size() == 0) {
			*((QAniTemporalKeyframe*)kfm(QAniKeyframe::Temporal)) = *tkf;
			tkf->setTexture(img);
			kfm.addKeyframe(tkf);

			itm.removeInstancesAt(QAniInstance::Temporal, t);
			QAniKeyframe * pkf = kfm.findPrevKeyframe(t, QAniKeyframe::Temporal);
			QAniKeyframe * nkf = kfm.findNextKeyframe(t, QAniKeyframe::Temporal);
			if (pkf) itm.generateInstancesAccordingTo(pkf, tkf);
			if (nkf) itm.generateInstancesAccordingTo(tkf, nkf);
			updateGL();
		}
		else {
			QAniKeyframe * hkf = kfm.findKeyframeAt(t, QAniKeyframe::Temporal);
			QAniKeyframe * pkf = kfm.findPrevKeyframe(t, QAniKeyframe::Temporal);
			QAniKeyframe * nkf = kfm.findNextKeyframe(t, QAniKeyframe::Temporal);
			if (!hkf && pkf && !nkf) {
				if (itm.contains(QAniInstance::Temporal, pkf->getTime())) {
					*((QAniTemporalKeyframe*)kfm(QAniKeyframe::Temporal)) = *tkf;
					tkf->setTexture(img);
					kfm.addKeyframe(tkf);
				}
			}
			else if (!hkf && !pkf && nkf) {
				if (itm.contains(QAniInstance::Temporal, nkf->getTime())) {
					*((QAniTemporalKeyframe*)kfm(QAniKeyframe::Temporal)) = *tkf;
					tkf->setTexture(img);
					kfm.addKeyframe(tkf);
				}
			}
			else tkf;
		}
	}

	if (type == QAniKeyframe::TransferFunction || type == QAniKeyframe::ALL) {
		// Create a transfer function keyframe
		QAniTransferFunctionKeyframe * tfkf = new QAniTransferFunctionKeyframe(t,
							renderer->getTFEditor()->getTFResolution(),
							renderer->getTFEditor()->getTFDrawArray(),
							renderer->getTFEditor()->getGaussians(),
							renderer->getTFEditor()->getColors());
		if (!(*((QAniTransferFunctionKeyframe*)kfm(QAniKeyframe::TransferFunction)) == *tfkf) || kfm[QAniKeyframe::TransferFunction].size() == 0) {
			*((QAniTransferFunctionKeyframe*)kfm(QAniKeyframe::TransferFunction)) = *tfkf;
			tfkf->setTexture(img);
			kfm.addKeyframe(tfkf);

			itm.removeInstancesAt(QAniInstance::TransferFunction, t);
			QAniKeyframe * pkf = kfm.findPrevKeyframe(t, QAniKeyframe::TransferFunction);
			QAniKeyframe * nkf = kfm.findNextKeyframe(t, QAniKeyframe::TransferFunction);
			if (pkf) itm.generateInstancesAccordingTo(pkf, tfkf);
			if (nkf) itm.generateInstancesAccordingTo(tfkf, nkf);
			updateGL();
		}
		else {
			QAniKeyframe * hkf = kfm.findKeyframeAt(t, QAniKeyframe::TransferFunction);
			QAniKeyframe * pkf = kfm.findPrevKeyframe(t, QAniKeyframe::TransferFunction);
			QAniKeyframe * nkf = kfm.findNextKeyframe(t, QAniKeyframe::TransferFunction);
			if (!hkf && pkf && !nkf) {
				if (itm.contains(QAniInstance::TransferFunction, pkf->getTime())) {
					*((QAniTransferFunctionKeyframe*)kfm(QAniKeyframe::TransferFunction)) = *tfkf;
					tfkf->setTexture(img);
					kfm.addKeyframe(tfkf);
				}
			}
			else if (!hkf && !pkf && nkf) {
				if (itm.contains(QAniInstance::TransferFunction, nkf->getTime())) {
					*((QAniTransferFunctionKeyframe*)kfm(QAniKeyframe::TransferFunction)) = *tfkf;
					tfkf->setTexture(img);
					kfm.addKeyframe(tfkf);
				}
			}
			else delete tfkf;
		}

	}

	if (type == QAniKeyframe::Slice || type == QAniKeyframe::ALL) {
		// Create a slice keyfram
		QAniSliceKeyframe * skf = new QAniSliceKeyframe(t, renderer->getSlicer());
		if (!(*((QAniSliceKeyframe*)kfm(QAniKeyframe::Slice)) == *skf) || kfm[QAniKeyframe::Slice].size() == 0) {
			*((QAniSliceKeyframe*)kfm(QAniKeyframe::Slice)) = *skf;
			skf->setTexture(img);
			kfm.addKeyframe(skf);

			itm.removeInstancesAt(QAniInstance::Slice, t);
			QAniKeyframe * pkf = kfm.findPrevKeyframe(t, QAniKeyframe::Slice);
			QAniKeyframe * nkf = kfm.findNextKeyframe(t, QAniKeyframe::Slice);
			if (pkf) itm.generateInstancesAccordingTo(pkf, skf);
			if (nkf) itm.generateInstancesAccordingTo(skf, nkf);
			updateGL();
		}
		else {
			QAniKeyframe * hkf = kfm.findKeyframeAt(t, QAniKeyframe::Slice);
			QAniKeyframe * pkf = kfm.findPrevKeyframe(t, QAniKeyframe::Slice);
			QAniKeyframe * nkf = kfm.findNextKeyframe(t, QAniKeyframe::Slice);
			if (!hkf && pkf && !nkf) {
				if (itm.contains(QAniInstance::Slice, pkf->getTime())) {
					*((QAniSliceKeyframe*)kfm(QAniKeyframe::Slice)) = *skf;
					skf->setTexture(img);
					kfm.addKeyframe(skf);
				}
			}
			else if (!hkf && !pkf && nkf) {
				if (itm.contains(QAniInstance::Slice, nkf->getTime())) {
					*((QAniSliceKeyframe*)kfm(QAniKeyframe::Slice)) = *skf;
					skf->setTexture(img);
					kfm.addKeyframe(skf);
				}
			}
			else delete skf;
		}
	}

	m_graph->buildGraph(&(kfm[QAniKeyframe::ALL]));
}

void QAniEditor::setTemplate(QAniTemplate::Type type, float leftx, float rightx, GLint tex) {
	// create 2 keyframes and 1 instance:
	// 1) create left keyframe according to timeline state
	// 2) remove overlapped instances and keyframes
	// 3) create an instance
	// 4) create right keyframe according to left keyframe and the instance
	switch (type) {
	case QAniTemplate::SpacialOverview: {
		QAniCameraKeyframe * start = new QAniCameraKeyframe(leftx);
		getFrameAt(leftx, start);
		QAniCameraInstance * instance = new QAniCameraInstance(2*M_PI, start->getOptions().u);
		QAniCameraKeyframe * end = instance->generateKeyframe(start, rightx - leftx);
		instance->setStart(start);
		instance->setEnd(end);
		instance->setTexture(tex);

		// remove overlapped instances and keyframes
		itm.removeInstancesIn(QAniInstance::Camera, leftx, rightx);
		kfm.removeKeyframesIn(QAniKeyframe::Camera, leftx, rightx);

		// add new keyframes and instances
		kfm.addKeyframe(start);
		kfm.addKeyframe(end);
		itm.addInstance(instance);

		// generate adjacent instances
		QAniCameraKeyframe * pkf = (QAniCameraKeyframe*)kfm.findPrevKeyframe(leftx, QAniKeyframe::Camera);
		QAniCameraKeyframe * nkf = (QAniCameraKeyframe*)kfm.findNextKeyframe(rightx, QAniKeyframe::Camera);
		if (pkf) if (!(*pkf == *start)) itm.generateInstancesAccordingTo(pkf, start);
		if (nkf) if (!(*end == *nkf)) itm.generateInstancesAccordingTo(end, nkf);

		// set the snapshots up
		start->setTexture(getSnapshotAt(start->getTime()));
		end->setTexture(getSnapshotAt(end->getTime()));

		break; }
	case QAniTemplate::TemporalOverview: {
		QAniTemporalKeyframe * start = new QAniTemporalKeyframe(leftx, 1);
		QAniTemporalInstance * instance = new QAniTemporalInstance(renderer->getTotalSteps() - 1);
		QAniTemporalKeyframe * end = new QAniTemporalKeyframe(rightx, renderer->getTotalSteps());
		instance->setStart(start);
		instance->setEnd(end);
		instance->setTexture(tex);


		// remove overlapped instances and keyframes
		itm.removeInstancesIn(QAniInstance::Temporal, leftx, rightx);
		kfm.removeKeyframesIn(QAniKeyframe::Temporal, leftx, rightx);

		// add new keyframes and instances
		kfm.addKeyframe(start);
		kfm.addKeyframe(end);
		itm.addInstance(instance);

		// generate adjacent instances
		QAniTemporalKeyframe * pkf = (QAniTemporalKeyframe*)kfm.findPrevKeyframe(leftx, QAniKeyframe::Temporal);
		QAniTemporalKeyframe * nkf = (QAniTemporalKeyframe*)kfm.findNextKeyframe(rightx, QAniKeyframe::Temporal);
		if (pkf) if (!(*pkf == *start)) itm.generateInstancesAccordingTo(pkf, start);
		if (nkf) if (!(*end == *nkf)) itm.generateInstancesAccordingTo(end, nkf);

		// set the snapshots up
		start->setTexture(getSnapshotAt(start->getTime()));
		end->setTexture(getSnapshotAt(end->getTime()));

		break; }
	case QAniTemplate::TransferFunctionOverview: {
		QAniTransferFunctionKeyframe * left = new QAniTransferFunctionKeyframe(leftx);
		QAniTransferFunctionKeyframe * right = new QAniTransferFunctionKeyframe(rightx);
		getFrameAt(leftx, left);
		getFrameAt(rightx, right);
		int resolution = renderer->getTFEditor()->getTFResolution();
		float * drawArray = new float[resolution];
		for (int i = 0; i < resolution; ++i) drawArray[i] = 0.0;
		GaussianObject go(0.0, 0.02, 0.04, resolution);
		QVector<GaussianObject> gaussians;
		gaussians.push_back(go);
		QAniTransferFunctionKeyframe * start = new QAniTransferFunctionKeyframe(leftx, resolution, drawArray, &gaussians, left->getColors());
		go.m_mean = 1.0;
		gaussians.clear();
		gaussians.push_back(go);
		QAniTransferFunctionKeyframe * end = new QAniTransferFunctionKeyframe(rightx, resolution, drawArray, &gaussians, right->getColors());

		// remove overlapped instances and keyframes
		itm.removeInstancesIn(QAniInstance::TransferFunction, leftx, rightx);
		kfm.removeKeyframesIn(QAniKeyframe::TransferFunction, leftx, rightx);

		// add new keyframes and instances
		kfm.addKeyframe(start);
		kfm.addKeyframe(end);
		QAniInstance * instance = itm.generateInstancesAccordingTo(start, end);
		instance->setTexture(tex);


		// generate adjacent instances
		QAniTransferFunctionKeyframe * pkf = (QAniTransferFunctionKeyframe*)kfm.findPrevKeyframe(leftx, QAniKeyframe::TransferFunction);
		QAniTransferFunctionKeyframe * nkf = (QAniTransferFunctionKeyframe*)kfm.findNextKeyframe(rightx, QAniKeyframe::TransferFunction);
		if (pkf) if (!(*pkf == *start)) itm.generateInstancesAccordingTo(pkf, start);
		if (nkf) if (!(*end == *nkf)) itm.generateInstancesAccordingTo(end, nkf);

		// set the snapshots up
		start->setTexture(getSnapshotAt(start->getTime()));
		end->setTexture(getSnapshotAt(end->getTime()));

		
		break; }
	}

	updateKeyframeAfter(leftx);
	m_graph->buildGraph(&(kfm[QAniKeyframe::ALL]));
	updateGL();
}

void QAniEditor::currentTimeChange() {
	float currentTime = m_timeline->currentx;
	m_updating += 4;
	
	// update Camera
	getFrameAt(currentTime, kfm(QAniKeyframe::Camera));
	emit updateCamera(((QAniCameraKeyframe*)kfm(QAniKeyframe::Camera))->getOptions());

	// update Temporal Step
	getFrameAt(currentTime, kfm(QAniKeyframe::Temporal));
	emit updateTemporal(((QAniTemporalKeyframe*)kfm(QAniKeyframe::Temporal))->getTimeStep(), false);

	// update Transfer Function Step
	getFrameAt(currentTime, kfm(QAniKeyframe::TransferFunction));
	emit updateTransferFunction(((QAniTransferFunctionKeyframe*)kfm(QAniKeyframe::TransferFunction))->getResolution(),
				    ((QAniTransferFunctionKeyframe*)kfm(QAniKeyframe::TransferFunction))->getDrawArray(),
				    ((QAniTransferFunctionKeyframe*)kfm(QAniKeyframe::TransferFunction))->getGaussians(),
				    ((QAniTransferFunctionKeyframe*)kfm(QAniKeyframe::TransferFunction))->getColors(),
				    false);

	// update Slice Step
	getFrameAt(currentTime, kfm(QAniKeyframe::Slice));
	emit updateSlice(((QAniSliceKeyframe*)kfm(QAniKeyframe::Slice))->getSlicer());

	while(m_updating);
	emit pleaseUpdate();
	//renderer->updateGL();
}

void QAniEditor::getFrameAt(float t, QAniKeyframe * frame) {
	QAniKeyframe::Type type = frame->getType();
	frame->setTime(t);
	if (kfm[type].size() > 0) {
		if (kfm[type].size() == 1 || t <= kfm[type].first()->getTime()) *frame = *(kfm[type].first());
		else if (t >= kfm[type].last()->getTime()) *frame = *(kfm[type].last());
		else {
			QAniInstance * it = itm.contains((QAniInstance::Type)type, t);
			if (it) it->generateInterpolatedFrame(frame);
			else *frame = *(kfm.findPrevKeyframe(t, type));
		}
	}
	else {
		switch (type) {
		case QAniKeyframe::Camera:
			renderer->getCamera().saveOptions(((QAniCameraKeyframe*)frame)->getOptions());
			break;
		case QAniKeyframe::Temporal:
			*((QAniTemporalKeyframe*)frame) = QAniTemporalKeyframe(t, renderer->getCurrentStep());
			break;
		case QAniKeyframe::TransferFunction:
			*((QAniTransferFunctionKeyframe*)frame) = QAniTransferFunctionKeyframe(t,
								renderer->getTFEditor()->getTFResolution(),
								renderer->getTFEditor()->getTFDrawArray(),
								renderer->getTFEditor()->getGaussians(),
								renderer->getTFEditor()->getColors());
			break;
		case QAniKeyframe::Slice:
			*((QAniSliceKeyframe*)frame) = QAniSliceKeyframe(t, renderer->getSlicer());
			break;
		}
	}
}

QImage QAniEditor::getSnapshotAt(float time) {
	QImage a;
	QAniCameraKeyframe *timeCamera = new QAniCameraKeyframe(time);
	QAniTemporalKeyframe * timeTemporal = new QAniTemporalKeyframe(time);
	QAniTransferFunctionKeyframe * timeTransferFunction = new QAniTransferFunctionKeyframe(time);
	QAniSliceKeyframe * timeSlice = new QAniSliceKeyframe(time);
	getFrameAt(time, timeCamera);
	getFrameAt(time, timeTemporal);
	getFrameAt(time, timeTransferFunction);
	getFrameAt(time, timeSlice);

	QAniCameraKeyframe * tmpCamera;
	QAniTemporalKeyframe * tmpTemporal;
	QAniTransferFunctionKeyframe * tmpTransferFunction;
	QAniSliceKeyframe * tmpSlice;

	tmpCamera = new QAniCameraKeyframe(time);
	renderer->getCamera().saveOptions(tmpCamera->getOptions());
	tmpTemporal = new QAniTemporalKeyframe(time, renderer->getCurrentStep());
	tmpTransferFunction = new QAniTransferFunctionKeyframe(time,
						renderer->getTFEditor()->getTFResolution(),
						renderer->getTFEditor()->getTFDrawArray(),
						renderer->getTFEditor()->getGaussians(),
						renderer->getTFEditor()->getColors());
	tmpSlice = new QAniSliceKeyframe(time, renderer->getSlicer());

	m_updating += 4;
	emit updateCamera(timeCamera->getOptions());
	emit updateTemporal(timeTemporal->getTimeStep(), true);
	emit updateTransferFunction(timeTransferFunction->getResolution(),
				    timeTransferFunction->getDrawArray(),
				    timeTransferFunction->getGaussians(),
				    timeTransferFunction->getColors(),
				    true);
	emit updateSlice(timeSlice->getSlicer());
	while(m_updating);
//	renderer->renderToTexture();
	QImage img = renderer->getTexture();

	m_updating += 4;
	emit updateCamera(tmpCamera->getOptions());
	emit updateTemporal(tmpTemporal->getTimeStep(), true);
	emit updateTransferFunction(tmpTransferFunction->getResolution(),
				    tmpTransferFunction->getDrawArray(),
				    tmpTransferFunction->getGaussians(),
				    tmpTransferFunction->getColors(),
				    true);
	emit updateSlice(tmpSlice->getSlicer());

	makeCurrent();
	updateGL();

	return img;
}

void QAniEditor::updateKeyframeAfter(float time) {
	// update kfm[QAniKeyframe::ALL]
	float max = -999999999.9f;
	for (int i = 0; i < kfm[QAniKeyframe::ALL].size(); ++i) {
		if (kfm[QAniKeyframe::ALL][i]->getTime() >= max) max = kfm[QAniKeyframe::ALL][i]->getTime();
		else {
			QAniKeyframe * tmp = kfm[QAniKeyframe::ALL][i];
			kfm[QAniKeyframe::ALL].removeAt(i);
			int j;
			for (j = i - 1; j >= 0; j--) {
				if (tmp->getTime() >= kfm[QAniKeyframe::ALL][j]->getTime()) {
					kfm[QAniKeyframe::ALL].insert(j+1, tmp);
					break;
				}
			}
			if (j == -1) kfm[QAniKeyframe::ALL].push_front(tmp);
		}
	}

	// update snapshots
	for (int i = 0; i < kfm[QAniKeyframe::ALL].size(); ++i) {
		if (kfm[QAniKeyframe::ALL][i]->getTime() >= time) {
			kfm[QAniKeyframe::ALL][i]->updateTexture(getSnapshotAt(kfm[QAniKeyframe::ALL][i]->getTime()));
		}
	}

	m_graph->buildGraph(&(kfm[QAniKeyframe::ALL]));
}

void QAniEditor::popMenu() {
	m_menu->exec(QCursor::pos());
}

void QAniEditor::showGraph() {
	m_graph->show();
}

void QAniEditor::saveAnimation() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Save Animation"), ".", tr("ANImation files (*.ani);;All files (*.*)"));
	if(!filename.isEmpty()) {
		if(!filename.contains(QString(".")))
			filename += ".ani";

		QFile file(filename);
		if(!file.open(QIODevice::WriteOnly))
			return;
		m_graph->saveGraph(file);
		file.close();
	}
}

void QAniEditor::openAnimation() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Load Animation"), ".", tr("ANImation files (*.ani);;All files (*.*)"));
	if(!filename.isEmpty()) {
		QFile file(filename);

		if(!file.open(QIODevice::ReadOnly))
			return;

		m_graph->loadGraph(file);
		file.close();
	}
}

void QAniEditor::updateTimelineFromGrpah(QList<QAniKeyframe*> * list) {
	// remove current keyframes and instances
	itm.removeALL();
	kfm.removeALL();

	// add keyframes and instances
	for (int i = 0; i < (*list).size(); ++i) {
		QAniKeyframe * keyframe = (*list)[i];
		kfm.addKeyframe(keyframe);
		if (keyframe->getLeftInstance()) {
			QAniKeyframe * pkf = kfm.findPrevKeyframe(keyframe->getTime(), keyframe->getType());
			pkf->setRightInstance(keyframe->getLeftInstance());
			keyframe->getLeftInstance()->setStart(pkf);
			keyframe->getLeftInstance()->setEnd(keyframe);
			itm.addInstance(keyframe->getLeftInstance());
		}
	}
	currentTimeChange();
	updateGL();
}

void QAniEditor::updateComplete() {
	if(m_updating > 0)
		m_updating--;
}

void QAniEditor::updateAllComplete() {
	m_updating = 0;
}
