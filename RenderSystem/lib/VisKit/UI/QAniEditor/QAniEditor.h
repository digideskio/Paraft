#ifndef _QANIEDITOR_H_
#define _QANIEDITOR_H_
#include <GL/glew.h>
#include <Qt>
#include <QGLWidget>
#include <vector>
#include <QAction>
#include <QMenu>
#include "QAniClickable.h"
#include "QAniGraph.h"
#include "QAniKeyframe.h"
#include "QAniInstance.h"
#include "QAniInterface.h"
#include "shadermanager.h"
#include "QRenderWindow.h"
#include "camera.h"
#include "QTFEditor.h"
using namespace std;

class QAniClickable;
class QAniTimeline;
class QAniTimelineScaler;
class QAniToolBox;
class QAniTemplateBox;
class QAniTemplate;
class QAniGraph;

class QAniEditor : public QGLWidget
{
	Q_OBJECT
public:
	QAniEditor(QAniInterface *aniInterface, QWidget *parent = 0, const QGLWidget *shareWidget = 0);
	~QAniEditor();

protected:
	friend class QAniTimeline;
	friend class QAniTemplate;
	friend class QAniGraph;
	QAniTimeline *		m_timeline;
	QAniTimelineScaler *	m_timelineScaler;
	QAniToolBox *		m_toolBox;
	QAniTemplateBox *	m_templateBox;
	vector<QAniClickable *> clickables;

	QAniGraph *		m_graph;

	QMenu *			m_menu;
	QAction *		m_openGraph;
	QAction *		m_openFileAct;
	QAction *		m_saveFileAct;

	QAniInterface * renderer;

	QAniKeyframeManager kfm;
	QAniInstanceManager itm;

protected:
	virtual void initializeGL();
	virtual void resizeGL(int,int);
	virtual void paintGL();

	// Control
	virtual void mousePressEvent(QMouseEvent*);
	virtual void mouseMoveEvent(QMouseEvent*);
	virtual void mouseReleaseEvent(QMouseEvent*);
	virtual void wheelEvent(QWheelEvent*);
	virtual void keyPressEvent(QKeyEvent*);
	virtual void keyReleaseEvent(QKeyEvent*);

	void getFrameAt(float, QAniKeyframe*);
	QImage getSnapshotAt(float);
	void updateKeyframeAfter(float);

	int m_updating;

public slots:
	void setKeyframe(float, QAniKeyframe::Type = QAniKeyframe::ALL);
	void setTemplate(QAniTemplate::Type, float, float, GLint);
	void currentTimeChange();
	void popMenu();
	void showGraph();
	void saveAnimation();
	void openAnimation();

	void updateTimelineFromGrpah(QList<QAniKeyframe*> *);

	void updateComplete();
	void updateAllComplete();

public:
signals:
	void updateCamera(CameraOptions);
	void updateTemporal(size_t, bool);
	void updateTransferFunction(float,
				    float*,
				    QVector<GaussianObject>*,
				    QVector<TFColorTick>*,
				    bool);
	void updateSlice(Slicer);
	void pleaseUpdate();
};


#endif
