#ifndef _QRENDERWIDGET_H_
#define _QRENDERWIDGET_H_
#include <QObject>
#include <QTime>
#include <QImage>
#include <QList>
#include "QAniInterface.h"
#include "../camera/camera.h"
#include "../camera/vectors.h"
#include "../UI/QTFEditor/QTFEditor.h"
#include "../UI/QAniEditor/QAniInterface.h"
#include "box.h"
#include "slicer.h"
#include "LIGHTPARAM.h"

class QRenderWidget;

class MyTimer : public QObject {
	Q_OBJECT
public:
	MyTimer() { m_renderWidget = NULL; }
	void setRenderer(QRenderWidget * r) { m_renderWidget = r; }
private:
	QRenderWidget * m_renderWidget;
public slots:
	void freeMovementTimer();
	
};

class QRenderWidget: public QAniInterface
{
public:
	QRenderWidget();
	virtual ~QRenderWidget();

protected:
	friend class MyTimer;
	virtual void renderWidgetUpdated() = 0;
	virtual void resizeEvent(int,int);
	virtual void mousePressEvent(QMouseEvent*);
	virtual void mouseMoveEvent(QMouseEvent*);
	virtual void mouseReleaseEvent(QMouseEvent*);
	virtual void wheelEvent(QWheelEvent *);
	virtual void keyPressEvent(QKeyEvent*);
	virtual void keyReleaseEvent(QKeyEvent*);
	virtual void drawAxis(int options=0);

public:
	// for QAniEditor to fetch information
	// virtual ones are implemented in the inheriting class
	Camera &	getCamera() { return m_camera; }
	Slicer &	getSlicer() { return m_slicer; } // dummy object, to be compatible with single clipping version
	QList<Slicer> &	getSlicers() { return m_slicers; }
	virtual size_t	getTotalSteps() { return 0; }
	virtual size_t	getCurrentStep() { return 0; }
	virtual QTFEditor * getTFEditor() { return NULL; }
	virtual QImage	getTexture() { return QImage(); }

	void		getViewVec(float *v);
	void		switchProjection();
	void		setAxisOptions(int opt){m_axisOptions = opt;}

protected:
	void		pushMatrices();
	void		popMatrices();
	void		loadIdentities();

	int		m_axisOptions;
	bool		m_drawBoundingBox;
	void		startFPS();
	float		endFPS();

	enum MouseTarget { MTCamera, MTSliceTrack, MTSliceMove };
	MouseTarget	m_mousetarget;

	Camera		m_camera;
	Slicer		m_slicer; // dummy object, to be compatible with single clipping version
	QList<Slicer>	m_slicers;
	int		m_slicerIdx;
	Camera::ProjectionType		m_cameraProjection;
	Box		box;
	void		drawBox();

	void updateFreeMovement();
	MyTimer m_freeMovementTimer;
	bool m_freeMovementTimerunning;
	int m_freeMovement[8];
	void freeMovementTimer();

private:
	QTime m_timer;
	int m_start;
};

#endif
