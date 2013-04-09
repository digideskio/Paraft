#ifndef _QRENDERWINDOW_H_
#define _QRENDERWINDOW_H_
#include <GL/glew.h>
#include <Qt>
#include <QGLWidget>
#include <QGLFramebufferObject>
#include <QTime>
#include <QList>
#include "QAniInterface.h"
#include "../camera/camera.h"
#include "../camera/vectors.h"
#include "../UI/QTFEditor/QTFEditor.h"
#include "../UI/QAniEditor/QAniInterface.h"
#include "box.h"
#include "slicer.h"
#include "LIGHTPARAM.h"

//class CShader;
//class ShaderManager;

class GLProperty
{
public:
	GLint	viewport[4];
};

class QRenderWindow : public QGLWidget, public QAniInterface
{
	Q_OBJECT
public:
	QRenderWindow(QWidget * parent = 0, const QGLWidget * shareWidget = 0);
	QRenderWindow(const QGLFormat& format, QWidget * parent = 0, const QGLWidget * shareWidget = 0);
	virtual ~QRenderWindow();
	void	getViewport(GLint vp[4]) {glGetIntegerv(GL_VIEWPORT,vp);}
	void	setViewport(GLint vp[4]) {glViewport(vp[0],vp[1],vp[2],vp[3]);}
	//void	setShader(CShader*);

	//CShader		*m_shader;

	void		getViewVec(float *v);
	void		setIfRecording(bool r){m_ifRecording = r;}
	bool		getIfRecording(){return m_ifRecording;}
	void		takeScreenShot();
	void		switchProjection();

	void		setAxisOptions(int opt){m_axisOptions = opt;}

	// for QAniEditor to fetch information
	// virtual ones are implemented in the inheriting class
	Camera &	getCamera() { return m_camera; }
	Slicer &	getSlicer() { return m_slicer; } // dummy object, to be compatible with single clipping version
	QList<Slicer> &	getSlicers() { return m_slicers; }
	virtual size_t	getTotalSteps() { return 0; }
	virtual size_t	getCurrentStep() { return 0; }
	virtual QTFEditor * getTFEditor() { return NULL; }

	void		enableFBO(bool);
	void		renderToTexture();
	QImage		getTexture();

protected:
	virtual void initializeGL();
	virtual void resizeGL(int,int);
	virtual void paintGL();
	virtual void render() {}
	virtual void mousePressEvent(QMouseEvent*);
	virtual void mouseMoveEvent(QMouseEvent*);
	virtual void mouseReleaseEvent(QMouseEvent*);
	virtual void wheelEvent(QWheelEvent *);
	virtual void keyPressEvent(QKeyEvent*);
	virtual void keyReleaseEvent(QKeyEvent*);

	virtual void drawAxis(int options=0);

	void		pushMatrices();
	void		popMatrices();
	void		loadIdentities();

	bool		m_windowInit;
	GLProperty	m_glProp;
	int		m_width;
	int		m_height;
	float		m_whRatio;
	//bool		m_ifWidthDominate;

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
	bool		m_ifRecording;
	size_t		m_screenShotCounter;
	Camera::ProjectionType		m_cameraProjection;
	Box			box;
	void		drawBox();
	QColor		backgroundColor;

	void updateFreeMovement();
	bool m_freeMovementTimerunning;
	int m_freeMovement[8];

	QGLFramebufferObject * m_fbo;

private:
	QTime m_timer;
	int m_start;

public slots:
	virtual void renderNextStep() {}
	virtual void renderPreviousStep() {}
	virtual void renderNStep(size_t) {}
	void changeBGColor(const QColor&);

	void freeMovementTimer();
};

#endif
