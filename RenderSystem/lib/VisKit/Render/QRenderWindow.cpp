#include "QRenderWindow.h"
#include "CShader.h"
#include <QtGui>
#include <QGLFormat>
#include <QMouseEvent>
#include <QTime>
#include <QRect>
#include <QDesktopWidget>
#include <QImage>
using namespace std;

QRenderWindow::QRenderWindow(QWidget * parent, const QGLWidget * shareWidget)
:QGLWidget(parent, shareWidget)
{
	m_start = 0;
	//m_frames = 0;
	m_timer.start();
	m_windowInit = false;
	//m_ifWidthDominate = false;
	m_camera.setLook(0.5,0.5,0.5);
	m_camera.setFarclip(100.0);
	m_camera.setNearclip(0.1);
	m_camera.setDist(3);

	const QRect dSize = qApp->desktop()->screenGeometry();
	this->resize(dSize.width()/2,dSize.height()/2);
	m_ifRecording = false;
	m_screenShotCounter = 0;
	m_cameraProjection = Camera::Perspective;

	m_axisOptions = 3;
	m_drawBoundingBox = true;

	for(int i=0; i<8; ++i) m_freeMovement[i] = 0;
	m_freeMovementTimerunning = false;

	m_mousetarget = MTCamera;
	m_slicerIdx = -1;

	m_fbo = NULL;
}

QRenderWindow::QRenderWindow(const QGLFormat& format, QWidget * parent, const QGLWidget * shareWidget)
:QGLWidget(format, parent, shareWidget)
{
	m_start = 0;
	//m_frames = 0;
	m_timer.start();
	m_windowInit = false;
	//m_ifWidthDominate = false;
	m_camera.setLook(0.5,0.5,0.5);
	m_camera.setFarclip(100.0);
	m_camera.setNearclip(0.1);
	m_camera.setDist(3);

	const QRect dSize = qApp->desktop()->screenGeometry();
	this->resize(dSize.width()/2,dSize.height()/2);
	m_ifRecording = false;
	m_screenShotCounter = 0;
	m_cameraProjection = Camera::Perspective;

	m_axisOptions = 3;
	m_drawBoundingBox = true;

	for(int i=0; i<8; ++i) m_freeMovement[i] = 0;
	m_freeMovementTimerunning = false;

	m_mousetarget = MTCamera;
	m_slicerIdx = -1;

	m_fbo = NULL;
}

QRenderWindow::~QRenderWindow()
{
	makeCurrent();
	if (m_fbo) delete m_fbo;
}

void QRenderWindow::initializeGL()
{
	glewInit();
	qglClearColor(QColor::fromRgb(0,0,0));
}
void QRenderWindow::resizeGL(int width, int height)
{
	m_width = width;
	m_height = height;
	m_whRatio = (float)width/height;

	m_camera.setPerspective(45,width,height);

	for (int i = 0; i < m_slicers.size(); ++i)
		m_slicers[i].resize(width, height);

	if (m_fbo) delete m_fbo;
	m_fbo = new QGLFramebufferObject(50*width/height, 50);

	glViewport(0,0,width,height);
	update();
}
void QRenderWindow::paintGL()
{
	startFPS();
	render(); // must put render*() first since it clear depth/color buffer
	//drawAxis();
}

void QRenderWindow::changeBGColor(const QColor& newbg) {
	backgroundColor = newbg;
	qglClearColor(newbg);
	updateGL();
}

void QRenderWindow::drawAxis(int options)
{
	if(!options)
		options = m_axisOptions;

	pushMatrices();

	m_camera.push();
	m_camera.setFocus(0,0,0);
	if(options & 1) {
		glPushAttrib(GL_VIEWPORT_BIT);
		glViewport(0,0,100,100);

		m_camera.push();
		m_camera.setDist(1);
		m_camera.updateCamera();

		glBegin(GL_LINES);
		glColor3f(1,0,0);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.5, 0.0, 0.0);
		glColor3f(0,1,0);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.0, 0.5, 0.0);
		glColor3f(0,0,1);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.0, 0.0, 0.5);
		glEnd();

		glPopAttrib();
		m_camera.pop();
	}
	if(options & 2) {
		m_camera.updateCamera();
		glEnable(GL_DEPTH_TEST);
		// draw center axes
		glColor3f(1.0f,0.0f,0.0f);
		glBegin(GL_LINES);
		glVertex3f(-1.0f,0.0f,0.0f);
		glVertex3f( 1.0f,0.0f,0.0f);
		glEnd();
		glColor3f(0.0f,1.0f,0.0f);
		glBegin(GL_LINES);
		glVertex3f(0.0f,-1.0f,0.0f);
		glVertex3f(0.0f, 1.0f,0.0f);
		glEnd();
		glColor3f(0.0f,0.0f,1.0f);
		glBegin(GL_LINES);
		glVertex3f(0.0f,0.0f,-1.0f);
		glVertex3f(0.0f,0.0f, 1.0f);
		glEnd();
		glDisable(GL_DEPTH_TEST);
	}
	m_camera.pop();
	popMatrices();
}
void QRenderWindow::mousePressEvent(QMouseEvent* e)
{
	m_camera.start(e);
	if (m_slicerIdx > -1) m_slicers[m_slicerIdx].start(e, m_camera.getUp(), m_camera.getRight());
	e->accept();
}

void QRenderWindow::mouseReleaseEvent(QMouseEvent* e)
{
	e->accept();
}

void QRenderWindow::mouseMoveEvent(QMouseEvent* e)
{
	switch(m_mousetarget) {
	case MTSliceTrack:
		if (m_slicerIdx > -1) m_slicers[m_slicerIdx].track(e, m_camera.getCam(), m_camera.getUp(), m_camera.getRight());
		break;
	case MTSliceMove:
		if (m_slicerIdx > -1) m_slicers[m_slicerIdx].move(e, m_camera.getUp(), m_camera.getRight());
		break;
	case MTCamera:
	default:
		m_camera.track(e);
		break;
	}
	updateGL();
	e->accept();
}
void QRenderWindow::wheelEvent(QWheelEvent *e)
{
	m_camera.zoom(e->delta());
	updateGL();
	e->accept();
}
//void QRenderWindow::setShader(CShader *s)
//{
//	m_shader = s;
//	m_shader->setRenderWindow(this);
//}
void QRenderWindow::startFPS(){
	m_start = m_timer.elapsed();
}
float QRenderWindow::endFPS(){
	float fps = 0.0f;
	int end = m_timer.elapsed();
	if(end != m_start) {
		fps = 1000. / (float) (end - m_start);
	}
	return fps;
}
void QRenderWindow::getViewVec(float *v){
	if(m_camera.getProjection() == Camera::Ortho){
		v[0] = -m_camera.getCam().x();
		v[1] = -m_camera.getCam().y();
		v[2] = -m_camera.getCam().z();
	}
	else{
		v[0] = m_camera.CamX();
		v[1] = m_camera.CamY();
		v[2] = m_camera.CamZ();
	}
}
void QRenderWindow::takeScreenShot(){
	QString fn = QString("snapshot-%1.jpeg").arg(m_screenShotCounter++,4,10,QLatin1Char('0'));
	int hh = this->grabFrameBuffer().height();
	int ww = this->grabFrameBuffer().width();
	this->grabFrameBuffer().copy(0,0,ww,hh).save(fn);
}
void QRenderWindow::switchProjection(){
	if(m_cameraProjection == Camera::Perspective){
		m_camera.setProjection(Camera::Ortho);
		m_cameraProjection = Camera::Ortho;
	}
	else{
		m_camera.setProjection(Camera::Perspective);
		m_cameraProjection = Camera::Perspective;
	}
	updateGL();
}
void QRenderWindow::keyPressEvent(QKeyEvent* e) {
	if(e->key() == Qt::Key_W) {
		m_freeMovement[0] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_S) {
		m_freeMovement[1] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_A) {
		m_freeMovement[2] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_D) {
		m_freeMovement[3] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_Q) {
		m_freeMovement[4] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_E) {
		m_freeMovement[5] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_Space) {
		m_freeMovement[6] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_Control) {
		m_freeMovement[7] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
		}
	}
	QGLWidget::keyPressEvent(e);
}
void QRenderWindow::keyReleaseEvent(QKeyEvent* e) {
	if(e->key() == Qt::Key_P) {
		switchProjection();
	}
	else if(e->key() == Qt::Key_W){
		m_freeMovement[0] = 0;
	}
	else if(e->key() == Qt::Key_S){
		m_freeMovement[1] = 0;
	}
	else if(e->key() == Qt::Key_A){
		m_freeMovement[2] = 0;
	}
	else if(e->key() == Qt::Key_D){
		m_freeMovement[3] = 0;
	}
	else if(e->key() == Qt::Key_Q){
		m_freeMovement[4] = 0;
	}
	else if(e->key() == Qt::Key_E){
		m_freeMovement[5] = 0;
	}
	else if(e->key() == Qt::Key_Space){
		m_freeMovement[6] = 0;
	}
	else if(e->key() == Qt::Key_Control){
		m_freeMovement[7] = 0;
	}
	else if(e->key() == Qt::Key_F) {
		if(m_camera.getMode() == Camera::Fixed) m_camera.setMode(Camera::Free);
		else m_camera.setMode(Camera::Fixed);
		updateGL();
	}
	QGLWidget::keyReleaseEvent(e);
}

void QRenderWindow::pushMatrices() {
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
}

void QRenderWindow::popMatrices() {
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void QRenderWindow::loadIdentities() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}


void QRenderWindow::drawBox() {
	box.drawBox();
}

void QRenderWindow::freeMovementTimer() {
	updateFreeMovement();
	updateGL();
	if (m_freeMovement[0] || m_freeMovement[1] || m_freeMovement[2] || m_freeMovement[3] || m_freeMovement[4] || m_freeMovement[5] || m_freeMovement[6] || m_freeMovement[7]) {
		QTimer::singleShot(10, this, SLOT(freeMovementTimer()));
	}
	else {
		m_freeMovementTimerunning = false;
	}
}

void QRenderWindow::updateFreeMovement()
{
	if (m_freeMovement[0] != m_freeMovement[1]) {
		if (m_freeMovement[0]) m_camera.freeForward();
		else if (m_freeMovement[1]) m_camera.freeBackward();
	}
	if (m_freeMovement[2] != m_freeMovement[3]) {
		if (m_freeMovement[2]) m_camera.freeStrafeLeft();
		else if (m_freeMovement[3]) m_camera.freeStrafeRight();
	}
	if (m_freeMovement[4] != m_freeMovement[5]) {
		if (m_freeMovement[4]) m_camera.freeTiltLeft();
		else if (m_freeMovement[5]) m_camera.freeTiltRight();
	}
	if (m_freeMovement[6] != m_freeMovement[7]) {
		if (m_freeMovement[6]) m_camera.freeUp();
		else if (m_freeMovement[7]) m_camera.freeDown();
	}
}

void QRenderWindow::enableFBO(bool val) {
	if (!m_fbo) return;
	makeCurrent();
	if (val) m_fbo->bind();
	else m_fbo->release();
}

void QRenderWindow::renderToTexture() {
	if (!m_fbo) return;

	makeCurrent();
	int a = m_axisOptions;
	bool b = m_drawBoundingBox;
	m_axisOptions = 0;
	m_drawBoundingBox = false;

	enableFBO(true);
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	glViewport(0,0,50*m_whRatio,50);
	makeCurrent();
	updateGL();
	glFlush();
	glFinish();
	glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
	enableFBO(false);

	m_axisOptions = a;
	m_drawBoundingBox = b;
}

QImage QRenderWindow::getTexture() {
	if (!m_fbo) return QImage();
	renderToTexture();
	return m_fbo->toImage().rgbSwapped();
}
