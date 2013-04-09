#include "QRenderWidget.h"
#include <GL/glew.h>
#include <QTimer>
#include <QMouseEvent>
using namespace std;

void MyTimer::freeMovementTimer() {
	if (!m_renderWidget) return;
	m_renderWidget->freeMovementTimer();
}

QRenderWidget::QRenderWidget()
{
	m_start = 0;
	m_timer.start();
	m_camera.setLook(0.5,0.5,0.5);
	m_camera.setFarclip(100.0);
	m_camera.setNearclip(0.1);
	m_camera.setDist(3);
	m_cameraProjection = Camera::Perspective;

	m_axisOptions = 3;
	m_drawBoundingBox = true;
	m_mousetarget = MTCamera;

	for(int i=0; i<8; ++i) m_freeMovement[i] = 0;
	m_freeMovementTimerunning = false;
	m_freeMovementTimer.setRenderer(this);

	m_slicerIdx = -1;
}

QRenderWidget::~QRenderWidget()
{
}

void QRenderWidget::resizeEvent(int width, int height)
{
	if (width == 0 || height == 0 ) return;
	m_camera.setPerspective(45,width,height);
	for (int i = 0; i < m_slicers.size(); ++i)
		m_slicers[i].resize(width,height);
}

void QRenderWidget::drawAxis(int options)
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
void QRenderWidget::mousePressEvent(QMouseEvent* e)
{
	m_camera.start(e);
	if (m_slicerIdx > -1) m_slicers[m_slicerIdx].start(e, m_camera.getUp(), m_camera.getRight());
	e->accept();
}

void QRenderWidget::mouseReleaseEvent(QMouseEvent* e)
{
	e->accept();
}

void QRenderWidget::mouseMoveEvent(QMouseEvent* e)
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
	renderWidgetUpdated();
	e->accept();
}
void QRenderWidget::wheelEvent(QWheelEvent *e)
{
	m_camera.zoom(e->delta());
	renderWidgetUpdated();
	e->accept();
}
void QRenderWidget::startFPS(){
	m_start = m_timer.elapsed();
}
float QRenderWidget::endFPS(){
	float fps = 0.0f;
	int end = m_timer.elapsed();
	if(end != m_start) {
		fps = 1000. / (float) (end - m_start);
	}
	return fps;
}
void QRenderWidget::getViewVec(float *v){
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
void QRenderWidget::switchProjection(){
	if(m_cameraProjection == Camera::Perspective){
		m_camera.setProjection(Camera::Ortho);
		m_cameraProjection = Camera::Ortho;
	}
	else{
		m_camera.setProjection(Camera::Perspective);
		m_cameraProjection = Camera::Perspective;
	}
	renderWidgetUpdated();
}
void QRenderWidget::keyPressEvent(QKeyEvent* e) {
 	if(e->key() == Qt::Key_W) {
		m_freeMovement[0] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_S) {
		m_freeMovement[1] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_A) {
		m_freeMovement[2] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_D) {
		m_freeMovement[3] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_Q) {
		m_freeMovement[4] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_E) {
		m_freeMovement[5] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_Space) {
		m_freeMovement[6] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
		}
	}
	else if(e->key() == Qt::Key_Control) {
		m_freeMovement[7] = 1;
		if (!m_freeMovementTimerunning) {
			m_freeMovementTimerunning = true;
			QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
		}
	}
}
void QRenderWidget::keyReleaseEvent(QKeyEvent* e) {
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
		renderWidgetUpdated();
	}
}

void QRenderWidget::pushMatrices() {
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
}

void QRenderWidget::popMatrices() {
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void QRenderWidget::loadIdentities() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}


void QRenderWidget::drawBox() {
	box.drawBox();
}

void QRenderWidget::freeMovementTimer() {
	updateFreeMovement();
	renderWidgetUpdated();
	if (m_freeMovement[0] || m_freeMovement[1] || m_freeMovement[2] || m_freeMovement[3] || m_freeMovement[4] || m_freeMovement[5] || m_freeMovement[6] || m_freeMovement[7]) {
		QTimer::singleShot(10, &m_freeMovementTimer, SLOT(freeMovementTimer()));
	}
	else {
		m_freeMovementTimerunning = false;
	}
}

void QRenderWidget::updateFreeMovement()
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
