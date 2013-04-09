#include "QHistogram.h"
#include "QTFEditor.h"
#include "histogram.h"
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QImage>
#include <QRect>
#include <QResizeEvent>
#include <QSize>
#include <QBrush>
#include <QMenu>
#include <cmath>
using namespace std;

template <typename T>
inline T clamp(T v, T min, T max) {
	return (v < min ? min : (v > max ? max : v));
}

#define EPSILON 0.0001
bool Range::isOnEdge(float value) {
	return (fabs(min - value) < EPSILON || fabs(max - value) < EPSILON);
}

QHistogram::QHistogram(QWidget *parent)
:QTFAbstractPanel(parent)
{
	this->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
	this->setMinimumSize(400,100);
	this->setMaximumHeight(300);

	m_histogramType = 1; 
	m_backgroundMesh = 1; // dot
	m_yTransform = 0; // linear
	//m_pHistogramData = m_pTFEditor->m_histogramData;
	initLayout();
	initMenu();
	changeStyle2Line();
	changeBM2DotLine();
	changeYT2Linear();
	m_panelName = QString("Histogram");
}

void QHistogram::initLayout()
{
	QTFAbstractPanel::initLayout(25,1,25,5);
}
void QHistogram::initMenu()
{
	m_optionMenu = new QMenu(this);
	QMenu	*styleMenu = new QMenu(tr("Style"),this);

	changeStyle2LineAct = new QAction(tr("Line"),this);
	changeStyle2LineAct->setCheckable(true);
	connect(changeStyle2LineAct, SIGNAL(triggered()), this, SLOT(changeStyle2Line()));	
	changeStyle2BarAct = new QAction(tr("Bar"),this);
	changeStyle2BarAct->setCheckable(true);
	connect(changeStyle2BarAct, SIGNAL(triggered()), this, SLOT(changeStyle2Bar()));	
	
	styleMenu->addAction(changeStyle2LineAct);
	styleMenu->addAction(changeStyle2BarAct);

	QMenu	*backgroundMeshMenu = new QMenu(tr("Background Grid"),this);
	changeBM2NoneAct = new QAction(tr("None"),this);
	changeBM2NoneAct->setCheckable(true);
	connect(changeBM2NoneAct, SIGNAL(triggered()), this, SLOT(changeBM2None()));
	changeBM2DotLineAct = new QAction(tr("Dot Line"),this);
	changeBM2DotLineAct->setCheckable(true);
	connect(changeBM2DotLineAct, SIGNAL(triggered()), this, SLOT(changeBM2DotLine()));
	backgroundMeshMenu->addAction(changeBM2NoneAct);
	backgroundMeshMenu->addAction(changeBM2DotLineAct);
	
	QMenu	*yTransformMenu = new QMenu(tr("Y Axis Transform"),this);
	changeYT2LinearAct = new QAction(tr("Linear"),this);
	changeYT2LinearAct->setCheckable(true);
	connect(changeYT2LinearAct, SIGNAL(triggered()), this, SLOT(changeYT2Linear()));
	changeYT2LogAct = new QAction(tr("Log_10"),this);
	changeYT2LogAct->setCheckable(true);
	connect(changeYT2LogAct, SIGNAL(triggered()), this, SLOT(changeYT2Log()));
	yTransformMenu->addAction(changeYT2LinearAct);
	yTransformMenu->addAction(changeYT2LogAct);	
	
	m_optionMenu->addMenu(styleMenu);
	m_optionMenu->addMenu(backgroundMeshMenu);
	m_optionMenu->addMenu(yTransformMenu);

	m_zeroMenu = new QMenu(this);
	removeZeroRangeAct = new QAction(tr("Remove Zero Range"), this);
	connect(removeZeroRangeAct, SIGNAL(triggered()),
			this, SLOT(removeZeroRange()));
	toggleZeroLockAct = new QAction(tr("Toggle lock"), this);
	connect(toggleZeroLockAct, SIGNAL(triggered()),
			this, SLOT(toggleZeroLock()));
	m_zeroMenu->addAction(removeZeroRangeAct);
	m_zeroMenu->addAction(toggleZeroLockAct);
}

void QHistogram::realPaintEvent(QPaintEvent *)
{
	QPainter painter(this);
	painter.drawImage(rect(),*m_panelImage);
	Histogram* histogram = m_pTFEditor->getHistogram();

	if(histogram != NULL)
	{
		if(m_histogramType == 2) //bar
		{
			painter.setBrush(QColor(255,255,0,150));//Qt::yellow);
			painter.setPen(Qt::blue);
			for(unsigned int x=0;x<(unsigned int)histogram->getLength();++x)
			{
				float binheight = 1;
				if(m_yTransform == 1)
					binheight = 1e-5 + (log10((float)(*histogram)[x] + 1)/m_histogramMax) * m_binMaxHeight;
				else
					binheight = ((float)(*histogram)[x])/m_histogramMax * m_binMaxHeight;
				float xpos = (float)(m_panelLMargin + 1)+ (float)x * m_binWidth;
				float ypos = m_panelUMargin + m_panelHeight - binheight;
				QRect binobj(xpos,ypos,m_binWidth,binheight);
				painter.drawRect(binobj);
			}
		}
		else if(m_histogramType == 1) // curve
		{
			if(!m_histogramMax)
				return;
			painter.setBrush(QColor(255,255,0,150));
			painter.setPen(Qt::blue);	
			QPainterPath path;
			path.moveTo(m_panelLMargin,m_panelUMargin+m_panelHeight);
			
			for(unsigned int x=0;x<(unsigned int)histogram->getLength();++x)
			{
				float binheight = 1;
				if(m_yTransform == 1)
					binheight = 1e-5 + (log10((float)(*histogram)[x] + 1)/m_histogramMax) * m_binMaxHeight;
				else
					binheight = ((float)(*histogram)[x])/m_histogramMax * m_binMaxHeight;
				float xpos = (float)(m_panelLMargin)+ (float)x * m_binWidth + 0.5*m_binWidth;
				float ypos = m_panelUMargin + m_panelHeight - binheight;
				path.lineTo(xpos,ypos);
			}
			
			path.lineTo(rect().width()-m_panelRMargin,m_panelUMargin+m_panelHeight);
			painter.drawPath(path);
		}
	}
}

void QHistogram::updateHistogram()
{
	Histogram* histogram = m_pTFEditor->getHistogram();
	if(histogram != NULL)
	{
		m_histogramMax = -1e10;
		m_histogramMin =  1e10;
	
		m_binWidth = (float)(m_panelWidth - 2)/ histogram->getLength();
		m_binMaxHeight = 0.98*m_panelHeight;
		
		if(m_yTransform == 0) // linear
		{	
			for(unsigned int x=0;x<histogram->getLength();++x)
			{
				if(m_histogramMax < (*histogram)[x])
					m_histogramMax = (*histogram)[x];
				if(m_histogramMin > (*histogram)[x])
					m_histogramMin = (*histogram)[x];
			}
		}
		else if(m_yTransform == 1) // log
		{
			for(unsigned int x=0;x<histogram->getLength();++x)
			{
				if(m_histogramMax < log10((float)(*histogram)[x] + 1))
					m_histogramMax = log10((float)(*histogram)[x] + 1)+1e-10;
				if(m_histogramMin > log10((float)(*histogram)[x] + 1))
					m_histogramMin = log10((float)(*histogram)[x] + 1)+1e-10;
				
				if(fabs(m_histogramMin) < 1e-10f)
					m_histogramMin = 1e-10f;
			}
		}

		repaint();
	}
}

void QHistogram::updatePanelImage()
{
	QPainter painter(m_panelImage);
	m_panelImage->fill(qRgb(255, 255, 255));

	QPen pen1(Qt::black, 1);
	painter.setPen(pen1);
	painter.drawRect(m_panelLMargin,m_panelUMargin,m_panelWidth,m_panelHeight);
	
	if(m_backgroundMesh == 1) // mesh
	{
		QPen	pen2(Qt::gray,1);
		pen2.setStyle(Qt::DashLine);
		painter.setPen(pen2);
		
		QPointF	px1,px2,py1,py2;
		for(int x=1;x<10;++x)
		{
			px1 = QPointF(m_panelLMargin + ((float)m_panelWidth/10.0f) * x,m_panelUMargin);
			px2 = QPointF(m_panelLMargin + ((float)m_panelWidth/10.0f) * x,m_panelUMargin+m_panelHeight);
			painter.drawLine(px1,px2);
			
			py1 = QPointF(m_panelLMargin,m_panelUMargin+((float)m_panelHeight/10.0f) * x);
			py2 = QPointF(rect().width()-m_panelRMargin,m_panelUMargin+((float)m_panelHeight/10.0f) * x);
			painter.drawLine(py1,py2);
		}
	}
	updateHistogram();
}
float QHistogram::convertX(int value) {
	return clamp((value - m_panelLMargin)/(m_panelWidth - 2.f), 0.f, 1.f);
}
void QHistogram::mousePressEvent(QMouseEvent* event)
{
	if(event->button() == Qt::LeftButton) {
		if(inZero(event->x())) {
			if(isOnZeroEdge(event->x())) {
				m_tempZero = (*m_currentZero);
				m_zeros.erase(m_currentZero);
				if(fabs(convertX(event->x()) - m_tempZero.min) < EPSILON) { //we drag tempZero's max around when drawing the temp
					float temp = m_tempZero.min;
					m_tempZero.min = m_tempZero.max;
					m_tempZero.max = temp;
				}
				m_drawingZero = true;
			} else
				m_zeroMenu->exec(QCursor::pos());
		} else
			startZero(event->x());
	}
	else if(event->button() == Qt::RightButton)
	{
		m_optionMenu->exec(QCursor::pos());
	}
}
void QHistogram::removeZeroRange() {
	if(m_currentZero != m_zeros.end())
		m_currentZero = m_zeros.erase(m_currentZero);
}
void QHistogram::toggleZeroLock() {
}
void QHistogram::mouseMoveEvent(QMouseEvent* e) {
	drawZero(e->x());
}
void QHistogram::mouseReleaseEvent(QMouseEvent* e) {
	endZero(e->x());
}
void QHistogram::drawZero(int) {
	if(!m_drawingZero)
		return;

}
void QHistogram::startZero(int v) {
	m_tempZero.min = convertX(v);
	m_tempZero.max = convertX(v);
	m_drawingZero = true;
}
void QHistogram::endZero(int v) {
	if(!m_drawingZero)
		return;
	m_drawingZero = false;

	m_tempZero.max = convertX(v);
	if(m_tempZero.max == m_tempZero.min)
		return;
	if(m_tempZero.max < m_tempZero.min) { //in wrong order, swap
		float temp = m_tempZero.max;
		m_tempZero.max = m_tempZero.min;
		m_tempZero.min = temp;
	}
	if(inZero(m_tempZero.min)) { //connected with another zero, merge
		(*m_currentZero).merge(m_tempZero);
		for(QList<Range>::iterator it = m_zeros.begin(); it != m_zeros.end(); it++) { //check to see if any ranges got eaten
			if(it == m_currentZero) //ignore current range
				continue;
			while((*m_currentZero).inside((*it).min)) { //eaten? remove
				it = m_zeros.erase(it);
			}
		}
	}
}
bool QHistogram::isOnZeroEdge(int value) {
	float x = convertX(value);
	for(m_currentZero = m_zeros.begin(); m_currentZero != m_zeros.end(); m_currentZero++) {
		if((*m_currentZero).isOnEdge(x))
			return true;
	}
	return false;
}
bool QHistogram::inZero(int value) {
	float x = convertX(value);
	for(m_currentZero = m_zeros.begin(); m_currentZero != m_zeros.end(); m_currentZero++) {
		if((*m_currentZero).inside(x))
			return true;
	}
	return false;
}
void QHistogram::changeStyle2Line()
{
	changeStyle2LineAct->setChecked(true);
	changeStyle2BarAct->setChecked(false);
	m_histogramType = 1;
	repaint();
}
void QHistogram::changeStyle2Bar()
{
	changeStyle2LineAct->setChecked(false);
	changeStyle2BarAct->setChecked(true);	
	m_histogramType = 2;
	repaint();
}
void QHistogram::changeBM2None()
{
	changeBM2NoneAct->setChecked(true);
	changeBM2DotLineAct->setChecked(false);		
	m_backgroundMesh = 0;
	updatePanelImage();
	repaint();
}
void QHistogram::changeBM2DotLine()
{
	changeBM2NoneAct->setChecked(false);
	changeBM2DotLineAct->setChecked(true);	
	m_backgroundMesh = 1;
	updatePanelImage();
	repaint();
}
void QHistogram::changeYT2Linear()
{
	changeYT2LinearAct->setChecked(true);
	changeYT2LogAct->setChecked(false);
	m_yTransform = 0;
	updateHistogram();
	repaint();
}
void QHistogram::changeYT2Log()
{
	changeYT2LinearAct->setChecked(false);
	changeYT2LogAct->setChecked(true);
	m_yTransform = 1;
	updateHistogram();
	repaint();
}
