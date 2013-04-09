#include "QHSVColorPanel.h"
#include <QPainter>
#include <QBrush>
#include <QPainterPath>
#include <QMouseEvent>
#include <QPointF>
#include <QMenu>
#include <QAction>

#include <cmath>
#include <iostream>
using namespace std;

QHSVColorPanel::QHSVColorPanel(int w,int h,QWidget *parent):
QWidget(parent),m_width(w),m_height(h),m_RADIUS(0.017453292f),m_HRADIUS(0.008726646f),m_Pi(3.1415926f), m_2Pi(6.2831852f)
{
	this->setFixedSize(m_width,m_height);
	this->resize(m_width,m_height);

	m_colorPickerImage = new QImage(m_width,m_height,QImage::Format_RGB32);
	
	m_bandWidth = 20;
	m_H = 0.0f;
	m_S = 1.0f;
	m_V = 1.0f;
	m_lastPoint = QPoint(m_width/2,0);
	m_dragH = false;
	m_dragSV = false;
	m_isLeftClick = false;
	m_isRightClick = false;
	m_showColorInfo = false;
	
	m_backgroundColor = QColor::fromRgb(200,200,200,255);
	
	initMenu();
}
void QHSVColorPanel::initMenu()
{
	m_optionMenu = new QMenu(this);
	
	QMenu	*m_backgroundColorMenu = new QMenu(tr("Background Color"),this);
	QAction	*m_bgnBlackAct,*m_bgnGrayAct,*m_bgnWhiteAct;	
	
    m_bgnBlackAct = new QAction(tr("Black"), this);
    connect(m_bgnBlackAct, SIGNAL(triggered()), this, SLOT(changeBackgroundToBlack()));	
	m_bgnGrayAct = new QAction(tr("Gray"), this);
    connect(m_bgnGrayAct, SIGNAL(triggered()), this, SLOT(changeBackgroundToGray()));	
	m_bgnWhiteAct = new QAction(tr("White"), this);
    connect(m_bgnWhiteAct, SIGNAL(triggered()), this, SLOT(changeBackgroundToWhite()));	
	
	m_backgroundColorMenu->addAction(m_bgnBlackAct);
	m_backgroundColorMenu->addAction(m_bgnGrayAct);
	m_backgroundColorMenu->addAction(m_bgnWhiteAct);
	
	turnOnOffColorInfoAct = new QAction(tr("Turn On Color Info"), this);
    connect(turnOnOffColorInfoAct, SIGNAL(triggered()), this, SLOT(turnOnOffColorInfo()));	

	m_optionMenu->addMenu(m_backgroundColorMenu);	
	m_optionMenu->addAction(turnOnOffColorInfoAct);
}
void QHSVColorPanel::paintEvent(QPaintEvent *)
{
	QPainter painter(this);
	
	// draw color picker image
	painter.drawImage(rect(), *m_colorPickerImage);
	
	QPen pen1(Qt::black, 1);
	QPen pen2(Qt::white, 1);
	QPen pen3(Qt::yellow, 1);
	
	painter.setPen(pen1);
	painter.drawEllipse(m_HPt,5,5);
	
	if(m_V <= 0.5)
		painter.setPen(pen2);
	else
		painter.setPen(pen1);
	painter.drawEllipse(m_SVPt,5,5);
	
	if(m_isLeftClick && (m_dragSV || m_dragH) && m_showColorInfo)
	{
		int hh,ss,vv;
		getHSVi(hh,ss,vv);
		QRect infoBlock(m_lastPoint.x()+5,m_lastPoint.y()+5,50,20);
		QString location = QString("H:%1,S:%2,V:%3").arg(hh).arg(ss).arg(vv);
		painter.setPen(pen3);
	//	painter.drawRect(infoBlock);
		QPoint nowPt;
		nowPt.setX(0);
		nowPt.setY(m_lastPoint.y()+5);
		painter.drawText(nowPt, location);
	}
}
void QHSVColorPanel::mouseMoveEvent(QMouseEvent* event)
{
	if(m_isLeftClick)
	{
		if(m_dragH)
		{
			m_lastPoint = event->pos();
			point2H();
			updateColorTriangle();
			emit hsvColorChangedH();
		}
		else if(m_dragSV)
		{
			m_lastPoint = event->pos();
			point2SV();
			updateColorTriangle();
			emit hsvColorChangedSV();
		}
		else
		{
		}
		repaint();
	}
}
void QHSVColorPanel::mousePressEvent(QMouseEvent* event)
{
	if(event->button() == Qt::LeftButton)
		m_isLeftClick = true;
	else if(event->button() == Qt::RightButton)
		m_isRightClick = true;
	
	if(m_isLeftClick == true)
	{
		if(m_colorTriagle.containsPoint(event->pos(),Qt::OddEvenFill))
		{
			m_dragSV = true;
			m_dragH = false;
			m_lastPoint = event->pos();
			point2SV();
			emit hsvColorChangedSV();
		}
		else
		{
			m_dragSV = false;
			m_dragH = true;	
			m_lastPoint = event->pos();
			point2H();
			updateColorTriangle();
			emit hsvColorChangedH();
		}
		repaint();
	}
	else if(m_isRightClick == true)
	{
		//QPoint nowPt = event->pos();
		//mapToGlobal(nowPt);
		m_optionMenu->exec(QCursor::pos());
	}
}
void QHSVColorPanel::mouseReleaseEvent(QMouseEvent* event)
{
	if(event->button() == Qt::LeftButton)
		m_isLeftClick = false;
	else if(event->button() == Qt::RightButton)
		m_isRightClick = false;
	
	m_dragSV = false;
	m_dragH = false;
}
void QHSVColorPanel::resizeEvent(QResizeEvent *event)
{
	m_width = rect().width();
	m_height = rect().height();
	calculateGeometry();
	resizeImage(m_colorPickerImage,QSize(rect().width(),rect().height()));
	updateColorBend();
	updateColorTriangle();
	QWidget::resizeEvent(event);	
}
void QHSVColorPanel::updateColorBend()
{
	QPainter painter(m_colorPickerImage);
	m_colorPickerImage->fill(m_backgroundColor.rgb());
	int doubleBand = m_bandWidth * 2;
	// draw circle
	for(int s=0;s<720;s++)
	{
		for(int t=0;t<doubleBand;t++)
		{
			QColor	qc = QColor::fromHsvF((float)s/720.0,1,1);
			painter.setPen(qc);
			float x = (m_bandRadiusB - t/2.0) * cos(m_HRADIUS * s);
			float y = (m_bandRadiusB - t/2.0) * sin(m_HRADIUS * s);
			painter.drawPoint(m_center.x() + x, m_center.y() - y);
		}
	}
}
void QHSVColorPanel::updateColorTriangle()
{
	QPainter painter(m_colorPickerImage);
	
	// draw triangular
	painter.setPen(m_backgroundColor.rgb());	
	painter.drawPolygon(m_colorTriagle);	
	
	// fill triangle
	float startx = 27;//m_triPtB.x();
	float endx = 122;//m_triPtC.x();

	if (m_triHeight > 1000) return;

	for(int h=m_triHeight;h>=0;--h)
	{
		float tempWidth = fabs(endx - startx);
		//float leftmostx = startx;
		float vh = (float)h/(float)m_triHeight;
		for(int w=startx;w<endx;++w)
		{
			float vw = (w-startx)/tempWidth;
			if(vw > 1.0f)	vw = 1.0f;
			else if(vw < 0.0f) vw = 0.0f;
			painter.setPen(QColor::fromHsvF(m_H,vw,vh,1));
			painter.drawPoint(QPoint(w,h+m_bandWidth));
		}
		startx += m_triStep;
		endx -= m_triStep;	
	}	
}
void QHSVColorPanel::resizeImage(QImage *image, const QSize &newSize)
{
	if(image->size() == newSize)
		return;
	QImage newImage(newSize, QImage::Format_RGB32);
	newImage.fill(qRgb(255, 255, 255));
	QPainter painter(&newImage);
	//painter.drawImage(QPoint(0, 0), *image);
	//*image = newImage;
	delete image;
	image = new QImage(newImage);
}
void QHSVColorPanel::calculateGeometry()
{
	if(m_width != m_height)
	{
		if(m_width < m_height)
			m_height = m_width;
		else
			m_width = m_height;
	}
	
	m_center = QPoint(m_width/2,m_height/2);
	m_bandRadiusB = m_width/2;
	m_bandRadiusL = (m_bandRadiusB - m_bandWidth);
	m_bandRadiusM = m_bandRadiusB - m_bandWidth/2;
	
	m_triPtA = QPoint(75,20);//QPoint(m_center.x() + m_bandRadiusL*cos(m_RADIUS * 90.0),  
			   //		  m_center.y() - m_bandRadiusL*sin(m_RADIUS * 90.0));
	m_triPtB = QPoint(27,102);//QPoint(m_center.x() + m_bandRadiusL*cos(m_RADIUS * 210.0), 
			   //		  m_center.y() - m_bandRadiusL*sin(m_RADIUS * 210.0));
	m_triPtC = QPoint(122,102);//QPoint(m_center.x() + m_bandRadiusL*cos(m_RADIUS * 330.0), 
			   //         m_center.y() - m_bandRadiusL*sin(m_RADIUS * 330.0));
	
	m_triWidth = 95;//abs(m_triPtC.x() - m_triPtB.x());
	m_triHeight = 82;//abs(m_triPtC.y() - m_triPtA.y());
	
	m_colorTriagle.clear();
	m_colorTriagle << m_triPtA << m_triPtB << m_triPtC;

	m_baseVec = QPoint(m_width/2,0);
	
	m_triStep = 0.5*m_triWidth/m_triHeight;
	// update HSV to points
	h2Point();
	sv2Point();
}
void QHSVColorPanel::h2Point()
{
	float deg = m_H * 360.0;///255.0;
	m_HPt = QPoint(m_center.x() + m_bandRadiusM*cos(m_RADIUS * deg),  
				   m_center.y() - m_bandRadiusM*sin(m_RADIUS * deg));
	//cout << "h2Point:H(x,y)=" << m_HPt.x() <<","<<m_HPt.y()<<endl;
}

void QHSVColorPanel::point2H()
{
	m_mousePosVec = m_lastPoint - m_center;
	double dott = m_baseVec.x()*m_mousePosVec.x() + m_baseVec.y()*m_mousePosVec.y();
	double da = sqrt((double)(m_baseVec.x()*m_baseVec.x() + m_baseVec.y()*m_baseVec.y()));
	double db = sqrt((double)(m_mousePosVec.x()*m_mousePosVec.x() + m_mousePosVec.y()*m_mousePosVec.y()));
	double det = dott/(da*db);
	double radian = acos(det);
	double r1 = m_bandRadiusM*cos(radian);
	double r2 = m_bandRadiusM*sin(radian);
	if(det >= 0)
	{
		//if(radian <= 1.5707963)
		if(m_lastPoint.y() < m_center.y())
		{
			m_HPt = QPoint(m_center.x() + r1, m_center.y() - r2);
			m_H = radian/(m_2Pi);	
		}
		else
		{
			m_HPt = QPoint(m_center.x() + r1, m_center.y() + r2);
			m_H = ((m_Pi - radian)+m_Pi)/(m_2Pi);
		}
	}
	else
	{
		if(m_lastPoint.y() < m_center.y())
		{
			m_HPt = QPoint(m_center.x() + r1, m_center.y() - r2);
			m_H = radian/(m_2Pi);
		}
		else
		{
			m_HPt = QPoint(m_center.x() + r1, m_center.y() + r2);
			m_H = ((m_Pi - radian)+m_Pi)/(m_2Pi);
		}
	}
	
	//cout << "point2H:m_H="<<m_H<<endl;
}
void QHSVColorPanel::sv2Point()
{
	float heightV = m_V * m_triHeight;
	int VV = m_triPtA.y() + heightV;
	
	float startx = m_triPtB.x() + abs(heightV-m_triHeight) * m_triStep;
	float endx = m_triPtC.x() - abs(heightV-m_triHeight) * m_triStep;
	float widthS = fabs(endx-startx);
	int SS = startx + m_S * widthS;
	
	m_SVPt = QPoint(SS,VV);
	
	//cout << "sv2Point:S pt=" << SS <<", V pt=" << VV << endl;
}
void QHSVColorPanel::point2SV()
{
	float heightV = m_lastPoint.y() - m_triPtA.y();
	m_V = heightV/m_triHeight;

	if(m_V > 1.0f) m_V = 1.0f;
	else if(m_V < 0.0f) m_V = 0.0f;
	float startx = m_triPtB.x() + abs(heightV-m_triHeight) * m_triStep;
	float endx = m_triPtC.x() - abs(heightV-m_triHeight) * m_triStep;
	m_S = (m_lastPoint.x() - startx) / abs(endx-startx);
	if(m_S < 0.0f)
		m_S = 0.0f;
	else if(m_S > 1.0f)
		m_S = 1.0f;
	
	if(m_colorTriagle.containsPoint(m_lastPoint,Qt::OddEvenFill))
	{
		m_SVPt = m_lastPoint;
	}
	else
	{
		
	}
	
	//cout << "point2SV:m_V="<<m_V<<",m_S="<<m_S<<endl;
		
}
void QHSVColorPanel::getHSVf(float &h,float &s,float &v) //0~1
{
	h = m_H;
	s = m_S;
	v = m_V;
}
void QHSVColorPanel::getRGBf(float &r,float &g,float &b) //0~1
{
	qreal rr,gg,bb;
	QColor temp = QColor::fromHsvF(m_H,m_S,m_V);
	temp.getRgbF(&rr,&gg,&bb);
	r = rr;
	g = gg;
	b = bb;
}
void QHSVColorPanel::getHSVi(int &h,int &s,int &v) // 0~255
{
	h = 255*m_H;
	s = 255*m_S;
	v = 255*m_V;
}
void QHSVColorPanel::getRGBi(int &r,int &g,int &b) // 0~255
{
	QColor temp = QColor::fromHsvF(m_H,m_S,m_V);
	temp.getRgb(&r,&g,&b);
}
void QHSVColorPanel::updateColorPanelHSVf(float &h,float &s,float &v)
{
	m_H = h;
	m_S = s;
	m_V = v;
	
	h2Point();
	sv2Point();
	updateColorTriangle();

	repaint();
}
void QHSVColorPanel::updateColorPanelHf(float &h)
{
	m_H = h;
	h2Point();
	updateColorTriangle();
	emit hsvColorChangedH();
	repaint();
}
void QHSVColorPanel::updateColorPanelSVf(float &s,float &v)
{
	m_S = s;
	m_V = v;
	sv2Point();
	updateColorTriangle();
	emit hsvColorChangedSV();
	repaint();
}
void QHSVColorPanel::changeBackgroundToBlack()
{
	m_backgroundColor = QColor::fromRgb(1,1,1,255);
	updateColorBend();
	updateColorTriangle();
	repaint();
}

void QHSVColorPanel::changeBackgroundToGray()
{
	m_backgroundColor = QColor::fromRgb(200,200,200,255);
	updateColorBend();
	updateColorTriangle();
	repaint();
}
void QHSVColorPanel::changeBackgroundToWhite()
{
	m_backgroundColor = QColor::fromRgb(255,255,255,255);
	updateColorBend();
	updateColorTriangle();
	repaint();
}
void QHSVColorPanel::turnOnOffColorInfo()
{
	if(m_showColorInfo == true) // now is on, we want to turn it off
	{
		m_showColorInfo = false;
		turnOnOffColorInfoAct->setText(tr("Turn On Color Info"));
	}
	else
	{
		m_showColorInfo = true;
		turnOnOffColorInfoAct->setText(tr("Turn Off Color Info"));	
	}
}
QColor	QHSVColorPanel::getQColor()
{
	QColor temp = QColor::fromHsvF(m_H, m_S, m_V, 1.0);
	return temp;
}
void	QHSVColorPanel::setQColor(QColor	&cr)
{
	qreal hh,ss,vv;
	cr.getHsvF(&hh,&ss,&vv);
	float hhf,ssf,vvf;
	hhf = (float)hh;
	ssf = (float)ss;
	vvf = (float)vv;
	updateColorPanelHf(hhf);
	updateColorPanelSVf(ssf,vvf);
}
