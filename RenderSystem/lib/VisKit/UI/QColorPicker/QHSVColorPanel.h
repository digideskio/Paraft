#ifndef _QHSVCOLORPANEL_H_
#define _QHSVCOLORPANEL_H_

#include <QWidget>
#include <QPoint>
#include <QPolygonF>

class QMenu;
class QAction;
class QImage;
class QHSVColorPanel : public QWidget
{
	Q_OBJECT
public:
	QHSVColorPanel(int w,int h,QWidget *parent=0);

	void	getHSVf(float &h,float &s,float &v); //0~1
	void	getRGBf(float &r,float &g,float &b); //0~1
	void	getHSVi(int &h,int &s,int &v); // 0~100
	void	getRGBi(int &r,int &g,int &b); // 0~255
	QColor	getQColor();
	void	setQColor(QColor	&cr);
	
	void	updateColorPanelHSVf(float &h,float &s,float &v);
	void	updateColorPanelHf(float &h);
	void	updateColorPanelSVf(float &s,float &v);

	int			m_width, m_height;	
	int			m_bandWidth; // color band width
	QPoint		m_center;
	int			m_bandRadiusB, // outter band radius 
				m_bandRadiusL, // inner band radius
				m_bandRadiusM; // between outter and inner radius 
	QPoint		m_triPtA, m_triPtB, m_triPtC;
	int			m_triWidth, m_triHeight;
	
	QImage		*m_colorPickerImage;	
	
	const float m_RADIUS, m_HRADIUS;
	const float	m_Pi, m_2Pi;
	
	QPoint	m_lastPoint;
	float	m_H, m_S, m_V;
	QPoint	m_HPt, m_SVPt;
	QPolygonF	m_colorTriagle;
	float	m_triStep;
	
	void	calculateGeometry();
	void	updateColorBend();
	void	updateColorTriangle();
	void	resizeImage(QImage *image, const QSize &newSize);
	void	h2Point();
	void	sv2Point();
	void	point2H();
	void	point2SV();
	QPoint	m_baseVec, m_mousePosVec;
	
	bool	m_dragH, m_dragSV;
	bool	m_isLeftClick, m_isRightClick;
	
	QMenu	*m_optionMenu;
	//QMenu	*m_backgroundColorMenu;
	//QAction	*m_bgnBlackAct,*m_bgnGrayAct,*m_bgnWhiteAct;
	
	QColor	m_backgroundColor;
	
	bool	m_showColorInfo;
	QAction	*turnOnOffColorInfoAct;
private:
	void initMenu();
	
public slots:
	void changeBackgroundToBlack();
	void changeBackgroundToGray();
	void changeBackgroundToWhite();
	
	void turnOnOffColorInfo();

protected:
	void paintEvent(QPaintEvent *);
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);	
	void resizeEvent(QResizeEvent *);

signals:
	void	hsvColorChangedH();
	void	hsvColorChangedSV();
};

#endif
