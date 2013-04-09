#ifndef _QCOLORPICKER_H_
#define _QCOLORPICKER_H_

#include <QWidget>

using namespace std;

class QHSVColorPanel;
class QColorMonitor;
class QColorPalette;

class QColorPicker : public QWidget
{
	Q_OBJECT
public:
	QColorPicker(QWidget *parent=0);
	~QColorPicker();
	
	void	getHSVf(float &h,float &s,float &v); //0~1
	void	getRGBf(float &r,float &g,float &b); //0~1
	void	getHSVi(int &h,int &s,int &v); // 0~255
	void	getRGBi(int &r,int &g,int &b); // 0~255
	QColor	getQColor();
	void	setQColor(QColor	&cr);
	
	void	updateColorPanelHSVf(float &h,float &s,float &v);
	void	updateColorPanelHf(float &h);
	void	updateColorPanelSVf(float &s,float &v);
	
	void	getColorPanelGeometry(QRect &rect);

private:
	QColorMonitor	*m_monitor;
	QHSVColorPanel	*m_hsvPanel;
	QColorPalette	*m_colorPalette;

	void	initLayout();
	void	readSettings();
	void	writeSettings();

public slots:
	void	panelColorChangedH();
	void	panelColorChangedSV();
signals:
	void	colorChanged(int);
	void	colorPassiveChanged(int);
};

#endif

