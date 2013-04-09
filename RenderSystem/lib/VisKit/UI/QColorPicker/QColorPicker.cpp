#include "QColorPicker.h"
#include "QHSVColorPanel.h"
#include "QColorMonitor.h"
#include "QColorPalette.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFile>
#include <QTextStream>
using namespace std;

QColorPicker::QColorPicker(QWidget *parent)
:QWidget(parent)
{
	this->setFixedSize(400,190);
	
	initLayout();
}
QColorPicker::~QColorPicker()
{
	writeSettings();
	delete m_colorPalette;
	delete m_monitor;
	delete m_hsvPanel;
}
void QColorPicker::initLayout()
{
	QHBoxLayout	*hlayout = new QHBoxLayout(this);
	hlayout->setSpacing(0);
	hlayout->setMargin(0);

	QVBoxLayout *vlayout = new QVBoxLayout;
	
	vlayout->setAlignment(Qt::AlignTop|Qt::AlignLeft);
	
	m_hsvPanel = new QHSVColorPanel(150,150,this);
	m_colorPalette = new QColorPalette(this);
	m_monitor = new QColorMonitor(this);
		
	connect(m_hsvPanel,SIGNAL(hsvColorChangedH()),this,SLOT(panelColorChangedH()));
	connect(m_hsvPanel,SIGNAL(hsvColorChangedSV()),this,SLOT(panelColorChangedSV()));
	
	vlayout->addWidget(m_hsvPanel);
	vlayout->addWidget(m_colorPalette);
	
	hlayout->addLayout(vlayout);
	hlayout->addWidget(m_monitor);
	setLayout(hlayout);
	
	readSettings();
}

void QColorPicker::panelColorChangedH()
{
	emit colorChanged(1);
}
void QColorPicker::panelColorChangedSV()
{
	emit colorChanged(2);
}
void QColorPicker::getHSVf(float &h,float &s,float &v) //0~1
{
	m_hsvPanel->getHSVf(h,s,v);
}
void QColorPicker::getRGBf(float &r,float &g,float &b) //0~1
{
	m_hsvPanel->getRGBf(r,g,b);
}
void QColorPicker::getHSVi(int &h,int &s,int &v) // 0~100
{
	m_hsvPanel->getHSVi(h,s,v);
}
void QColorPicker::getRGBi(int &r,int &g,int &b) // 0~255
{
	m_hsvPanel->getRGBi(r,g,b);
}
void QColorPicker::updateColorPanelHSVf(float &h,float &s,float &v)
{
	m_hsvPanel->updateColorPanelHSVf(h,s,v);
	m_colorPalette->updateColor(1);
	
	emit colorPassiveChanged(0);
}
void QColorPicker::updateColorPanelHf(float &h)
{
	m_hsvPanel->updateColorPanelHf(h);
}
void QColorPicker::updateColorPanelSVf(float &s,float &v)
{
	m_hsvPanel->updateColorPanelSVf(s,v);
}
void QColorPicker::getColorPanelGeometry(QRect &rect)
{
	rect = m_hsvPanel->rect();
}
void QColorPicker::readSettings()
{
	QFile inpFile("colorSetting.txt");
	
	if(!inpFile.open(QIODevice::ReadOnly | QIODevice::Text))
	{
	}
	else
	{
		QTextStream	inp(&inpFile);
		int cc = 0;
		inp >> cc;
		m_colorPalette->setColorCounter(cc);
		qreal hhr,ssr,vvr;
		float hh,ss,vv;
		inp >> hhr >> ssr >> vvr;
		hh = hhr;
		ss = ssr;
		vv = vvr;
		m_hsvPanel->updateColorPanelHf(hh);
		m_hsvPanel->updateColorPanelSVf(ss,vv);

		for(int x=0;x<m_colorPalette->m_colorArray.size();++x)
		{
			inp >> hhr >> ssr >> vvr;
			QColor nc = QColor::fromHsvF(hhr,ssr,vvr);
			m_colorPalette->m_colorArray[x]->setColor(nc);
		}	
		inpFile.close();
	}
}
void QColorPicker::writeSettings()
{
	QFile outFile("colorSetting.txt");
	
	if(!outFile.open(QIODevice::WriteOnly | QIODevice::Text))
		return;
	
	QTextStream	out(&outFile);
	
	out << m_colorPalette->getColorCounter() << "\n";
	QColor nc = m_colorPalette->m_nowColor->getColor();
	qreal hhr,ssr,vvr;
	nc.getHsvF(&hhr,&ssr,&vvr);
	out << hhr <<" "<< ssr <<" "<< vvr << "\n";
	
	for(int x=0;x<m_colorPalette->m_colorArray.size();++x)
	{
		nc = m_colorPalette->m_colorArray[x]->getColor();
		nc.getHsvF(&hhr,&ssr,&vvr);
		out << hhr <<" "<< ssr <<" "<< vvr << "\n";
	}
	
	outFile.close();
}
QColor	QColorPicker::getQColor()
{
	return m_hsvPanel->getQColor();
}
void	QColorPicker::setQColor(QColor	&cr)
{
	m_hsvPanel->setQColor(cr);
}
/* void QColorPicker::updateColorPanelRGBf(float &r,float &g,float &b)
{
	QColor temp = QColor::fromRgbF(r,g,b);
	qreal hh,ss,vv;
	temp.getHsvF(&hh,&ss,&vv);
	float h = (float)hh;
	float s = (float)ss;
	float v = (float)vv;
	m_hsvPanel->updateColorPanelHSVf(h,s,v);
} */
