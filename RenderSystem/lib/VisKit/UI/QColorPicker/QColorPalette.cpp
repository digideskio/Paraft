#include "QColorPalette.h"
#include "QColorPicker.h"
#include <QPainter>
#include <QPen>
#include <QHBoxLayout>
#include <QSignalMapper>

QColorButton::QColorButton(QWidget *parent,QColor cr)
:QPushButton(parent),m_color(cr)
{
	//this->setFixedSize(30,30);
}
void QColorButton::paintEvent(QPaintEvent *)
{
	QPainter painter(this);

	QBrush brush1(m_color);
	painter.setBrush(brush1);
	painter.drawRect(rect()); 
}
void QColorButton::mouseReleaseEvent(QMouseEvent *event)
{
	event->ignore();
}
QColorPalette::QColorPalette(QWidget *parent)
:QWidget(parent)
{
	m_pColorPicker = (QColorPicker*)parent;
	connect(m_pColorPicker,SIGNAL(colorChanged(int)),this,SLOT(updateColor(int)));

	initLayout();
	m_colorCounter = 0;
	
}
void QColorPalette::initLayout()
{
	QRect colorPanelRect; 
	m_pColorPicker->getColorPanelGeometry(colorPanelRect);
	m_nosColorBlockLayout = 6;
	m_nosColorPoolButton = 4;
	m_colorBlockWidth = colorPanelRect.width()/m_nosColorBlockLayout;
	m_colorBlockHeight = 20;
	m_colorBlockSpacing = m_colorBlockWidth; // one block space
	
	this->setFixedHeight(m_colorBlockHeight);
	
	signalMapper = new QSignalMapper(this);

	m_nowColor = new QColorButton(this);
	m_nowColor->setGeometry(0,0,m_colorBlockWidth,m_colorBlockHeight);

	for(int a=0;a<m_nosColorPoolButton;a++)
	{
		QColorButton	*btn = new QColorButton(this);
		connect(btn,SIGNAL(clicked()),signalMapper, SLOT(map()));
		signalMapper->setMapping(btn,QString("%1").arg(a));
		m_colorArray.push_back(btn);
		btn->setGeometry(m_colorBlockWidth+m_colorBlockSpacing+a*m_colorBlockWidth,0,
						 m_colorBlockWidth,m_colorBlockHeight);
	}
	connect(signalMapper,SIGNAL(mapped(const QString &)),this,SLOT(colorArrayClicked(const QString &)));
}
void QColorPalette::updateColor(int)
{
	float hh,ss,vv;
	m_pColorPicker->getHSVf(hh,ss,vv);
	QColor temp = QColor::fromHsvF(hh,ss,vv);
	m_nowColor->setColor(temp);
	repaint();
}
void QColorPalette::colorArrayClicked(const QString &text)
{
	if(text == "0")
		m_nowColor->setColor(m_colorArray[0]->getColor());
	else if(text == "1")
		m_nowColor->setColor(m_colorArray[1]->getColor());
	else if(text == "2")
		m_nowColor->setColor(m_colorArray[2]->getColor());
	else //if(text == "3")
		m_nowColor->setColor(m_colorArray[3]->getColor());
}
void QColorPalette::mouseReleaseEvent(QMouseEvent *event)
{
	if(m_nowColor->geometry().contains(event->pos()))
	{
		m_colorArray[m_colorCounter++]->setColor(m_nowColor->getColor());
		if(m_colorCounter > 3)
			m_colorCounter = 0;
	}
	else
	{
		if(event->button() == Qt::LeftButton)
		{
			int idx = -1;
			for(int x=0;x<m_colorArray.size();x++)
			{
				if(m_colorArray[x]->geometry().contains(event->pos()))
				{
					idx = x;
					break;
				}
			}
			if(idx != -1)
			{
				m_nowColor->setColor(m_colorArray[idx]->getColor());
				QColor temp = m_nowColor->getColor();
				qreal hhr,ssr,vvr;
				float hh,ss,vv;
				temp.getHsvF(&hhr,&ssr,&vvr);
				if(hhr == -1.0f)
					hhr = 0.0f;
				hh = hhr;
				ss = ssr;
				vv = vvr;
				m_pColorPicker->updateColorPanelHf(hh);
				m_pColorPicker->updateColorPanelSVf(ss,vv);
			}
		}
		else if(event->button() == Qt::RightButton)
		{
			int idx = -1;
			for(int x=0;x<m_colorArray.size();x++)
			{
				if(m_colorArray[x]->geometry().contains(event->pos()))
				{
					idx = x;
					break;
				}
			}
			if(idx != -1)
			{
				int s = 0;
				for(s=idx;s<m_colorArray.size()-1;++s)
				{
					m_colorArray[s]->setColor(m_colorArray[s+1]->getColor());
				}
				QColor	temp = QColor::fromRgbF(1.0,1.0,1.0,1.0);
				m_colorArray[s]->setColor(temp);
				
			//	if(m_colorCounter == 0)
			//		m_colorCounter = m_colorArray.size()-1;
			//	else
			//		m_colorCounter--;
				
			}
		}
		else
		{}
		
	}
}
