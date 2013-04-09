#include "QTFColorMap.h"
#include "QTFEditor.h"
#include <QPainter>
#include <QResizeEvent>
#include <QColor>
#include <QMenu>
#include <QAction>

QTFColorMap::QTFColorMap(QWidget *parent)
:QTFAbstractPanel(parent), m_colorBackground(0,0,0)
{
	this->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
	this->setMinimumSize(400,25);
	this->setFixedHeight(25);
	m_backgroundColor = Qt::black;
	m_panelName = QString("Transfer Function");
	
	m_backgroundMenu = new QMenu(tr("Background Color"), this);
	changeBG2WhiteAct = new QAction(tr("White"), this);
	changeBG2BlackAct = new QAction(tr("Black"), this);
	changeBG2SelectedAct = new QAction(tr("Selected"), this);
	
	connect(changeBG2WhiteAct, SIGNAL(triggered()), this, SLOT(changeBG2White()));
	connect(changeBG2BlackAct, SIGNAL(triggered()), this, SLOT(changeBG2Black()));
	connect(changeBG2SelectedAct, SIGNAL(triggered()), this, SLOT(changeBG2Selected()));
	
	m_backgroundMenu->addAction(changeBG2WhiteAct);
	m_backgroundMenu->addAction(changeBG2BlackAct);
	m_backgroundMenu->addAction(changeBG2SelectedAct);
	
	initLayout();
}

void QTFColorMap::initLayout()
{
	QTFAbstractPanel::initLayout(25,3,25,3);
}
void QTFColorMap::realPaintEvent(QPaintEvent *)
{
	QPainter painter(this);
	painter.drawImage(rect(),*m_panelImage);
}

void QTFColorMap::updatePanelImage()
{
	QPainter painter(m_panelImage);
	painter.fillRect(rect(),m_backgroundColor);
	QRect	a = rect();
	int tfmapres = m_pTFEditor->getTFColorMapResolution();
	float colorblockwidth = (m_panelWidth)/(float)tfmapres;
	for(int x=0;x<tfmapres;++x)
	{
		float aa = (m_pTFEditor->getTFColorMap())[4*x+3];
		float rr = (m_pTFEditor->getTFColorMap())[4*x]*aa + (1 - aa)*m_colorBackground.redF();
		float gg = (m_pTFEditor->getTFColorMap())[4*x+1]*aa + (1 - aa)*m_colorBackground.greenF();
		float bb = (m_pTFEditor->getTFColorMap())[4*x+2]*aa + (1 - aa)*m_colorBackground.blueF();
		
		if(rr > 1.0f) rr = 1.0f;
		if(gg > 1.0f) gg = 1.0f;
		if(bb > 1.0f) bb = 1.0f;
		QColor cr;
		if(fabs(aa)<1e-10)	cr = m_colorBackground;
		else				cr.setRgbF(rr,gg,bb,1.);
		//QPen	pen2(cr);
		//QBrush	brush2(QColor::fromRgbF((m_pTFEditor->getTFColorMap())[4*x],
		//								(m_pTFEditor->getTFColorMap())[4*x+1],
		//								(m_pTFEditor->getTFColorMap())[4*x+2],
		//								(m_pTFEditor->getTFColorMap())[4*x+3]));
		//painter.setPen(pen2);
		//painter.setBrush(brush2);

		QBrush	brush2(cr);
		QRectF rcf((float)m_panelLMargin+(float)x*colorblockwidth, 
				   (float)m_panelUMargin + 1, 
				   (float)colorblockwidth, 
				   (float)m_panelHeight-2);
		painter.fillRect(rcf,brush2);
	}
	QPen pen1(Qt::black, 1);
	painter.setPen(pen1);
	painter.drawRect(m_panelLMargin,m_panelUMargin+1,m_panelWidth,m_panelHeight-2);
	
}
void QTFColorMap::mousePressEvent(QMouseEvent* event)
{
	if(event->button() == Qt::RightButton)
	{
		m_backgroundMenu->exec(QCursor::pos());
	}
}

void QTFColorMap::updateTFColorMap()
{
	updatePanelImage();
	repaint();
}

void QTFColorMap::changeBG2White() {
	changeBGColor(QColor(255,255,255));
}
void QTFColorMap::changeBG2Black() {
	changeBGColor(QColor(0,0,0));
}
void QTFColorMap::changeBG2Selected() {
	changeBGColor(m_pTFEditor->getColorPicker()->getQColor());
}

void QTFColorMap::changeBGColor(QColor c) {
	m_colorBackground = c;
	updatePanelImage();
	repaint();
	emit(bgColorChanged(c));
}

