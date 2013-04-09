#include "QTFEditor.h"
#include "QTFPanel.h"
#include <QPainter>
#include <QPoint>
#include <QMouseEvent>
#include <QPainterPath>
#include <QPolygon>
#include <QLinearGradient>
#include <QSlider>
#include <QFileDialog>
#include <QPushButton>
#include <QMenu>
#include <QTextStream>
#include <QGridLayout>
#include <QLineEdit>
#include <QString>
#include <iostream>
#include "../NLTFEditor/nltfeditor.h"

//#include "gsl/gsl_rng.h"
//#include "gsl/gsl_randist.h"

using namespace std;

TFMappingEditor::TFMappingEditor(QWidget *parent):QWidget(parent){
	setWindowFlags(Qt::Window);
	setWindowTitle(tr("Mapping editor"));

        x1 = 10;
	y1 = 0.1;
        x2 = 50;
	y2 = 0.5;

	QGridLayout *layout = new QGridLayout;
	layout->setSpacing(2);
	layout->setAlignment(Qt::AlignTop);

	layout->addWidget(new QLabel(tr(" ")),0,0);
	layout->addWidget(new QLabel(tr("X")),0,1);
	layout->addWidget(new QLabel(tr("Y")),0,2);

	layout->addWidget(new QLabel(tr("x0=")),1,0);
	layout->addWidget(new QLabel(tr("0")),1,1);
	layout->addWidget(new QLabel(tr("0.0")),1,2);

	layout->addWidget(new QLabel(tr("x1=")),2,0);
	x1edit = new QLineEdit(QString("%1").arg(x1));
	y1edit = new QLineEdit(QString("%1").arg(y1));

	layout->addWidget(x1edit,2,1);
	layout->addWidget(y1edit,2,2);

	layout->addWidget(new QLabel(tr("x2=")),3,0);
	x2edit = new QLineEdit(QString("%1").arg(x2));
	y2edit = new QLineEdit(QString("%1").arg(y2));

	layout->addWidget(x2edit,3,1);
	layout->addWidget(y2edit,3,2);

	layout->addWidget(new QLabel(tr("x3=")),4,0);
        layout->addWidget(new QLabel(tr("100")),4,1);
	layout->addWidget(new QLabel(tr("1.0")),4,2);

	QPushButton *updateBtn = new QPushButton(tr("Update"));
	connect(updateBtn,SIGNAL(pressed()),this,SLOT(updateButtonPressed()));
	layout->addWidget(updateBtn,5,1);

	this->setLayout(layout);
	this->setGeometry(50,50,150,300);
}
TFMappingEditor::~TFMappingEditor(){

}
void TFMappingEditor::updateButtonPressed(){
	x1 = x1edit->text().toInt();
	x2 = x2edit->text().toInt();
	y1 = y1edit->text().toFloat();
	y2 = y2edit->text().toFloat();
	emit mappingChanged(x1,y1,x2,y2);
}

QTFPanel::QTFPanel(int transferWidth, QWidget *parent)
:QTFAbstractPanel(parent)
{
	this->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
	this->setMinimumSize(400,190);
	this->setMouseTracking(true);
	m_pTFEditor = (QTFEditor*)parent;
	m_isLeftClick = false;
	m_isRightClick= false;
	m_ifOnDrawing = false;
	m_ifOnDrawColorTick = false;
	m_clickedTicks = -1;
	m_vertStringNeedUpdate = false;
	m_vertTranSliderRange = 10;
	m_ifCTRLPressed = false;
	m_ifSHIFTPressed = false;
	turnOnNonlinearXYMapping = false;
        m_mappingRes = 100;

        m_xyMapping.resize(m_mappingRes+1);
        m_xyLinearMapping.resize(m_mappingRes+1);
        for(int a=0;a<(m_mappingRes+1);a++){
                m_xyMapping[a] = log(1.f + (a)*15.f/m_mappingRes)/log(15.f);
                m_xyLinearMapping[a] = (a)*(1.f/(float)m_mappingRes);
	}

	m_buttonSize = QSize(20,20);

	m_backgroundMesh = 1; // dot liine
	m_panelImage = new QImage(rect().width(),rect().height(),QImage::Format_RGB32);
	m_tfImage = new QImage(rect().width(), 1, QImage::Format_RGB32);
	m_tfResolution = 1024;
	m_tfDrawArray = new float[(int)m_tfResolution];
	for(int x=0;x<m_tfResolution;++x)
		m_tfDrawArray[x] = 0.0f;

	m_tfColorTick << TFColorTick(0.0f, QColor::fromHsvF(0.611,1.0,1.0))
	              << TFColorTick(m_tfResolution * 1 / 6, QColor::fromHsvF(0.500,1.0,1.0))
	              << TFColorTick(m_tfResolution * 2 / 6, QColor::fromHsvF(0.353,1.0,1.0))
	              << TFColorTick(m_tfResolution * 3 / 6, QColor::fromHsvF(0.176,1.0,1.0))
	              << TFColorTick(m_tfResolution * 4 / 6, QColor::fromHsvF(0.086,1.0,1.0))
	              << TFColorTick(m_tfResolution * 5 / 6, QColor::fromHsvF(0.000,1.0,1.0))
	              << TFColorTick(m_tfResolution * 6 / 6, QColor::fromHsvF(0.784,1.0,1.0));
	initLayout();

	m_settingMenu = new QMenu(this);

	QMenu	*backgroundMeshMenu = new QMenu(tr("Background Grid"),this);
	changeBM2NoneAct = new QAction(tr("None"),this);
	changeBM2NoneAct->setCheckable(true);
	connect(changeBM2NoneAct, SIGNAL(triggered()), this, SLOT(changeBM2None()));
	changeBM2DotLineAct = new QAction(tr("Dot Line"),this);
	changeBM2DotLineAct->setCheckable(true);
	connect(changeBM2DotLineAct, SIGNAL(triggered()), this, SLOT(changeBM2DotLine()));
	backgroundMeshMenu->addAction(changeBM2NoneAct);
	backgroundMeshMenu->addAction(changeBM2DotLineAct);

	QMenu *transferOptionsMenu = new QMenu(tr("Transfer Options"), this);
	changeCombine2OrAct = new QAction(tr("Use Greater"),this);
	changeCombine2AndAct = new QAction(tr("Additive"), this);
	changeCombine2OrAct->setCheckable(true);
	changeCombine2OrAct->setChecked(true);
	changeCombine2AndAct->setCheckable(true);
	connect(changeCombine2OrAct, SIGNAL(triggered()), this, SLOT(changeCombine2Or()));
	connect(changeCombine2AndAct, SIGNAL(triggered()), this, SLOT(changeCombine2And()));
	transferOptionsMenu->addAction(changeCombine2AndAct);
	transferOptionsMenu->addAction(changeCombine2OrAct);

	openFileAct = new QAction(tr("Load Settings"),this);
	saveFileAct = new QAction(tr("Save Settings"),this);
	connect(openFileAct, SIGNAL(triggered()), this, SLOT(openFile()));
	connect(saveFileAct, SIGNAL(triggered()), this, SLOT(saveFile()));

	toggleInstantAct = new QAction(tr("Toggle Instant Updates"),this);
	toggleInstantAct->setCheckable(true);
	connect(toggleInstantAct, SIGNAL(triggered()), this, SLOT(toggleInstant()));

	toggleDrawLabels = new QAction(tr("Draw Labels"), this);
	toggleDrawLabels->setCheckable(true);
	toggleDrawLabels->setChecked(true);
	connect(toggleDrawLabels, SIGNAL(triggered(bool)), this, SLOT(togglePositionLabel(bool)));

	toggleDrawXYMapping = new QAction(tr("Turn on Mapping"), this);
	toggleDrawXYMapping->setCheckable(true);
	toggleDrawXYMapping->setChecked(turnOnNonlinearXYMapping);
	connect(toggleDrawXYMapping, SIGNAL(triggered(bool)), this, SLOT(toggleXYMapping()));

	m_settingMenu->addMenu(backgroundMeshMenu);
	m_settingMenu->addMenu(transferOptionsMenu);
	m_settingMenu->addAction(openFileAct);
	m_settingMenu->addAction(saveFileAct);
	m_settingMenu->addAction(toggleInstantAct);
	m_settingMenu->addAction(toggleDrawLabels);
	m_settingMenu->addAction(toggleDrawXYMapping);

	m_gaussianObjectArray.resize(0);
	m_clickedObjectControlBox = -1;
	m_ifOnMovingObjectControlBox = false;
	m_objectControlBoxSide = -1;

	m_tfColorMapResoultion = transferWidth;
	m_tfColorMap = new float[m_tfColorMapResoultion*4]; // r,g,b,a - 4 channels
	memset(m_tfColorMap, 0, m_tfColorMapResoultion*16);

	changeBM2DotLine();
	setCollapsable(false);
	m_zeroRangesArray.append(ZeroRange(0,1));
	m_instant = false;
	rangemin = 0;
	rangemax = 1;
	setMouseTracking(true);
	drawLabels = true;
	lastyval = -1;

	m_xyMappingMap = new float[m_tfColorMapResoultion]; // r - 1 channel
	memset(m_xyMappingMap, 0, m_tfColorMapResoultion*sizeof(float));
	m_xyLinearMappingMap = new float[m_tfColorMapResoultion]; // r - 1 channel
	memset(m_xyLinearMappingMap, 0, m_tfColorMapResoultion*sizeof(float));

	m_mappingEditor = new TFMappingEditor;
	//m_nltfEditor = new NLTFEditor;
	connect(m_mappingEditor,SIGNAL(mappingChanged(int,float,int,float)),this,SLOT(generateMapping(int,float,int,float)));
	//m_mappingEditor->hide();

	generateMapping(m_mappingEditor->x1,m_mappingEditor->y1,m_mappingEditor->x2,m_mappingEditor->y2);
	updateXYLinearMappingMap();
	//updateXYMappingMap();
}
QTFPanel::~QTFPanel()
{
	delete [] m_tfDrawArray;
	delete [] m_tfColorMap;

	delete m_settingMenu;
}

void QTFPanel::setRange(float rmin, float rmax) {
	rangemin = rmin;
	rangemax = rmax;
}
void QTFPanel::initLayout()
{
	QTFAbstractPanel::initLayout(25,1,25,35);

	m_colorTickHeight	= 15;

	if(m_colorTickHeight > m_panelDMargin)
		m_panelDMargin = m_panelDMargin;

	m_panelArea		= QRect(m_panelLMargin,m_panelUMargin,m_panelWidth,m_panelHeight);
	m_panelAreaClick	= QRect(m_panelLMargin,m_panelUMargin,m_panelWidth,m_panelHeight+12);
	m_colorTickArea	= QRect(m_panelLMargin,m_panelHeight+11,m_panelWidth,m_colorTickHeight);

	m_tfRes2PtFactor	= (float)m_panelWidth/(m_tfResolution - 1);
	m_tfPt2ResFactor	= 1.0f/m_tfRes2PtFactor;

	m_vertTranSlider	= new QSlider(Qt::Vertical,this);
	m_vertTranSlider->setGeometry(0,0,m_panelLMargin,m_panelUMargin+(m_panelHeight/2));
//	m_vertTranSlider->setTickPosition(QSlider::TicksRight);
	m_vertTranSlider->setRange(1,m_vertTranSliderRange-1);
//	m_vertTranSlider->setTickInterval(1);
	m_vertTranSlider->setValue((int)m_vertTranSliderRange/2);
	m_vertTranSlider->setTracking(true);

	connect(m_vertTranSlider,SIGNAL(valueChanged(int)),this,SLOT(vertTranSliderChange(int)));

	m_functionButton = new QPushButton(tr("F"),this);
	m_functionButton->setGeometry(2,m_panelUMargin+(m_panelHeight/2)+2,m_buttonSize.width(),m_buttonSize.height());
	connect(m_functionButton,SIGNAL(released()),this,SLOT(functionButtonPressed()));

	m_settingButton = new QPushButton(tr("S"),this);
	m_settingButton->setGeometry(2,m_panelUMargin+(m_panelHeight/2)+2+m_buttonSize.height()+1,m_buttonSize.width(),m_buttonSize.height());
	connect(m_settingButton,SIGNAL(released()),this,SLOT(settingButtonPressed()));

	m_mappingButton = new QPushButton(tr("M"),this);
	m_mappingButton->setGeometry(2,m_panelUMargin+(m_panelHeight/2)+4+2*m_buttonSize.height()+2,m_buttonSize.width(),m_buttonSize.height());
	connect(m_mappingButton,SIGNAL(released()),this,SLOT(mappingButtonPressed()));
	m_mappingButton->hide();
}
void QTFPanel::realPaintEvent(QPaintEvent *)
{
	QPainter painter(this);
	painter.drawImage(rect(),*m_panelImage);

	// draw drawing lines
	QPen	pen1(Qt::black,1);
	QPen	pen2(Qt::black);
	QPen	pen3(Qt::white,1);
	QPen	pen4(Qt::green,1);
	painter.setPen(pen1);
	QBrush	brush1(QColor(255,255,255,100));
	painter.setBrush(brush1);
	QPainterPath path;
	float xposs = m_panelLMargin;
	float yposs = m_panelUMargin + m_panelHeight - 1;
	path.moveTo(xposs,yposs);
	// draw with transform
	int vertSliderValue = m_vertTranSlider->value();
	float halfHeight = 0.5 * m_panelHeight;
	float lowerPart = (float)vertSliderValue/m_vertTranSliderRange;
	float upperPart = 1.0f - lowerPart;//(float)(10-vertSliderValue)/10.0f;

	// draw background grid
	if(m_backgroundMesh == 1) // mesh
	{
		QPen	penG(Qt::gray,1);
		penG.setStyle(Qt::DashLine);
		QPen	penW(Qt::white,1);
		penW.setStyle(Qt::DashLine);
		painter.setPen(penG);

		QPointF	px1,px2,py1,py2;
		float gridYheight;
		for(int x=1;x<=9;x=x+1)
		{
			if(x == 5)
				painter.setPen(penW);
			else
				painter.setPen(penG);

			if(x <= vertSliderValue)
			{
				gridYheight = m_panelUMargin+m_panelHeight-((float)x/vertSliderValue) * halfHeight;
			}
			else
			{
				int upp = m_vertTranSliderRange - vertSliderValue;
				gridYheight = m_panelUMargin+m_panelHeight-halfHeight-((float)(x-vertSliderValue)/upp) * halfHeight;
			}
			//py2 = QPointF(rect().width()-m_panelRMargin,m_panelUMargin+((float)m_panelHeight/10.0f) * x);
			py1 = QPointF(m_panelLMargin+1,gridYheight);
			py2 = QPointF(rect().width()-m_panelRMargin,gridYheight);
			painter.drawLine(py1,py2);
		}
	}

	if(turnOnNonlinearXYMapping){
		QPen	penG(Qt::darkRed,3);
		penG.setStyle(Qt::SolidLine);
		painter.setPen(penG);

		QPointF startP, endP, midP;

		float gridYheight = m_panelUMargin + m_panelHeight;
                float xInterval = (rect().width() - m_panelRMargin - m_panelLMargin)/(float)m_xyMapping.size();

	//	startP = QPointF(m_panelLMargin+1,gridYheight);
	//	midP = QPointF(m_panelLMargin+1+xInterval,gridYheight-m_xyMapping[0]*m_panelHeight);
	//	painter.drawLine(startP,midP);
		//startP = midP;
		for(int x=1;x<m_xyMapping.size();++x){
			startP = QPointF(m_panelLMargin+1+(x-1)*xInterval,gridYheight-m_xyMapping[x-1]*m_panelHeight);
			midP = QPointF(m_panelLMargin+1+(x)*xInterval,gridYheight-m_xyMapping[x]*m_panelHeight);
			painter.drawLine(startP,midP);
		}
		//endP = QPointF(rect().width()-m_panelRMargin,m_panelUMargin);
	}

	// log transform?

	//draw total line
	if(m_combineMode == 1) {
		QBrush	brush2(QColor(251,126,233,128));
		QPen pen;
		//pen.setStyle(Qt::DashLine);
		pen.setWidth(2);
		painter.setBrush(brush2);
		painter.setPen(pen);
		double alpha;
		QPainterPath path;
		path.moveTo(m_panelLMargin, m_panelUMargin + m_panelHeight - 1);
		for(int c=0;c<m_tfResolution;++c)
		{
			xposs = m_panelLMargin + c*m_tfRes2PtFactor;
			alpha = alphaValue(c);
			float ratio,rHeight;
			if(alpha <= lowerPart)
			{
				ratio = alpha/lowerPart;
				if(ratio > 1.0f) ratio = 1.0f;
				rHeight = ratio * halfHeight;
			}
			else
			{
				ratio = (alpha - lowerPart) / upperPart;
				if(ratio > 1.0f) ratio = 1.0f;
				rHeight = halfHeight * (1.0f+ratio); //ratio * 0.5 * m_panelHeight + 0.5 * m_panelHeight
			}
			yposs = m_panelUMargin + m_panelHeight - rHeight - 1;
			path.lineTo(xposs,yposs);
		}
		path.lineTo(rect().width()-m_panelRMargin,m_panelUMargin + m_panelHeight);
		painter.drawPath(path);
		painter.setBrush(brush1);
	}
	painter.setPen(pen1);

	brush1.setColor(QColor(255,255,255,100));
	for(int c=0;c<m_tfResolution;++c)
	{
		xposs = m_panelLMargin + c*m_tfRes2PtFactor;
		float ratio,rHeight;
		if(m_tfDrawArray[c] <= lowerPart)
		{
			ratio = m_tfDrawArray[c]/lowerPart;
			if(ratio > 1.0f) ratio = 1.0f;
			rHeight = ratio * halfHeight;
		}
		else
		{
			ratio = (m_tfDrawArray[c] - lowerPart) / upperPart;
			if(ratio > 1.0f) ratio = 1.0f;
			rHeight = halfHeight * (1.0f+ratio); //ratio * 0.5 * m_panelHeight + 0.5 * m_panelHeight
		}
		yposs = m_panelUMargin + m_panelHeight - rHeight - 1;
		path.lineTo(xposs,yposs);
	}
	path.lineTo(rect().width()-m_panelRMargin,m_panelUMargin + m_panelHeight);
	painter.drawPath(path);

	// draw objects
	brush1.setColor(QColor(255,255,255,100));
	if(m_gaussianObjectArray.size() != 0)
	{
		for(int x=0;x<m_gaussianObjectArray.size();++x)
		{
			QPainterPath pathObj;
			xposs = m_panelLMargin;
			yposs = m_panelUMargin + m_panelHeight - 1;
			pathObj.moveTo(xposs,yposs);
			for(int c=0;c<m_tfResolution;++c)
			{
				xposs = m_panelLMargin + c*m_tfRes2PtFactor;
				float ratio,rHeight;
				if(m_gaussianObjectArray[x].m_distribution[c] <= lowerPart)
				{
					ratio = m_gaussianObjectArray[x].m_distribution[c]/lowerPart;
					if(ratio > 1.0f) ratio = 1.0f;
					rHeight = ratio * halfHeight;
				}
				else
				{
					ratio = (m_gaussianObjectArray[x].m_distribution[c] - lowerPart) / upperPart;
					if(ratio > 1.0f) ratio = 1.0f;
					rHeight = halfHeight * (1.0f+ratio); //ratio * 0.5 * m_panelHeight + 0.5 * m_panelHeight
				}
				yposs = m_panelUMargin + m_panelHeight - rHeight - 1;
				pathObj.lineTo(xposs,yposs);
			}
			pathObj.lineTo(rect().width()-m_panelRMargin,m_panelUMargin + m_panelHeight-1);
			painter.drawPath(pathObj);
		}
	}


	// draw color ticks

	float xposs1,yposs1,xposs2,yposs2,xposs3,yposs3;
	for(int d=0;d<m_tfColorTick.size();++d)
	{
		xposs1 = m_panelLMargin + m_tfColorTick[d].m_resX * m_tfRes2PtFactor;
		yposs1 = m_panelUMargin + m_panelHeight + 10;
		xposs2 = m_panelLMargin + m_tfColorTick[d].m_resX * m_tfRes2PtFactor - 10;
		yposs2 = m_panelUMargin + m_panelHeight + 10 + m_colorTickHeight;
		xposs3 = m_panelLMargin + m_tfColorTick[d].m_resX * m_tfRes2PtFactor + 10;
		yposs3 = m_panelUMargin + m_panelHeight + 10 + m_colorTickHeight;
		QPolygon	tri;
		tri << QPoint(xposs1,yposs1) << QPoint(xposs2,yposs2) << QPoint(xposs3,yposs3);
		QBrush		brush2(m_tfColorTick[d].m_color);
		painter.setPen(pen2);
		painter.setBrush(brush2);
		painter.drawPolygon(tri);
	}
	// draw selected tick
	if(m_clickedTicks >= 0 && m_clickedTicks<m_tfColorTick.size())
	{
		painter.setPen(pen3);
		xposs1 = m_panelLMargin + m_tfColorTick[m_clickedTicks].m_resX * m_tfRes2PtFactor;
		yposs1 = m_panelUMargin + m_panelHeight + 10;
		xposs2 = m_panelLMargin + m_tfColorTick[m_clickedTicks].m_resX * m_tfRes2PtFactor - 10;
		yposs2 = m_panelUMargin + m_panelHeight + 10 + m_colorTickHeight;
		xposs3 = m_panelLMargin + m_tfColorTick[m_clickedTicks].m_resX * m_tfRes2PtFactor + 10;
		yposs3 = m_panelUMargin + m_panelHeight + 10 + m_colorTickHeight;
		QPolygon	tri;
		tri << QPoint(xposs1,yposs1) << QPoint(xposs2,yposs2) << QPoint(xposs3,yposs3);
		QBrush		brusht(m_tfColorTick[m_clickedTicks].m_color);
		painter.setBrush(brusht);
		painter.drawPolygon(tri);
	}
	// vert transform string
	if(m_vertStringNeedUpdate)
	{
		painter.setPen(pen2);
		QPoint pts = mapFromGlobal(QCursor::pos());
		QPoint	pt = QPoint(m_panelLMargin,pts.y());//m_panelLMargin,yyy);//m_panelUMargin+0.5*m_panelHeight);// QCursor::pos();
		painter.drawText(pt, m_vertTranPercetangeString);
		m_vertStringNeedUpdate = false;
	}
	// draw object control node
	if(m_gaussianObjectArray.size() != 0)
	{
		for(int x=0;x<m_gaussianObjectArray.size();++x)
		{
//			if(x == m_clickedObjectControlBox)
//				painter.setPen(pen3);
//			else
//				painter.setPen(pen1);

//			xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 7;
//			yposs = m_panelUMargin + m_panelHeight - 14;
//
//			painter.drawRect(xposs,yposs,14,14);
//
//			painter.setBrush(Qt::red);
//			xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth;
//			yposs = m_panelUMargin + m_panelHeight - 14;
//
//			painter.drawRect(xposs,yposs,7,14);
			painter.setPen(pen1);
			if(x == m_clickedObjectControlBox) {
				// horizontal move
				xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 10;
				yposs = m_panelUMargin + m_panelHeight - 25;
				painter.setBrush(QColor(255,255,255,125));
				painter.drawRect(xposs,yposs,20,15);
				painter.setBrush(Qt::black);
				painter.drawLine(xposs+1, yposs+8, xposs+8, yposs+8);
				painter.drawLine(xposs+19, yposs+8, xposs+12, yposs+8);
				painter.drawLine(xposs+1, yposs+8, xposs+4, yposs+5);
				painter.drawLine(xposs+1, yposs+8, xposs+4, yposs+11);
				painter.drawLine(xposs+19, yposs+8, xposs+16, yposs+5);
				painter.drawLine(xposs+19, yposs+8, xposs+16, yposs+11);

				// horizontal scale
				xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 10;
				yposs = m_panelUMargin + m_panelHeight - 40;
				painter.setBrush(QColor(255,255,255,125));
				painter.drawRect(xposs,yposs,20,15);
				painter.setBrush(Qt::black);
				painter.drawLine(xposs+1, yposs+8, xposs+19, yposs+8);
				painter.drawLine(xposs+1, yposs+8, xposs+4, yposs+5);
				painter.drawLine(xposs+1, yposs+8, xposs+4, yposs+11);
				painter.drawLine(xposs+19, yposs+8, xposs+16, yposs+5);
				painter.drawLine(xposs+19, yposs+8, xposs+16, yposs+11);

				// vertical scale
				xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 10;
				yposs = m_panelUMargin + m_panelHeight - 55;
				painter.setBrush(QColor(255,255,255,125));
				painter.drawRect(xposs,yposs,20,15);
				painter.setBrush(Qt::black);
				painter.drawLine(xposs+10, yposs+14, xposs+10, yposs+1);
				painter.drawLine(xposs+10, yposs+1, xposs+7, yposs+4);
				painter.drawLine(xposs+10, yposs+1, xposs+13, yposs+4);
			}
			xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 6;
			yposs = m_panelUMargin + m_panelHeight - 10;
			painter.setBrush(QColor(255,255,255,125));
			painter.drawRect(xposs,yposs,12,9);
			painter.drawLine(xposs+6, yposs+1, xposs+6, yposs+8);
			painter.drawLine(xposs+2, yposs+2, xposs+10, yposs+8);
			painter.drawLine(xposs+2, yposs+8, xposs+10, yposs+2);
		}
	}
	//draw range labels
	if(drawLabels) {
		QRect bounding;
		painter.setPen(pen1);
		painter.setBrush(brush1);
		painter.drawText(QPoint(5, height() - m_panelDMargin), QString::number(rangemin));
		painter.drawText(QPoint(width() - m_panelRMargin + 5, height() - m_panelDMargin), QString::number(rangemax));
		QPoint pt = mapFromGlobal(QCursor::pos());
		if(!(pt.x() < m_panelLMargin || pt.x() > (width() - m_panelRMargin) || pt.y() < m_panelUMargin || pt.y() > (height() - m_panelDMargin))) {
			float value = ((pt.x() - m_panelLMargin)/(float)m_panelWidth )*(rangemax - rangemin) + rangemin;
			painter.drawText(QRect(pt.x(), pt.y() - 10, 100, 100), 0, QString::number(value), &bounding);
			painter.drawRect(bounding);
			painter.drawText(QRect(pt.x(), pt.y() - 10, 100, 100), 0, QString::number(value));
		}
	}

}
void QTFPanel::resizeEvent(QResizeEvent *event)
{
	m_panelWidth	= rect().width() - m_panelLMargin - m_panelRMargin;
	m_panelHeight	= rect().height() - m_panelUMargin - m_panelDMargin;
	m_panelArea		= QRect(m_panelLMargin,m_panelUMargin,m_panelWidth,m_panelHeight);
	m_panelAreaClick	= QRect(m_panelLMargin,m_panelUMargin,m_panelWidth,m_panelHeight+12);
	m_colorTickArea	= QRect(m_panelLMargin,m_panelUMargin+m_panelHeight+10,m_panelWidth,m_colorTickHeight);

	m_tfRes2PtFactor	= (float)m_panelWidth/(m_tfResolution - 1);
	m_tfPt2ResFactor	= 1.0f/m_tfRes2PtFactor;

	m_vertTranSlider->setGeometry(0,0,m_panelLMargin,(m_panelUMargin+m_panelHeight)/2);
	m_functionButton->setGeometry(2,m_panelUMargin+(m_panelHeight/2)+2,m_buttonSize.width(),m_buttonSize.height());
	m_settingButton->setGeometry(2,m_panelUMargin+(m_panelHeight/2)+2+m_buttonSize.height()+1,m_buttonSize.width(),m_buttonSize.height());

	if(m_tfImage->width() != event->size().width()) {
		delete m_tfImage;
		m_tfImage = new QImage(event->size().width(), 1, QImage::Format_RGB32);
	}

	QTFAbstractPanel::resizeEvent(event);
}
void QTFPanel::updatePanelImage()
{
	QPainter painter(m_panelImage);
	m_panelImage->fill(qRgb(255, 255, 255));

	QPen pen1(Qt::black, 1);
	painter.setPen(pen1);
	QBrush	brush1(QColor(220,220,220,255));
	painter.setBrush(brush1);

	painter.drawRect(m_colorTickArea);//m_panelLMargin,m_panelUMargin,m_panelWidth,m_panelHeight);

	QLinearGradient linearGrad(m_panelLMargin,m_panelUMargin+0.5*m_panelHeight,
							   rect().width()-m_panelRMargin,m_panelUMargin+0.5*m_panelHeight);

	for(int x=0;x<m_tfColorTick.size();++x)
	{
		float res = (m_tfColorTick[x].m_resX)/m_tfResolution;
		linearGrad.setColorAt(res,m_tfColorTick[x].m_color);
	}
	painter.fillRect(m_panelArea,linearGrad);
}
void QTFPanel::mousePressEvent(QMouseEvent *event)
{
	if(event->button() == Qt::LeftButton && event->button() == Qt::RightButton)
	{
		m_isLeftClick = false;
		m_isRightClick= false;
		m_ifOnDrawing = false;
	}
	if(event->button() == Qt::LeftButton && ((m_clickedObjectControlBox = clickOnObjctControlBox((QPoint&)event->pos(),m_objectControlBoxSide)) != -1))
	{
		m_isLeftClick = true;
		m_ifOnMovingObjectControlBox = true;
		m_nowPoint = event->pos();
		m_lastPoint = m_nowPoint;
		updateTFColorMap();
		repaint();
	}
	else if(event->button() == Qt::RightButton && ((m_clickedObjectControlBox = clickOnObjctControlBox((QPoint&)event->pos())) != -1))
	{
		m_isRightClick = true;
		m_gaussianObjectArray.remove(m_clickedObjectControlBox);
		m_clickedObjectControlBox = -1;
		m_nowPoint = event->pos();
		m_lastPoint = m_nowPoint;
		updateTFColorMap();
		repaint();
	}
	else if(event->button() == Qt::LeftButton && m_panelAreaClick.contains(event->pos()))
	{
		m_clickedTicks = -1;
		m_isLeftClick = true;
		m_nowPoint = event->pos();

		m_nowResIndex = (m_nowPoint.rx() - m_panelLMargin) * m_tfPt2ResFactor;

		int vertSliderValue = m_vertTranSlider->value();
		float lowerPart = (float)vertSliderValue/m_vertTranSliderRange;
		float upperPart = 1.0f - lowerPart;


		float yheight = (float)(m_panelHeight - (m_nowPoint.ry() - m_panelUMargin));
		float halfHeight = 0.5 * m_panelHeight;
		if(yheight <= halfHeight)
		{yval = (yheight/halfHeight)*lowerPart;}
		else
		{yval = ((yheight - halfHeight)/halfHeight) * upperPart + lowerPart;}

		if(yval > 1.0f) yval = 1.0f;
		else if(yval < 0.0f) yval = 0.0f;

	//	float yval = (float)(m_panelHeight - (m_nowPoint.ry() - m_panelUMargin))/m_panelHeight;
		lastyval = yval;
		if(!(event->modifiers() & Qt::ControlModifier) || fabs(m_tfDrawArray[m_nowResIndex]) > 0.000001)
			m_tfDrawArray[m_nowResIndex] = yval;

		m_lastPoint = m_nowPoint;
		m_lastResIndex = m_nowResIndex;
		m_ifOnDrawing = true;
		updateTFColorMap();
		repaint();
	}
	else if(event->button() == Qt::RightButton && m_panelAreaClick.contains(event->pos()))
	{
		m_isRightClick = true;
		m_nowPoint = event->pos();
		yval = lastyval = 0;

		m_nowResIndex = (m_nowPoint.rx() - m_panelLMargin) * m_tfPt2ResFactor;
		m_tfDrawArray[m_nowResIndex] = 0.0f;

		m_lastPoint = m_nowPoint;
		m_lastResIndex = m_nowResIndex;
		m_ifOnDrawing = true;
		updateTFColorMap();
		repaint();
	}
	else if(event->button() == Qt::LeftButton &&
			(m_colorTickArea.contains(event->pos()) ||
			(m_clickedTicks = clickOnTicks((QPoint&)event->pos())) != -1))
	{
		m_isLeftClick = true;
		QPoint tp = event->pos();

		if((m_clickedTicks = clickOnTicks(tp)) == -1)
		{
			int tickResIndex = (tp.rx() - m_panelLMargin) * m_tfPt2ResFactor;
			if(tickResIndex < 0) tickResIndex = 0;
			else if(tickResIndex > m_tfResolution) tickResIndex = m_tfResolution;

			QColor tc = QColor::fromRgb(m_panelImage->pixel(tp.x(),m_panelUMargin+0.5*m_panelHeight));
			TFColorTick	temp(tickResIndex,tc);
			m_tfColorTick.append(temp);
			m_clickedTicks = m_tfColorTick.size()-1;
		}
		else  // click an existed tick, set color to colorpicker
		{
			m_pTFEditor->getColorPicker()->setQColor(m_tfColorTick[m_clickedTicks].m_color);
		}
		m_ifOnDrawColorTick = true;
		updatePanelImage();
		updateTFColorMap();
		repaint();
	}
	else if(event->button() == Qt::RightButton &&
			(m_colorTickArea.contains(event->pos()) ||
			(m_clickedTicks = clickOnTicks((QPoint&)event->pos())) != -1))
	{
		m_isRightClick = true;
		QPoint tp = event->pos();

		if(m_tfColorTick.size() > 2)
		{
			m_clickedTicks = clickOnTicks(tp);
			if((m_clickedTicks >= 0) && (m_clickedTicks<(m_tfColorTick.size()))) // exclude empty space (-1)
			{
				m_tfColorTick.remove(m_clickedTicks);
				m_clickedTicks = -1;
				updatePanelImage();
				updateTFColorMap();
				repaint();
			}
		}
	}
}
void QTFPanel::updateXYLinearMappingMap(){
	int base = 0;
	int count = 0;
	unsigned int nosColorPerInterval = m_tfColorMapResoultion / (m_xyMapping.size()-1);

	// 0~m_xyMapping[0]
//	float minn = 0;
//	float maxx = m_xyLinearMapping[0];
//	float xstep = (maxx - minn)/nosColorPerInterval;

//	for(count = 0;count < nosColorPerInterval;count++){
//		m_xyLinearMappingMap[count] = minn + xstep * count;
//		base++;
//	}

	float minn,maxx,xstep;
	int x;
	for(x=1;x<m_xyLinearMapping.size()-1;++x){
		minn = m_xyLinearMapping[x-1];
		maxx = m_xyLinearMapping[x];
		xstep = (maxx - minn)/nosColorPerInterval;

		float *ptr = m_xyLinearMappingMap + base;
		for(int count = 0;count < nosColorPerInterval;count++){
			ptr[count] = minn + xstep * count;
			base++;
		}
	}

	unsigned int ncp = (m_tfColorMapResoultion - base);
	minn = m_xyLinearMapping[x-1];
	maxx = m_xyLinearMapping[x];
	xstep = (maxx - minn)/ncp;
	count = 0;

	for(int q = (m_tfColorMapResoultion-1);q >= base;q--){
		m_xyLinearMappingMap[q] = 1.0 - (xstep * count);
		count++;
	}
}
void QTFPanel::updateXYMappingMap(){
	unsigned int nosColorPerInterval = m_tfColorMapResoultion / (m_xyMapping.size()-1);

	// 0~m_xyMapping[0]
//	float minn = 0;
//	float maxx = m_xyMapping[0];
//	float xstep = (maxx - minn)/nosColorPerInterval;

//	for(count = 0;count < nosColorPerInterval;count++){
//		m_xyMappingMap[count] = minn + xstep * count;
//		base++;
//	}

	float minn,maxx,xstep;
	int base = 0, count = 0, x = 1;
	for(x=1;x<m_xyMapping.size()-1;++x){
		minn = m_xyMapping[x-1];
		maxx = m_xyMapping[x];
		xstep = (maxx - minn)/nosColorPerInterval;

		float *ptr = m_xyMappingMap + base;
		for(int count = 0;count < nosColorPerInterval;count++){
			ptr[count] = minn + xstep * count;
			base++;
		}
	}

	unsigned int ncp = (m_tfColorMapResoultion - base);
	minn = m_xyMapping[x-1];
	maxx = m_xyMapping[x];
	xstep = (maxx - minn)/ncp;
	count = 0;

	for(int q = (m_tfColorMapResoultion-1);q >= base;q--){
		m_xyMappingMap[q] = 1.0 - (xstep * count);
		count++;
	}
}
void QTFPanel::generateMapping(int x1,float y1,int x2, float y2){
	// piecewise linear 0<x1<x2<10
	if(x1 > x2) swap(x1,x2);
	if(y1 > y2) swap(y1,y2);
	// x1 should be between 0~x2
	// x2 should be between x1~9

        if(x1 <= 0 || x2 >=m_mappingRes || y1 <= 0.0f || y2 >= 1.0f){
		qDebug("Range wrong!");
		return;
	}

	float base = 0.0f;
	int count = 1;
	int interval1 = x1;
	float interval1Step = y1 / x1;

//	qDebug("i=%f",interval1Step);
	m_xyMapping[0] = 0.0;
	for(int lc=0;lc<interval1;++count,++lc)
		m_xyMapping[count] = interval1Step * (lc+1);

	base = m_xyMapping[count-1];

	int interval2 = (x2-x1);
	float interval2Step = (y2-y1) / interval2;
//	qDebug("i=%f",interval2Step);

	for(int lc=0;lc<interval2;++count,++lc)
		m_xyMapping[count] = base + interval2Step * (lc+1);

	base = m_xyMapping[count-1];

        int interval3 = m_mappingRes-x2;
	float interval3Step = (1.0f - y2) / interval3;
//	qDebug("i=%f",interval3Step);

	for(int lc=0;lc<interval3;++count,++lc)
		m_xyMapping[count] = base + interval3Step * (lc+1);

//	for(int r=0;r<count;++r)
//		qDebug("%d-%f",r,m_xyMapping[r]);

	updateXYMappingMap();

	//updateTFColorMap();
	this->repaint();
	emit tfMappingChanged(getXYMappingMap(),m_tfColorMap);
}

void QTFPanel::updateTFColorMap()
{
	QColor tc;
	int colorPosx;
	float alphaPosyIdx = m_tfResolution/m_tfColorMapResoultion;
	for(int x=0;x<m_tfColorMapResoultion;++x)
	{
		colorPosx = ((x + 0.5f)/(float) (m_tfColorMapResoultion)) * (m_panelWidth);
		tc = QColor::fromRgb(m_panelImage->pixel(m_panelLMargin+colorPosx,m_panelUMargin+m_panelHeight-15));
		//alphaPosyIdx = colorPosx * m_tfPt2ResFactor;
		m_tfColorMap[4*x] = (float)tc.redF();
		m_tfColorMap[4*x+1] = (float)tc.greenF();
		m_tfColorMap[4*x+2] = (float)tc.blueF();
		m_tfColorMap[4*x+3] = alphaValue((int)(alphaPosyIdx*x));
	}
// 	cout << m_tfColorMap[4*1023 + 3] << endl;
// 	cout << m_tfColorMap[4*1022 + 3] << endl;
// 	cout << m_tfColorMap[4*1021 + 3] << endl;
// 	cout << alphaPosyIdx << endl;

	updateXYMappingMap();
	emit tfColorMapChange();

}
void QTFPanel::mouseDoubleClickEvent(QMouseEvent *event)
{
	if(event->button() == Qt::LeftButton && m_colorTickArea.contains(event->pos()))
	{
		m_isLeftClick = true;
		m_tfColorTick[m_clickedTicks].m_color = m_pTFEditor->getColorPicker()->getQColor();
		m_ifOnDrawColorTick = true;
		updatePanelImage();
		updateTFColorMap();
		repaint();
	}
}
int QTFPanel::clickOnTicks(QPoint &pt)
{
	float xposs1,yposs1,xposs2,yposs2,xposs3,yposs3;
	for(int d=0;d<m_tfColorTick.size();++d)
	{
		xposs1 = m_panelLMargin + m_tfColorTick[d].m_resX * m_tfRes2PtFactor;
		yposs1 = m_panelUMargin + m_panelHeight + 10;
		xposs2 = m_panelLMargin + m_tfColorTick[d].m_resX * m_tfRes2PtFactor - 10;
		yposs2 = m_panelUMargin + m_panelHeight + 10 + m_colorTickHeight;
		xposs3 = m_panelLMargin + m_tfColorTick[d].m_resX * m_tfRes2PtFactor + 10;
		yposs3 = m_panelUMargin + m_panelHeight + 10 + m_colorTickHeight;
		QPolygon	tri;
		tri << QPoint(xposs1,yposs1) << QPoint(xposs2,yposs2) << QPoint(xposs3,yposs3);
		if(tri.containsPoint(pt, Qt::OddEvenFill))
			return d;
	}
	return -1;
}
int QTFPanel::clickOnObjctControlBox(QPoint &pt)
{
	if(m_gaussianObjectArray.size() != 0)
	{
		float xposs,yposs;
		for(int x=0;x<m_gaussianObjectArray.size();++x)
		{
			if(x == m_clickedObjectControlBox) {
				xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 10;
				yposs = m_panelUMargin + m_panelHeight - 55;
				QRect temp1(xposs,yposs,20,45);
				if(temp1.contains(pt))
				{
					return x;
				}
			}
			xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 6;
			yposs = m_panelUMargin + m_panelHeight - 10;
			QRect temp2(xposs,yposs,12,9);
			if(temp2.contains(pt))
			{
				return x;
			}
		}
	}
	return -1;
}
int QTFPanel::clickOnObjctControlBox(QPoint	&pt,int &side)
{
	if(m_gaussianObjectArray.size() != 0)
	{
		float xposs,yposs;
		for(int x=0;x<m_gaussianObjectArray.size();++x)
		{
			if(x == m_clickedObjectControlBox) {
				xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 10;
				yposs = m_panelUMargin + m_panelHeight - 55;
				QRect temp1(xposs,yposs,20,45);
				if(temp1.contains(pt))
				{
					QRect temp2(xposs,yposs,20,15);
					QRect temp3(xposs,yposs+15,20,15);
					QRect temp4(xposs,yposs+30,20,15);
					if(temp2.contains(pt))
						side = 1;  // top
					else if (temp3.contains(pt))
						side = 2;  // middle
					else if (temp4.contains(pt))
						side = 3;  // bottom
					return x;
				}
			}
			side = 0;
			xposs = m_panelLMargin + m_gaussianObjectArray[x].m_mean * m_panelWidth - 6;
			yposs = m_panelUMargin + m_panelHeight - 10;
			QRect temp1(xposs,yposs,12,9);
			if(temp1.contains(pt)) return x;
		}
	}
	return -1;
}
void QTFPanel::mouseMoveEvent(QMouseEvent *event)
{
	bool noUpdate = false;
	if(m_isLeftClick && m_ifOnMovingObjectControlBox)
	{
		m_nowPoint = event->pos();
		float xposs = (float)(m_nowPoint.rx() - m_panelLMargin)/m_panelWidth;
		float xpossold = (float)(m_lastPoint.rx() - m_panelLMargin)/m_panelWidth;
		float ypossnow = (float)(m_panelUMargin+m_panelHeight-(m_nowPoint.ry() - m_panelUMargin)) / m_panelHeight;
		float ypossold = (float)(m_panelUMargin+m_panelHeight-(m_lastPoint.ry() - m_panelUMargin)) / m_panelHeight;
		float diffx = xposs - xpossold;
		float diffy = ypossnow - ypossold;

		if(xposs < 0.0f) xposs = 0.0f;
		else if(xposs > 1.0f) xposs = 1.0f;

		if(m_objectControlBoxSide==1) // top : vertical scale
		{
			m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor += 0.1*diffy;
			if(m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor < 0.0f)
				m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor = 0.0f;
			else if(m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor > 1.0f)
				m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor = 1.0f;
		}
		else if(m_objectControlBoxSide==2) // middle : horizontal scale
		{
			double oldsigma = m_gaussianObjectArray[m_clickedObjectControlBox].m_sigma;
			m_gaussianObjectArray[m_clickedObjectControlBox].m_sigma += 0.1*diffx;
			if(m_gaussianObjectArray[m_clickedObjectControlBox].m_sigma <= 0.0f)
				m_gaussianObjectArray[m_clickedObjectControlBox].m_sigma = 1e-3;
			double newheight = m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor * (1.0/(oldsigma*sqrt(2.0*3.1415926))) / (1.0/(m_gaussianObjectArray[m_clickedObjectControlBox].m_sigma*sqrt(2.0*3.1415926)));
			m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor = newheight;
		}
		else if(m_objectControlBoxSide==3) // bottom : horizontal move
		{
			m_gaussianObjectArray[m_clickedObjectControlBox].m_mean = xposs;
		}
//		if(m_objectControlBoxSide==2) // left
//		{
//			m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor += 0.1*diffy;
//			if(m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor < 0.0f)
//				m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor = 0.0f;
//			else if(m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor > 1.0f)
//				m_gaussianObjectArray[m_clickedObjectControlBox].m_heightFactor = 1.0f;
//		}
//		else if((m_objectControlBoxSide==1) && (m_ifCTRLPressed == false))// right side
//		{
//			m_gaussianObjectArray[m_clickedObjectControlBox].m_sigma += 0.1*diffy;
//			if(m_gaussianObjectArray[m_clickedObjectControlBox].m_sigma <= 0.0f)
//				m_gaussianObjectArray[m_clickedObjectControlBox].m_sigma = 1e-3;
//		}
//		else if((m_objectControlBoxSide==1) && (m_ifCTRLPressed == true))// right side
//		{
//		}
		m_gaussianObjectArray[m_clickedObjectControlBox].update();
		m_lastPoint = m_nowPoint;
		updateTFColorMap();
		repaint();
	}
	else if(m_isLeftClick && m_ifOnDrawing)
	{
		m_nowPoint = event->pos();
		m_nowResIndex = (m_nowPoint.rx() - m_panelLMargin) * m_tfPt2ResFactor;
		if(m_nowResIndex < 0) m_nowResIndex = 0;
		else if(m_nowResIndex >= m_tfResolution) m_nowResIndex = m_tfResolution-1;

		int vertSliderValue = m_vertTranSlider->value();
		float lowerPart = (float)vertSliderValue/m_vertTranSliderRange;
		float upperPart = 1.0f - lowerPart;


		float yheight = (float)(m_panelHeight - (m_nowPoint.ry() - m_panelUMargin));
		float halfHeight = 0.5 * m_panelHeight;
		if(yheight <= halfHeight)
		{yval = (yheight/halfHeight)*lowerPart;}
		else
		{yval = ((yheight - halfHeight)/halfHeight) * upperPart + lowerPart;}

		if(yval > 1.0f) yval = 1.0f;
		else if(yval < 0.0f) yval = 0.0f;

		if(!(event->modifiers() & Qt::ControlModifier) || (fabs(m_tfDrawArray[m_nowResIndex]) > 0.00001))
			m_tfDrawArray[m_nowResIndex] = yval;
		m_lastPoint = m_nowPoint;
		if(abs(m_nowResIndex - m_lastResIndex) > 1)
		{
			interpolateResPoint((event->modifiers() & Qt::ControlModifier));
		}
		m_lastResIndex = m_nowResIndex;
		updateTFColorMap();
		repaint();
	}
	else if(m_isRightClick && m_ifOnDrawing)
	{
		m_nowPoint = event->pos();
		m_nowResIndex = (m_lastPoint.rx() - m_panelLMargin) * m_tfPt2ResFactor;
		if(m_nowResIndex < 0) m_nowResIndex = 0;
		else if(m_nowResIndex >= m_tfResolution) m_nowResIndex = m_tfResolution-1;
		m_tfDrawArray[m_nowResIndex] = 0.0f;
		m_lastPoint = m_nowPoint;
		if(abs(m_nowResIndex - m_lastResIndex) > 1)
		{
			yval = lastyval = 0;
			interpolateResPoint();
		}
		m_lastResIndex = m_nowResIndex;
		updateTFColorMap();
		repaint();
		//	m_optionMenu->exec(QCursor::pos());
	}
	else if(m_isLeftClick && m_ifOnDrawColorTick)
	{
		QPoint	pt = event->pos();
		int xResPos = (pt.rx() - m_panelLMargin) * m_tfPt2ResFactor;
		if(xResPos < 0) xResPos = 0;
		else if(xResPos > m_tfResolution) xResPos = m_tfResolution;
		m_tfColorTick[m_clickedTicks].m_resX = xResPos;

		updatePanelImage();
		updateTFColorMap();
		repaint();
	}
	else if(((m_clickedObjectControlBox = clickOnObjctControlBox((QPoint&)event->pos(),m_objectControlBoxSide)) != -1))
	{
		repaint();
	}
	else if(!(event->x() < m_panelLMargin || event->x() > (width() - m_panelRMargin) || event->y() < m_panelUMargin || event->y() > (height() - m_panelDMargin))) {
		updatePanelImage();
		repaint();
		noUpdate = true;
	}
	if(!noUpdate && m_instant)
		emit tfChanged(m_tfColorMap);
}
void QTFPanel::mouseReleaseEvent(QMouseEvent* event)
{
	if(event->button() == Qt::LeftButton && m_isLeftClick)
		m_isLeftClick = false;
	if(event->button() == Qt::RightButton && m_isRightClick)
		m_isRightClick = false;
	m_ifOnDrawing = false;
	if(m_ifOnDrawColorTick)
	{
		m_ifOnDrawColorTick = false;
	}
	m_ifOnMovingObjectControlBox = false;
	generateZeroRanges();
	emit tfChanged(m_tfColorMap);
}

void QTFPanel::generateZeroRanges() {
	m_zeroRangesArray.clear();
	float start = -1;
	for(int i = 0; i < m_tfResolution; i++) {
		if(alphaValue(i) < 1e-3) {
			start = i/(m_tfResolution - 1);
			i++;
			for(; i < m_tfResolution; i++) {
				if(alphaValue(i) > 1e-3) {
					m_zeroRangesArray.append(ZeroRange(start, (i-1)/(m_tfResolution - 1)));
					start = -1;
					break;
				}
			}
		}
	}
	if(start != -1) {
		m_zeroRangesArray.append(ZeroRange(start, 1.0f));
	}
// 	for(int i = 0; i < m_zeroRangesArray.size(); i++) {
// 		cout << m_zeroRangesArray[i].start << " " << m_zeroRangesArray[i].end << endl;
// 	}
}

float QTFPanel::alphaValue(int x) {
	if(x < 0 || x >= m_tfResolution)
		return 0;
	float alpha;
	alpha = m_tfDrawArray[x];
	if(m_combineMode == 0) { //or
		for(int i = 0; i < m_gaussianObjectArray.size(); i++) {
			alpha = alpha > m_gaussianObjectArray[i].m_distribution[x] ?
					alpha : m_gaussianObjectArray[i].m_distribution[x];
		}
	} else if(m_combineMode == 1) { //and
		for(int i = 0; i < m_gaussianObjectArray.size(); i++) {
			alpha += m_gaussianObjectArray[i].m_distribution[x];
		}
	}
	alpha = alpha > 1.f ? 1.f : alpha < 1e-10 ? 0 : alpha;
	return alpha;
}

/*
void QTFPanel::keyPressEvent(QKeyEvent *event)
{
	if(event->key() == Qt::Key_Control)
		m_ifCTRLPressed = true;
	if(event->key() == Qt::Key_Shift)
		m_ifSHIFTPressed = true;
}
void QTFPanel::keyReleaseEvent(QKeyEvent *event)
{
	if(event->key() == Qt::Key_Control)
		m_ifCTRLPressed = false;
	if(event->key() == Qt::Key_Shift)
		m_ifSHIFTPressed = false;
}
*/
void QTFPanel::interpolateResPoint(bool ctrl)
{
	int		resIdx1,resIdx2;
	float	resIdx1y,resIdx2y;

	if(m_nowResIndex > m_lastResIndex)
	{
		resIdx1 = m_lastResIndex;
		resIdx2 = m_nowResIndex;
		resIdx1y = lastyval;
		resIdx2y = yval;
	}
	else
	{
		resIdx2 = m_lastResIndex;
		resIdx1 = m_nowResIndex;
		resIdx1y = yval;
		resIdx2y = lastyval;
	}
	int nosPt = resIdx2 - resIdx1 - 1;
	float slope = (resIdx2y-resIdx1y)/(resIdx2-resIdx1);

	for(int x=1;x<=nosPt;++x)
	{
		float val = resIdx1y + slope * x;

		if((!ctrl) || (fabs(m_tfDrawArray[resIdx1 + x]) > 0.00001))
			m_tfDrawArray[resIdx1+x] = val;
	}
	lastyval = yval;
}


void QTFPanel::changeSelectedTickColor(QColor &cr)
{
	if(m_clickedTicks >= 0 && m_clickedTicks<m_tfColorTick.size())
	{
		m_tfColorTick[m_clickedTicks].m_color = cr;
		updatePanelImage();
		updateTFColorMap();
		repaint();
		if(m_instant)
			emit tfChanged(m_tfColorMap);
	}
}
void QTFPanel::vertTranSliderChange(int value)
{
//	int minn = m_vertTranSlider->minimum();
//	int maxx = m_vertTranSlider->maximum();
//	int vall = value;

	//float perct = (float)vall/(float)(maxx-minn+1);
	m_vertTranPercetangeString = QString("%1").arg(value);
	m_vertStringNeedUpdate = true;
	repaint();
}
void QTFPanel::functionButtonPressed()
{
	//static int counter = 0;

	GaussianObject temp(0.5,0.03,0.05,m_tfResolution);
	temp.update();
	m_gaussianObjectArray << temp;
	updatePanelImage();
	updateTFColorMap();
	generateZeroRanges();
	repaint();
	emit tfChanged(m_tfColorMap);
	//counter++;
}
//double QTFPanel::gaussianGen(double x,double mean,double sigma)
//{
//	double val = 1.0/(sigma*sqrt(2.0*3.1415926)) * exp(-(x-mean)*(x-mean)/(2.0*sigma*sigma));
//	return val;
//}
void QTFPanel::settingButtonPressed()
{
	m_settingMenu->exec(QCursor::pos());
}
void QTFPanel::mappingButtonPressed(){
	if(m_mappingEditor->isVisible())
		m_mappingEditor->hide();
	else
		m_mappingEditor->show();
}

void QTFPanel::changeBM2None()
{
	changeBM2NoneAct->setChecked(true);
	changeBM2DotLineAct->setChecked(false);
	m_backgroundMesh = 0;
	//updatePanelImage();
	repaint();
}
void QTFPanel::changeBM2DotLine()
{
	changeBM2NoneAct->setChecked(false);
	changeBM2DotLineAct->setChecked(true);
	m_backgroundMesh = 1;
	//updatePanelImage();
	repaint();
}
void QTFPanel::changeCombine2Or() {
	changeCombine2OrAct->setChecked(true);
	changeCombine2AndAct->setChecked(false);
	m_combineMode = 0;
	updateTFColorMap();
	repaint();
}
void QTFPanel::changeCombine2And() {
	changeCombine2OrAct->setChecked(false);
	changeCombine2AndAct->setChecked(true);
	m_combineMode = 1;
	updateTFColorMap();
	repaint();
}

void QTFPanel::saveFile() {
	QString filename = QFileDialog::getSaveFileName(this, tr("Save Settings"), ".", tr("TFEditor files (*.tfe);;VisTransport files (*.vtf);;All files (*.*)"));
	if(!filename.isEmpty()) {
		if(!filename.contains(QString(".")))
			filename += ".tfe";

		QFile file(filename);
		if(!file.open(QIODevice::WriteOnly))
			return;
		if(filename.contains(QString("tfe")))
			saveFile(file);
		else if(filename.contains(QString("vtf")))
			saveFileVisTransport(file);
		file.close();
	}
}
void QTFPanel::openFile() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Load Saved Settings"), ".", tr("TFEditor files (*.tfe);;VisTransport files (*.vtf);;All files (*.*)"));
	if(!filename.isEmpty()) {
		QFile file(filename);

		if(!file.open(QIODevice::ReadOnly))
			return;

		if(filename.contains(QString("tfe")))
			openFile(file);
		else if(filename.contains(QString("vtf")))
			openFileVisTransport(file);
	//	else if(filename.contains(QString("tsf"),Qt::CaseInsensitive))
	//		openTSFFile(file);
		else //if(filename.constains(QString("tfe"),Qt::CaseInsensitive))
			openFile(file);

		file.close();
		emit tfChanged(m_tfColorMap);
	}
}
/*
QFile& QTFPanel::openTSFFile(QFile& file){
	if(!file.isReadable())
		return file;
	QTextStream inp(&file);
	inp >> m_tfResolution;

	float *buf = new float[m_tfResolution*3];

	delete [] m_tfDrawArray;
	m_tfDrawArray = new float[(int)m_tfResolution];

	for(int i=0;i<m_tfResolution;++i)
		inp >> buf[3*i] >> buf[3*i+1] >> buf[3*i+2] >> m_tfDrawArray[i];

	int colorsize = 2;
	colorsize = (int)m_tfResolution/20;
	if(colorsize < 3) colorsize = 0;
	else colorsize -= 2; // get rid of first and last points

	m_tfColorTick.clear();
	QColor c;
	// first
	c.setRgbF(buf[0],buf[1],buf[2]);
	TFColorTick	firsttick(0.0f,c);
	m_tfColorTick.push_back(firsttick);
	// in between
	float stepX = (float)1.0f/(colorsize+1.0);
	for(int i = 0; i < colorsize; i++) {
		float resX = stepX*(i+1);
		c.setRgbF(buf[(i+1)*3], buf[(i+1)*3+1], buf[(i+1)*3+2]);
		TFColorTick tick(resX, c);
		m_tfColorTick.push_back(tick);
	}
	// last
	c.setRgbF(buf[int(3*(m_tfResolution-1))],buf[int(3*(m_tfResolution-1)+1)],buf[int(3*(m_tfResolution-1)+2)]);
	TFColorTick	lasttick(1.0f,c);
	m_tfColorTick.push_back(lasttick);

	delete [] buf;

	updatePanelImage();
	updateTFColorMap();
	generateZeroRanges();
	repaint();
	return file;
}
 */
void QTFPanel::tfExternalChanged(float w, float * draw, QVector<GaussianObject> * gaussians, QVector<TFColorTick> * colors, bool silent) {
	m_tfResolution = w;
	delete [] m_tfDrawArray;
	m_tfDrawArray = new float[(int)m_tfResolution];
	memcpy(m_tfDrawArray, draw, sizeof(float) * m_tfResolution);
	m_gaussianObjectArray.clear();
	for (int i = 0; i < gaussians->size(); ++i) {
		GaussianObject obj((*gaussians)[i].m_mean,
				   (*gaussians)[i].m_sigma,
				   (*gaussians)[i].m_heightFactor,
				   m_tfResolution);
		obj.update();
		m_gaussianObjectArray.push_back(obj);
	}
	m_tfColorTick.clear();
	for (int i = 0; i < colors->size(); ++i) {
		TFColorTick tick((*colors)[i].m_resX,
				 (*colors)[i].m_color);
		m_tfColorTick.push_back(tick);
	}
	updateTFColorMap();
	if (!silent) {
		updatePanelImage();
		generateZeroRanges();
		repaint();
	}
	emit tfChanged(m_tfColorMap, false);
}
QIODevice& QTFPanel::openFile(QIODevice& file) {
	if(!file.isReadable())
		return file;
	file.read((char*)&m_tfResolution, 4);
	delete [] m_tfDrawArray;
	m_tfDrawArray = new float[(int)m_tfResolution];
	file.read((char*)m_tfDrawArray, (int)m_tfResolution*4);
	int size;
	file.read((char*)&size, 4);
	m_gaussianObjectArray.clear();
	double t[3];
	for(int i = 0; i < size; i++) {
		file.read((char*)t, 24);
		GaussianObject obj(t[0], t[1], t[2], m_tfResolution);
		obj.update();
		m_gaussianObjectArray.push_back(obj);
	}
	file.read((char*)&size, 4);
	float resX;
	m_tfColorTick.clear();
	QColor c;
	for(int i = 0; i < size; i++) {
		file.read((char*)&resX, 4);
		file.read((char*)t, 24);
		c.setRgbF(t[0], t[1], t[2]);
		TFColorTick tick(resX, c);
		m_tfColorTick.push_back(tick);
	}
	file.read((char*)&m_combineMode, 4);
	if(m_combineMode == 0) {
		changeCombine2OrAct->setChecked(true);
		changeCombine2AndAct->setChecked(false);
	} else if(m_combineMode == 1) {
		changeCombine2OrAct->setChecked(false);
		changeCombine2AndAct->setChecked(true);
	}
	file.read((char*)&size, 4);
	m_vertTranSlider->setValue(size);
	file.read((char*)&m_backgroundMesh, 4);
	file.read((char*)t, 24);
	c.setRgbF(t[0], t[1], t[2]);
	m_pTFEditor->getColorMap()->changeBGColor(c);
	updatePanelImage();
	updateTFColorMap();
	generateZeroRanges();
	repaint();
	return file;
}
QIODevice& QTFPanel::saveFile(QIODevice& file) {
	if(!file.isWritable())
		return file;
	file.write((char*)&m_tfResolution, 4);
	file.write((char*)m_tfDrawArray, m_tfResolution*4);
	int size = m_gaussianObjectArray.size();
	file.write((char*)&size, 4);
	for(int i = 0; i < size; i++) {
		file.write((char*)&(m_gaussianObjectArray[i].m_mean), 8);
		file.write((char*)&(m_gaussianObjectArray[i].m_sigma), 8);
		file.write((char*)&(m_gaussianObjectArray[i].m_heightFactor), 8);
	}
	size = m_tfColorTick.size();
	file.write((char*)&size, 4);
	double t[3];
	for(int i = 0; i < size; i++) {
		file.write((char*)&(m_tfColorTick[i].m_resX), 4);
		m_tfColorTick[i].m_color.getRgbF(t, t+1, t+2);
		file.write((char*)t, 24);
	}
	file.write((char*)&m_combineMode, 4);
	size = m_vertTranSlider->value();
	file.write((char*)&size, 4);
	file.write((char*)&m_backgroundMesh, 4);
	m_pTFEditor->getColorMap()->getBackgroundColor().getRgbF(t, t+1, t+2);
	file.write((char*)t, 24);
	return file;
}
void QTFPanel::loadSettings(TF & tf) {
	m_tfResolution = tf.tfResolution;
	delete [] m_tfDrawArray;
	m_tfDrawArray = new float[(int)m_tfResolution];
	for (int i = 0; i < m_tfResolution; ++i) m_tfDrawArray[i] = tf.tfDrawArray[i];

	m_gaussianObjectArray.clear();
	for (int i = 0; i < tf.gaussianObjectArray.size(); ++i) {
		GaussianObject obj(tf.gaussianObjectArray[i].m_mean,
				   tf.gaussianObjectArray[i].m_sigma,
				   tf.gaussianObjectArray[i].m_heightFactor,
				   m_tfResolution);
		obj.update();
		m_gaussianObjectArray.push_back(obj);
	}

	m_tfColorTick.clear();
	for (int i = 0; i < tf.tfColorTick.size(); ++i) {
		TFColorTick tick(tf.tfColorTick[i].m_resX, tf.tfColorTick[i].m_color);
		m_tfColorTick.push_back(tick);
	}
	m_combineMode = tf.combineMode;
	if(m_combineMode == 0) {
		changeCombine2OrAct->setChecked(true);
		changeCombine2AndAct->setChecked(false);
	} else if(m_combineMode == 1) {
		changeCombine2OrAct->setChecked(false);
		changeCombine2AndAct->setChecked(true);
	}
	m_vertTranSlider->setValue(tf.tranSliderValue);
	m_backgroundMesh = tf.backgroundMesh;
	m_pTFEditor->getColorMap()->changeBGColor(tf.backgroundColor);
	updatePanelImage();
	updateTFColorMap();
	generateZeroRanges();
	repaint();
}
void QTFPanel::saveSettings(TF & tf) {
	tf.tfResolution = m_tfResolution;
	delete [] tf.tfDrawArray;
	tf.tfDrawArray = new float[(int)m_tfResolution];
	for (int i = 0; i < m_tfResolution; ++i) tf.tfDrawArray[i] = m_tfDrawArray[i];
	tf.gaussianObjectArray.clear();
	for (int i = 0; i < m_gaussianObjectArray.size(); ++i) {
		GaussianObject obj(m_gaussianObjectArray[i].m_mean,
				   m_gaussianObjectArray[i].m_sigma,
				   m_gaussianObjectArray[i].m_heightFactor,
				   m_tfResolution);
		obj.update();
		tf.gaussianObjectArray.push_back(obj);
	}
	tf.tfColorTick.clear();
	for (int i = 0; i < m_tfColorTick.size(); ++i) {
		TFColorTick tick(m_tfColorTick[i].m_resX, m_tfColorTick[i].m_color);
		tf.tfColorTick.push_back(tick);
	}
	tf.combineMode = m_combineMode;
	tf.tranSliderValue = m_vertTranSlider->value();
	tf.backgroundMesh = m_backgroundMesh;
	tf.backgroundColor = m_pTFEditor->getColorMap()->getBackgroundColor();

}
QFile& QTFPanel::saveFileVisTransport(QFile& file){
	if(!file.isWritable())
		return file;
	for(int x=0;x<m_tfColorMapResoultion;++x)
	{
		unsigned char aa = m_tfColorMap[4*x+3]*255;
		unsigned char rr = m_tfColorMap[4*x  ]*255;
		unsigned char gg = m_tfColorMap[4*x+1]*255;
		unsigned char bb = m_tfColorMap[4*x+2]*255;

		file.write((const char*)(&aa), 1); // A
		file.write((const char*)&rr, 1); // R
		file.write((const char*)&gg, 1); // G
		file.write((const char*)&bb, 1); // B
	}


	return file;
}
QFile& QTFPanel::openFileVisTransport(QFile& file){
	return file;
}
void QTFPanel::toggleInstant() {
	m_instant = !m_instant;
	toggleInstantAct->setChecked(m_instant);
}


void QTFPanel::togglePositionLabel(bool t) {
	drawLabels = t;
	updatePanelImage();
	repaint();
	setMouseTracking(t);
}

void QTFPanel::toggleXYMapping(){
	turnOnNonlinearXYMapping = !turnOnNonlinearXYMapping;
	//updatePanelImage();
	updateXYMappingMap();
	repaint();

	emit tfMappingChanged(getXYMappingMap(),m_tfColorMap);
}
float * QTFPanel::computeColorMapFromTF(TF & tf, float * tfColorMap) {
	// updatePanelImage
	QPainter painter(m_tfImage);
	m_tfImage->fill(qRgb(255, 255, 255));

	QLinearGradient linearGrad(m_panelLMargin, 1, rect().width()-m_panelRMargin, 1);
	for(int x=0;x<tf.tfColorTick.size();++x)
	{
		float res = (tf.tfColorTick[x].m_resX)/tf.tfResolution;
		linearGrad.setColorAt(res,tf.tfColorTick[x].m_color);
	}
	painter.fillRect(QRect(m_panelLMargin, 0, m_panelWidth, 1), linearGrad);

	// updateTFColorMap
	if (!tfColorMap) tfColorMap = new float[4 * m_tfColorMapResoultion];
	QColor tc;
	int colorPosx;
	float alphaPosyIdx = (float)tf.tfResolution/(float)m_tfColorMapResoultion;

	for(int i = 0; i < tf.gaussianObjectArray.size(); i++)
		tf.gaussianObjectArray[i].update();
	
	for(int x=0;x<m_tfColorMapResoultion;++x)
	{
		colorPosx = ((x + 0.5f)/(float) (m_tfColorMapResoultion)) * (m_panelWidth);
		tc = QColor::fromRgb(m_tfImage->pixel(m_panelLMargin+colorPosx, 0));

		tfColorMap[4*x] = (float)tc.redF();
		tfColorMap[4*x+1] = (float)tc.greenF();
		tfColorMap[4*x+2] = (float)tc.blueF();

		// alphaValue
		colorPosx = alphaPosyIdx*x;
		float alpha = tf.tfDrawArray[colorPosx];
		if (colorPosx < 0 || colorPosx >= tf.tfResolution) alpha = 0.0f;
		else if(tf.combineMode == 0) { //or
			for(int i = 0; i < tf.gaussianObjectArray.size(); i++) {
				alpha = alpha > tf.gaussianObjectArray[i].m_distribution[colorPosx] ?
						alpha : tf.gaussianObjectArray[i].m_distribution[colorPosx];
			}
		} else if(tf.combineMode == 1) { //and
			for(int i = 0; i < tf.gaussianObjectArray.size(); i++) {
				alpha += tf.gaussianObjectArray[i].m_distribution[colorPosx];
			}
		}
		alpha = alpha > 1.f ? 1.f : alpha < 1e-10 ? 0 : alpha;
		tfColorMap[4*x+3] = alpha;
	}
	return tfColorMap;
}

QDataStream & operator<<(QDataStream & out, const GaussianObject & go) {
	out << go.m_mean << go.m_sigma << go.m_heightFactor << go.m_resolution;
	return out;
}
QDataStream & operator>>(QDataStream & in, GaussianObject & go) {
	in >> go.m_mean >> go.m_sigma >> go.m_heightFactor >> go.m_resolution;
	return in;
}
QDataStream & operator<<(QDataStream & out, const TFColorTick & ct) {
	out << ct.m_resX << ct.m_color;
	return out;
}
QDataStream & operator>>(QDataStream & in, TFColorTick & ct) {
	in >> ct.m_resX >> ct.m_color;
	return in;
}
QDataStream & operator<<(QDataStream & out, const TF & tf) {
	out << tf.tfResolution;
	for (int i = 0; i < (int)tf.tfResolution; ++i) out << tf.tfDrawArray[i];
	out << tf.gaussianObjectArray << tf.tfColorTick << tf.combineMode
	    << tf.tranSliderValue << tf.backgroundMesh << tf.backgroundColor;
	return out;
}
QDataStream & operator>>(QDataStream & in, TF & tf) {
	in >> tf.tfResolution;
	if (tf.tfDrawArray) delete [] tf.tfDrawArray;
	tf.tfDrawArray = new float[(int)tf.tfResolution];
	for (int i = 0; i < (int)tf.tfResolution; ++i) in >> tf.tfDrawArray[i];
	in >> tf.gaussianObjectArray >> tf.tfColorTick >> tf.combineMode
	    >> tf.tranSliderValue >> tf.backgroundMesh >> tf.backgroundColor;
	return in;
}
