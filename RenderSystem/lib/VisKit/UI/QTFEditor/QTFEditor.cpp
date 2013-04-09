#include "QTFEditor.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include "histogram.h"
#include <iostream>
#include <QBuffer>
using namespace std;

QTFEditor::QTFEditor(int transferWidth, QWidget *parent, const char *file)
:QWidget(parent)
{
	Q_INIT_RESOURCE(QTFEditor);
#ifndef _TESTAPP
	setWindowFlags(Qt::Tool);
#endif
	setWindowTitle(tr("Transfer Function Editor"));
	this->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
	m_histogramPanelSize	= QSize(400,100);
	m_tfPanelSize			= QSize(400,190);
	m_tfColorMapPanelSize	= QSize(400,25);
	m_colorPickerSize		= QSize(400,190);
	
	int maxXSize = -1000;
	int maxYSize = 0;
	if(maxXSize < m_histogramPanelSize.width())	 maxXSize = m_histogramPanelSize.width();
	if(maxXSize < m_tfPanelSize.width())		 maxXSize = m_tfPanelSize.width();
	if(maxXSize < m_tfColorMapPanelSize.width()) maxXSize = m_tfColorMapPanelSize.width();
	if(maxXSize < m_colorPickerSize.width()) 	 maxXSize = m_colorPickerSize.width();

	maxYSize = m_histogramPanelSize.height() + m_tfPanelSize.height() +
			   m_tfColorMapPanelSize.height() + m_colorPickerSize.height();
	
	m_tfEditorSize	= QSize(maxXSize,maxYSize);		
	
	this->setMinimumSize(m_tfEditorSize);
	m_histogramData = new Histogram(256);
	setHistogramMinMax(0.0,1.0);

	if(file) m_tfefilename = QString(file);
	else m_tfefilename = QString("defaultTF.tfe");

	initLayout(transferWidth);
}
QTFEditor::~QTFEditor()
{
	delete m_histogramData;
	writeDefaultSettings();
}
void QTFEditor::newHistogram(Histogram* hist) {
	if(hist)
		m_histogramData = hist;
	m_histogramPanel->updateHistogram();
}
void QTFEditor::initLayout(int transferWidth)
{
	QHBoxLayout	*hlayout = new QHBoxLayout(this);
	hlayout->setSpacing(0);
	hlayout->setMargin(0);

	QVBoxLayout *vlayout = new QVBoxLayout;
	vlayout->setAlignment(Qt::AlignTop|Qt::AlignLeft);
	
	m_histogramPanel = new QHistogram(this);
	m_histogramPanel->resize(m_histogramPanelSize);
	m_histogramPanel->updateHistogram();
	
	m_tfColorMapPanel = new QTFColorMap(this);
	m_tfColorMapPanel->resize(m_tfColorMapPanelSize);
	
	m_tfPanel = new QTFPanel(transferWidth, this);
	m_tfPanel->resize(m_tfPanelSize);
	m_tfPanel->setHideWith(m_tfColorMapPanel);
	
	connect(m_tfPanel,SIGNAL(tfColorMapChange()),this,SLOT(changedTFColorMap()));
	
	m_colorPicker = new QColorPicker(this);
	connect(m_colorPicker,SIGNAL(colorChanged(int)),this, SLOT(changeTickColor(int)));
	connect(m_colorPicker,SIGNAL(colorPassiveChanged(int)),this, SLOT(changeTickColor(int)));
	m_tfColorMapPanel->setHideAlso(m_colorPicker);
	vlayout->addWidget(m_histogramPanel);
	vlayout->addWidget(m_tfColorMapPanel);
	vlayout->addWidget(m_tfPanel);
	
	vlayout->addWidget(m_colorPicker);
	hlayout->addLayout(vlayout);
	setLayout(hlayout);	
	readDefaultSettings();
}

void QTFEditor::setHistogramMinMax(double min, double max) {
	m_histogramData->setMinMax(min,max);
}

void QTFEditor::incrementHistogram(double value) {
	m_histogramData->increment(value);
}
void QTFEditor::readDefaultSettings()
{
	QFile inpFile(m_tfefilename);
	//loadSettings(inpFile);
	//qDebug("%s\n",m_tfefilename.toAscii().constData());
	if(!inpFile.open(QIODevice::ReadOnly)){	}
	else{	loadSettings(inpFile);	}
	inpFile.close();
}
void QTFEditor::writeDefaultSettings()
{
	QFile outFile(m_tfefilename);
	if(!outFile.open(QIODevice::WriteOnly))
	{return;}

	saveSettings(outFile);
	//QTextStream	out(&outFile);
	//
	//out << m_colorPalette->getColorCounter() << "\n";
	//QColor nc = m_colorPalette->m_nowColor->getColor();
	//qreal hhr,ssr,vvr;
	//nc.getHsvF(&hhr,&ssr,&vvr);
	//out << hhr <<" "<< ssr <<" "<< vvr << "\n";
	//
	//for(int x=0;x<m_colorPalette->m_colorArray.size();++x)
	//{
	//	nc = m_colorPalette->m_colorArray[x]->getColor();
	//	nc.getHsvF(&hhr,&ssr,&vvr);
	//	out << hhr <<" "<< ssr <<" "<< vvr << "\n";
	//}
	
	outFile.close();
}
void QTFEditor::changeTickColor(int)
{
	QColor temp = m_colorPicker->getQColor();
	m_tfPanel->changeSelectedTickColor(temp);
}
void QTFEditor::changedTFColorMap()
{
	m_tfColorMapPanel->updateTFColorMap();
}
void QTFEditor::setHistogram(Histogram* h) {
	if(h != m_histogramData)
		delete m_histogramData;
	m_histogramData = h;
	m_histogramPanel->updateHistogram();
}
void QTFEditor::updateHistogram(Histogram * h) {
	*m_histogramData = *h;
}
QIODevice& QTFEditor::saveSettings(QIODevice& file) {
	if(!file.isWritable())
	{
		qWarning("Error: IODevice not writable.");
		//cout << "saveSetting:" << file.fileName().toStdString() << endl;
		return file;
	}
	return m_tfPanel->saveFile(file);
}

QIODevice& QTFEditor::loadSettings(QIODevice& file) {
	if(!file.isReadable())
		return file;
	
	return m_tfPanel->openFile(file);	
}
void QTFEditor::saveSettings(TF & tf) {
	m_tfPanel->saveSettings(tf);
}
void QTFEditor::loadSettings(TF & tf) {
	m_tfPanel->loadSettings(tf);
}
QColor QTFEditor::getColor(float pos){
	if(pos > 1.0f) pos = 1.0f;
	else if(pos < 0.0f) pos = 0.0f;
	
	int idx = pos * (getTFColorMapResolution()-1);
	float rr = getTFColorMap()[4*idx];
	float gg = getTFColorMap()[4*idx+1];
	float bb = getTFColorMap()[4*idx+2];
	float aa = getTFColorMap()[4*idx+3];
	return QColor::fromRgbF(rr,gg,bb,aa);	
}

void QTFEditor::loadByteArray(QByteArray ba) {
	QBuffer buf(&ba);
	buf.open(QIODevice::ReadOnly);
	loadSettings(buf);
}
