#include "QColorMonitor.h"
#include "QColorPicker.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSlider>
#include <QSpinBox>
#include <QLabel>
#include <QColor>

QColorMonitor::QColorMonitor(QWidget *parent):
QWidget(parent)
{
	m_pColorPicker = (QColorPicker*)parent;

	connect(m_pColorPicker,SIGNAL(colorChanged(int)),this, SLOT(adjustColorSwitches(int)));
	initLayout();
	adjustColorSwitches(1); // update H
	adjustColorSwitches(2); // update S,V
}
void QColorMonitor::initLayout()
{
	QVBoxLayout	*vlayout = new QVBoxLayout(this);
	vlayout->setSpacing(0);
	vlayout->setMargin(0);
	vlayout->setAlignment(Qt::AlignTop|Qt::AlignLeft);
	QHBoxLayout	*layout1 = new QHBoxLayout;
	m_HSlider = new QSlider(Qt::Horizontal,this);
	m_HSlider->setTickInterval(20);
	m_HSlider->setTickPosition(QSlider::TicksBelow);
	m_HSlider->setMaximum(255);
	m_HSlider->setMinimum(0);
	m_HBox	= new QSpinBox(this);
	m_HBox->setRange(0,255);
	m_HBox->setAccelerated(true);
	m_HBox->setWrapping(true);
	layout1->addWidget(new QLabel(tr(" H")));
	layout1->addWidget(m_HSlider);
	layout1->addWidget(m_HBox);

	QHBoxLayout	*layout2 = new QHBoxLayout;
	m_SSlider = new QSlider(Qt::Horizontal,this);
	m_SSlider->setTickInterval(20);
	m_SSlider->setTickPosition(QSlider::TicksBelow);
	m_SSlider->setMaximum(255);
	m_SSlider->setMinimum(0);
	m_SBox	= new QSpinBox(this);
	m_SBox->setRange(0,255);
	m_SBox->setAccelerated(true);
	m_SBox->setWrapping(true);
	layout2->addWidget(new QLabel(tr(" S")));
	layout2->addWidget(m_SSlider);
	layout2->addWidget(m_SBox);

	QHBoxLayout	*layout3 = new QHBoxLayout;
	m_VSlider = new QSlider(Qt::Horizontal,this);
	m_VSlider->setTickInterval(20);
	m_VSlider->setTickPosition(QSlider::TicksBelow);
	m_VSlider->setMaximum(255);
	m_VSlider->setMinimum(0);
	m_VBox	= new QSpinBox(this);
	m_VBox->setRange(0,255);
	m_VBox->setAccelerated(true);
	m_VBox->setWrapping(true);
	layout3->addWidget(new QLabel(tr(" V")));
	layout3->addWidget(m_VSlider);
	layout3->addWidget(m_VBox);

	QHBoxLayout	*layout4 = new QHBoxLayout;
//	m_RSlider = new QSlider(Qt::Horizontal,this);
//	m_RSlider->setTickInterval(20);
//	m_RSlider->setTickPosition(QSlider::TicksBelow);
//	m_RSlider->setMaximum(255);
//	m_RSlider->setMinimum(0);
	m_RBox	= new QSpinBox(this);
	m_RBox->setRange(0,255);
	m_RBox->setAccelerated(true);
	layout4->addWidget(new QLabel(tr(" R")));
	//layout4->addWidget(m_RSlider);
	layout4->addWidget(m_RBox);
//	m_RSlider->setEnabled(false);
	m_RBox->setEnabled(false);

	//QHBoxLayout	*layout5 = new QHBoxLayout;
//	m_GSlider = new QSlider(Qt::Horizontal,this);
//	m_GSlider->setTickInterval(20);
//	m_GSlider->setTickPosition(QSlider::TicksBelow);
//	m_GSlider->setMaximum(255);
//	m_GSlider->setMinimum(0);
	m_GBox	= new QSpinBox(this);
	m_GBox->setRange(0,255);
	m_GBox->setAccelerated(true);
	layout4->addWidget(new QLabel(tr(" G")));
	//layout5->addWidget(m_GSlider);
	layout4->addWidget(m_GBox);
//	m_GSlider->setEnabled(false);
	m_GBox->setEnabled(false);

	//QHBoxLayout	*layout6 = new QHBoxLayout;
//	m_BSlider = new QSlider(Qt::Horizontal,this);
//	m_BSlider->setTickInterval(20);
//	m_BSlider->setTickPosition(QSlider::TicksBelow);
//	m_BSlider->setMaximum(255);
//	m_BSlider->setMinimum(0);
	m_BBox	= new QSpinBox(this);
	m_BBox->setRange(0,255);
	m_BBox->setAccelerated(true);
	layout4->addWidget(new QLabel(tr(" B")));
	//layout6->addWidget(m_BSlider);
	layout4->addWidget(m_BBox);
//	m_BSlider->setEnabled(false);
	m_BBox->setEnabled(false);
	
	connect(m_HSlider,SIGNAL(sliderMoved(int)),m_HBox,SLOT(setValue(int)));
	connect(m_HBox,SIGNAL(valueChanged(int)),m_HSlider,SLOT(setValue(int)));
	//connect(m_HSlider,SIGNAL(sliderMoved(int)),this,SLOT(updatePanelColorHSV()));
	connect(m_HBox,SIGNAL(valueChanged(int)),this,SLOT(updatePanelColorHSV()));	
	
	connect(m_SSlider,SIGNAL(sliderMoved(int)),m_SBox,SLOT(setValue(int)));
	connect(m_SBox,SIGNAL(valueChanged(int)),m_SSlider,SLOT(setValue(int)));
	//connect(m_SSlider,SIGNAL(sliderMoved(int)),this,SLOT(updatePanelColorHSV()));
	connect(m_SBox,SIGNAL(valueChanged(int)),this,SLOT(updatePanelColorHSV()));	

	connect(m_VSlider,SIGNAL(sliderMoved(int)),m_VBox,SLOT(setValue(int)));
	connect(m_VBox,SIGNAL(valueChanged(int)),m_VSlider,SLOT(setValue(int)));
	//connect(m_VSlider,SIGNAL(sliderMoved(int)),this,SLOT(updatePanelColorHSV()));
	connect(m_VBox,SIGNAL(valueChanged(int)),this,SLOT(updatePanelColorHSV()));	

//	connect(m_RSlider,SIGNAL(sliderMoved(int)),m_RBox,SLOT(setValue(int)));
//	connect(m_RBox,SIGNAL(valueChanged(int)),m_RSlider,SLOT(setValue(int)));
//	connect(m_RSlider,SIGNAL(sliderMoved(int)),this,SLOT(updatePanelColorRGB()));
	//connect(m_RBox,SIGNAL(valueChanged(int)),this,SLOT(updatePanelColorRGB()));	

//	connect(m_GSlider,SIGNAL(sliderMoved(int)),m_GBox,SLOT(setValue(int)));
//	connect(m_GBox,SIGNAL(valueChanged(int)),m_GSlider,SLOT(setValue(int)));
//	connect(m_GSlider,SIGNAL(sliderMoved(int)),this,SLOT(updatePanelColorRGB()));
	//connect(m_GBox,SIGNAL(valueChanged(int)),this,SLOT(updatePanelColorRGB()));	

//	connect(m_BSlider,SIGNAL(sliderMoved(int)),m_BBox,SLOT(setValue(int)));
//	connect(m_BBox,SIGNAL(valueChanged(int)),m_BSlider,SLOT(setValue(int)));
//	connect(m_BSlider,SIGNAL(sliderMoved(int)),this,SLOT(updatePanelColorRGB()));
	//connect(m_BBox,SIGNAL(valueChanged(int)),this,SLOT(updatePanelColorRGB()));	
	
	vlayout->addLayout(layout1);
	vlayout->addLayout(layout2);
	vlayout->addLayout(layout3);
	vlayout->addLayout(layout4);
//	vlayout->addLayout(layout5);
//	vlayout->addLayout(layout6);
	setLayout(vlayout);
}	

void QColorMonitor::adjustColorSwitches(int type)
{
	int hh,ss,vv;
	int rr,gg,bb;
	m_pColorPicker->getHSVi(hh,ss,vv);
	m_pColorPicker->getRGBi(rr,gg,bb);

	if(type == 1) // change H
		m_HBox->setValue(hh);
	else if(type == 2) // change S,V
	{
		m_SBox->setValue(ss);
		m_VBox->setValue(vv);
	}
	m_RBox->setValue(rr);
	m_GBox->setValue(gg);
	m_BBox->setValue(bb);
}
void QColorMonitor::updatePanelColorHSV()
{
	float hh = (float)(m_HBox->value())/(m_HBox->maximum()-m_HBox->minimum()+1);
	float ss = (float)(m_SBox->value())/(m_SBox->maximum()-m_SBox->minimum()+1);
	float vv = (float)(m_VBox->value())/(m_VBox->maximum()-m_VBox->minimum()+1);
	
	// update RGB switches
	QColor temp = QColor::fromHsvF(hh,ss,vv);
	int rr,gg,bb; 
	temp.getRgb(&rr,&gg,&bb);
	m_RBox->setValue(rr);
	m_GBox->setValue(gg);
	m_BBox->setValue(bb);	
	
	m_pColorPicker->updateColorPanelHSVf(hh,ss,vv);
}
void QColorMonitor::updatePanelColorRGB()
{
//	float rr = (float)(m_RBox->value())/(m_RBox->maximum()-m_RBox->minimum()+1);
//	float gg = (float)(m_GBox->value())/(m_GBox->maximum()-m_GBox->minimum()+1);
//	float bb = (float)(m_BBox->value())/(m_BBox->maximum()-m_BBox->minimum()+1);

	int rr = m_RBox->value();
	int gg = m_GBox->value();
	int bb = m_BBox->value();
	
	// update HSV switches
	QColor temp = QColor::fromRgb(rr,gg,bb);
	int hh,ss,vv; 
	temp.getHsv(&hh,&ss,&vv); // 0~255 base
	//float factor = 0.3921; // 100/255
	m_HBox->setValue(hh);
	m_SBox->setValue(ss);
	m_VBox->setValue(vv);	
	
	//m_pColorPicker->updateColorPanelRGBf(rr,gg,bb);
}
