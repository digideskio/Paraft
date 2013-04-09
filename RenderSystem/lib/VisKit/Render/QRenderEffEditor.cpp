#include "QRenderEffEditor.h"
#include <QSlider>
#include <QCheckBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QTabWidget>
#include <QRadioButton>
#include <QPushButton>
#include <QComboBox>
#include <cmath>

#ifndef round
#define round(x) (floor(x + 0.5))
#endif

QDataStream & operator<<(QDataStream & out, const RenderEffect & re) {
	out << re.m_spacing << re.m_axisOptions << re.m_boundingBox << re.m_lighting
	    << re.m_light.Kamb << re.m_light.Kdif << re.m_light.Kspe << re.m_light.Kshi
	    << re.m_mouseTarget << re.m_slicers
	    << re.m_totalTimeStep << re.m_currentTimeStep;
	return out;
}
QDataStream & operator>>(QDataStream & in, RenderEffect & re) {
	in >> re.m_spacing >> re.m_axisOptions >> re.m_boundingBox >> re.m_lighting
	   >> re.m_light.Kamb >> re.m_light.Kdif >> re.m_light.Kspe >> re.m_light.Kshi
	   >> re.m_mouseTarget >> re.m_slicers
	   >> re.m_totalTimeStep >> re.m_currentTimeStep;
	return in;
}

QRenderEffEditor::QRenderEffEditor(QWidget *parent) : QTabWidget(parent) {
	setWindowFlags(Qt::Tool);
	setWindowTitle(tr("Render Effect"));
	this->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
	m_plightParam = new LIGHTPARAM;
	m_totalSteps = 1;
	m_updating = false;
	initLayout();
	initContent();
}

QRenderEffEditor::QRenderEffEditor(QWidget *parent,LIGHTPARAM *lp)
:QTabWidget(parent){
	setWindowFlags(Qt::Tool);
	setWindowTitle(tr("Render Effect"));
	this->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);

	m_plightParam = lp;
	m_totalSteps = 1;
	m_updating = false;
	initLayout();
	initContent();
}
void QRenderEffEditor::initLayout(){

	// General Tab

	QWidget * m_generalGroup = new QWidget();

	QVBoxLayout * generalLayout = new QVBoxLayout;
	generalLayout->setAlignment(Qt::AlignTop);	

	//
	QGridLayout * generalLayout1 = new QGridLayout;
	generalLayout1->setSpacing(2);
	generalLayout1->setAlignment(Qt::AlignTop);	

	generalLayout1->addWidget(new QLabel(tr("Spacing"),this),0,0);
	m_spacing = new QSlider(Qt::Horizontal,this);
	m_spacing->setTickInterval(1);
	m_spacing->setMaximum(1000);
	m_spacing->setMinimum(10);
	generalLayout1->addWidget(m_spacing,0,1);
	m_spacingLabel = new QLabel(this);
	m_spacingLabel->setFixedWidth(50);
	generalLayout1->addWidget(m_spacingLabel,0,2);
	connect(m_spacing,SIGNAL(valueChanged(int)),this,SLOT(setSliderSpacingValue(int)));
	m_spacing->setValue(100);

	//
	QGridLayout * generalLayout2 = new QGridLayout;
	generalLayout2->setSpacing(2);
	generalLayout2->setAlignment(Qt::AlignTop | Qt::AlignLeft);	

	m_mainAxis = new QCheckBox(tr("Main Axis"));
	m_mainAxis->setChecked(true);
	m_sideAxis = new QCheckBox(tr("Side Axis"));
	m_sideAxis->setChecked(true);
	m_boundingBox = new QCheckBox(tr("Bounding Box"));
	m_boundingBox->setChecked(true);
	generalLayout2->addWidget(new QLabel(tr("Show: ")),0,0);
	generalLayout2->addWidget(m_mainAxis,0,1);
	generalLayout2->addWidget(m_sideAxis,0,2);
	generalLayout2->addWidget(m_boundingBox,1,1,1,2);
	connect(m_mainAxis,SIGNAL(clicked(bool)),this,SLOT(setEnableMainAxis(bool)));
	connect(m_sideAxis,SIGNAL(clicked(bool)),this,SLOT(setEnableSideAxis(bool)));
	connect(m_boundingBox,SIGNAL(clicked(bool)),this,SLOT(setEnableBoundingBox(bool)));

	//
	generalLayout->addLayout(generalLayout1);
	generalLayout->addLayout(generalLayout2);

	m_generalGroup->setLayout(generalLayout);





	// Lighting Tab

	m_lightGroup = new QGroupBox(tr("Lighting"));
	m_lightGroup->setCheckable(true);
	m_lightGroup->setFlat(true);
	connect(m_lightGroup,SIGNAL(clicked(bool)),this,SLOT(setEnableLightGroup(bool)));

	QGridLayout *lightLayout = new QGridLayout;
	lightLayout->setSpacing(2);
	lightLayout->setAlignment(Qt::AlignTop);

	lightLayout->addWidget(new QLabel(tr("Ambient"),this),0,0);
	m_lightKa = new QSlider(Qt::Horizontal,this);
	m_lightKa->setTickInterval(1);	//m_lightka->setTickPosition(QSlider::TicksBelow);
	m_lightKa->setMaximum(100);
	m_lightKa->setMinimum(0);
	lightLayout->addWidget(m_lightKa,0,1);
	
	m_laLabel = new QLabel(this);
	m_laLabel->setFixedWidth(30);
	lightLayout->addWidget(m_laLabel,0,2);
	connect(m_lightKa,SIGNAL(valueChanged(int)),this,SLOT(setSliderAmbientValue(int)));
	m_lightKa->setValue(m_plightParam->Kamb*100);

	lightLayout->addWidget(new QLabel(tr("Diffuse"),this),1,0);
	m_lightKd = new QSlider(Qt::Horizontal,this);
	m_lightKd->setTickInterval(1);	//m_lightkd->setTickPosition(QSlider::TicksBelow);
	m_lightKd->setMaximum(100);
	m_lightKd->setMinimum(0);	
	lightLayout->addWidget(m_lightKd,1,1);

	m_ldLabel = new QLabel(this);
	lightLayout->addWidget(m_ldLabel,1,2);
	connect(m_lightKd,SIGNAL(valueChanged(int)),this,SLOT(setSliderDiffuseValue(int)));
	m_lightKd->setValue(m_plightParam->Kdif*100);	

	lightLayout->addWidget(new QLabel(tr("Specular"),this),2,0);
	m_lightKspec = new QSlider(Qt::Horizontal,this);
	m_lightKspec->setTickInterval(1);	//m_lightKspec->setTickPosition(QSlider::TicksBelow);
	m_lightKspec->setMaximum(100);
	m_lightKspec->setMinimum(0);	
	lightLayout->addWidget(m_lightKspec,2,1);

	m_lspLabel = new QLabel(this);
	lightLayout->addWidget(m_lspLabel,2,2);
	connect(m_lightKspec,SIGNAL(valueChanged(int)),this,SLOT(setSliderSpecularValue(int)));
	m_lightKspec->setValue(m_plightParam->Kspe*100);

	lightLayout->addWidget(new QLabel(tr("Shininess"),this),3,0);
	m_lightKShin = new QSlider(Qt::Horizontal,this);
	m_lightKShin->setTickInterval(1);	//m_lightShin->setTickPosition(QSlider::TicksBelow);
	m_lightKShin->setMaximum(12800);
	m_lightKShin->setMinimum(0);	
	lightLayout->addWidget(m_lightKShin,3,1);

	m_lshLabel = new QLabel(this);
	lightLayout->addWidget(m_lshLabel,3,2);
	connect(m_lightKShin,SIGNAL(valueChanged(int)),this,SLOT(setSliderShininessValue(int)));
	m_lightKShin->setValue(m_plightParam->Kshi*100);

	m_lightGroup->setLayout(lightLayout);
	m_lightGroup->setAlignment(Qt::AlignTop);



	// Slice Tab

//	m_sliceGroup = new QGroupBox(tr("Slicing"));
//	m_sliceGroup->setCheckable(true);
//	m_sliceGroup->setFlat(true);
//	m_sliceGroup->setChecked(false);
//	connect(m_sliceGroup,SIGNAL(clicked(bool)),this,SLOT(setEnableSliceGroup(bool)));
	QWidget * m_sliceGroup = new QWidget();

	QVBoxLayout * sliceLayout = new QVBoxLayout;
	sliceLayout->setAlignment(Qt::AlignTop);
	sliceLayout->setSpacing(0);

	//
	QGridLayout * sliceLayout1 = new QGridLayout;
	sliceLayout1->setSpacing(2);
	sliceLayout1->setAlignment(Qt::AlignTop | Qt::AlignLeft);

	sliceLayout1->addWidget(new QLabel(tr("Mouse: "), this), 0, 0);
	m_sliceMTCamera = new QRadioButton("camera");
	m_sliceMTCamera->setChecked(true);
	sliceLayout1->addWidget(m_sliceMTCamera, 0, 1);
	m_sliceMTTrack = new QRadioButton("track");
	sliceLayout1->addWidget(m_sliceMTTrack, 0, 2);
	m_sliceMTMove = new QRadioButton("move");
	sliceLayout1->addWidget(m_sliceMTMove, 0, 3);
	connect(m_sliceMTCamera,SIGNAL(clicked(bool)),this,SLOT(setSliceCameraClicked(bool)));
	connect(m_sliceMTTrack,SIGNAL(clicked(bool)),this,SLOT(setSliceTrackClicked(bool)));
	connect(m_sliceMTMove,SIGNAL(clicked(bool)),this,SLOT(setSliceMoveClicked(bool)));

	//
	QGridLayout * sliceLayout2 = new QGridLayout;
	sliceLayout2->setSpacing(2);
	sliceLayout2->setAlignment(Qt::AlignTop | Qt::AlignLeft);

	m_sliceList = new QComboBox();
	m_sliceListIdxInc = 0;
	sliceLayout2->addWidget(m_sliceList, 0, 0, 1, 3);
	QPushButton * addslice = new QPushButton("Add");
	sliceLayout2->addWidget(addslice, 1, 0);
	QPushButton * delslice = new QPushButton("Del");
	sliceLayout2->addWidget(delslice, 1, 1);
	m_sliceMasking = new QCheckBox(tr("Masking"));
	m_sliceMasking->setChecked(true);
	sliceLayout2->addWidget(m_sliceMasking, 1, 2);
	connect(addslice,SIGNAL(clicked(bool)),this,SLOT(setSliceAddClicked(bool)));
	connect(m_sliceList,SIGNAL(currentIndexChanged(int)),this,SLOT(setSliceIdxChanged(int)));
	connect(delslice,SIGNAL(clicked(bool)),this,SLOT(setSliceDelClicked(bool)));
	connect(m_sliceMasking,SIGNAL(clicked(bool)),this,SLOT(setSliceEnableMasking(bool)));

	//
	QGridLayout * sliceLayout3 = new QGridLayout;
	sliceLayout3->setSpacing(2);
	sliceLayout3->setAlignment(Qt::AlignTop);

	QPushButton * xp = new QPushButton("+x");
	xp->setFixedWidth(40);
	sliceLayout3->addWidget(xp, 0, 0);
	QPushButton * yp = new QPushButton("+y");
	yp->setFixedWidth(40);
	sliceLayout3->addWidget(yp, 0, 1);
	QPushButton * zp = new QPushButton("+z");
	zp->setFixedWidth(40);
	sliceLayout3->addWidget(zp, 0, 2);
	QPushButton * xn = new QPushButton("-x");
	xn->setFixedWidth(40);
	sliceLayout3->addWidget(xn, 0, 3);
	QPushButton * yn = new QPushButton("-y");
	yn->setFixedWidth(40);
	sliceLayout3->addWidget(yn, 0, 4);
	QPushButton * zn = new QPushButton("-z");
	zn->setFixedWidth(40);
	sliceLayout3->addWidget(zn, 0, 5);
	connect(xp,SIGNAL(clicked(bool)),this,SLOT(setSliceXPClicked(bool)));
	connect(yp,SIGNAL(clicked(bool)),this,SLOT(setSliceYPClicked(bool)));
	connect(zp,SIGNAL(clicked(bool)),this,SLOT(setSliceZPClicked(bool)));
	connect(xn,SIGNAL(clicked(bool)),this,SLOT(setSliceXNClicked(bool)));
	connect(yn,SIGNAL(clicked(bool)),this,SLOT(setSliceYNClicked(bool)));
	connect(zn,SIGNAL(clicked(bool)),this,SLOT(setSliceZNClicked(bool)));

	//
	QGridLayout * sliceLayout4 = new QGridLayout;
	sliceLayout4->setSpacing(2);
	sliceLayout4->setAlignment(Qt::AlignTop);

	sliceLayout4->addWidget(new QLabel(tr("Position: "),this), 0, 0);
	m_sliceDist = new QSlider(Qt::Horizontal,this);
	m_sliceDist->setTickInterval(1);
	m_sliceDist->setMaximum(2000);
	m_sliceDist->setMinimum(0);
	m_sliceDist->setValue(2000);
	sliceLayout4->addWidget(m_sliceDist, 0, 1);
	m_sliceDistLabel = new QLabel(tr("1.000"), this);
	m_sliceDistLabel->setFixedWidth(50);
	sliceLayout4->addWidget(m_sliceDistLabel, 0, 2);
	connect(m_sliceDist,SIGNAL(valueChanged(int)),this,SLOT(setSliceDistValue(int)));

	//
	sliceLayout->addLayout(sliceLayout1);
	sliceLayout->addLayout(sliceLayout2);
	sliceLayout->addLayout(sliceLayout3);
	sliceLayout->addLayout(sliceLayout4);

	m_sliceGroup->setLayout(sliceLayout);
//	m_sliceGroup->setAlignment(Qt::AlignTop);



	// Temporal
	QWidget * m_temporalGroup = new QWidget();

	//
	QGridLayout * temporalLayout = new QGridLayout;
	temporalLayout->setSpacing(2);
	temporalLayout->setAlignment(Qt::AlignTop);	

	temporalLayout->addWidget(new QLabel(tr("Timestep"),this),0,0);
	m_temporalTimestep = new QSlider(Qt::Horizontal,this);
	m_temporalTimestep->setTickInterval(1);
	m_temporalTimestep->setMaximum(m_totalSteps);
	m_temporalTimestep->setMinimum(1);
	temporalLayout->addWidget(m_temporalTimestep,0,1);
	m_temporalTimestepLabel = new QLabel(this);
	m_temporalTimestepLabel->setFixedWidth(50);
	temporalLayout->addWidget(m_temporalTimestepLabel,0,2);
	m_temporalTimestep->setValue(1);
	m_temporalTimestepLabel->setText(QString("1/%1").arg(m_totalSteps));
	connect(m_temporalTimestep,SIGNAL(valueChanged(int)),this,SLOT(setTemporalTimestepValue(int)));

	//
	m_temporalGroup->setLayout(temporalLayout);



	addTab(m_generalGroup, "General");
	addTab(m_lightGroup, "Light");
	addTab(m_sliceGroup, "Slice");
	addTab(m_temporalGroup, "Temporal");
}
void QRenderEffEditor::initContent(){

}
void QRenderEffEditor::setTotalSteps(int t){
	m_totalSteps = t;
	m_temporalTimestep->setMaximum(m_totalSteps);
	m_temporalTimestepLabel->setText(QString("%1/%2").arg(1).arg(m_totalSteps));
}
void QRenderEffEditor::loadSettings(const RenderEffect & re) {
	m_updating = true;

	m_spacing->setValue(round(re.m_spacing*100000.0));
	m_spacingLabel->setText(QString("%1").arg(re.m_spacing));
	if (re.m_axisOptions & 1) m_sideAxis->setChecked(true);
	else m_sideAxis->setChecked(false);
	if (re.m_axisOptions & 2) m_mainAxis->setChecked(true);
	else m_mainAxis->setChecked(false);
	if (re.m_boundingBox) m_boundingBox->setChecked(true);
	else m_boundingBox->setChecked(false);

	if (re.m_lighting) m_lightGroup->setChecked(true);
	else m_lightGroup->setChecked(false);
	m_lightKa->setValue(round(re.m_light.Kamb*100.0));
	m_lightKd->setValue(round(re.m_light.Kdif*100.0));
	m_lightKspec->setValue(round(re.m_light.Kspe*100.0));
	m_lightKShin->setValue(round(re.m_light.Kshi*100.0));
	m_laLabel->setText(QString("%1").arg(re.m_light.Kamb));
	m_ldLabel->setText(QString("%1").arg(re.m_light.Kdif));
	m_lspLabel->setText(QString("%1").arg(re.m_light.Kspe));
	m_lshLabel->setText(QString("%1").arg(re.m_light.Kshi));

	//if (re.m_slice) m_sliceGroup->setChecked(true);
	//else m_sliceGroup->setChecked(false);
	if (re.m_mouseTarget == 0) m_sliceMTCamera->setChecked(true);
	else if (re.m_mouseTarget == 1) m_sliceMTTrack->setChecked(true);
	else if (re.m_mouseTarget == 2) m_sliceMTMove->setChecked(true);
	m_sliceList->clear(); // QComboBox
	m_sliceListDist.clear(); // QList for dist
	m_sliceListMasking.clear(); // QList for masking
	m_sliceListIdxInc = 0;
	for (int i = 0; i < re.m_slicers.size(); ++i) {
		int dist = round((re.m_slicers[i].getDist()+1.0)*1000.0);
		m_sliceList->addItem(QString("Slice%1").arg(m_sliceListIdxInc++));
		m_sliceListDist.push_back(dist);
		m_sliceListMasking.push_back(re.m_slicers[i].isMasking());
	}
	if (m_sliceListIdxInc > 0) {
		m_sliceDist->setValue(m_sliceListDist[m_sliceListIdxInc-1]);
		m_sliceDistLabel->setText(QString("%1").arg(re.m_slicers[m_sliceListIdxInc-1].getDist()));
		if (m_sliceListMasking[m_sliceListIdxInc-1]) m_sliceMasking->setChecked(true);
		else m_sliceMasking->setChecked(false);
		m_sliceList->setCurrentIndex(m_sliceListIdxInc-1);
	}

	m_totalSteps = re.m_totalTimeStep;
	m_temporalTimestep->setMaximum(m_totalSteps);
	m_temporalTimestep->setValue(re.m_currentTimeStep);
	m_temporalTimestepLabel->setText(QString("%1/%2").arg(re.m_currentTimeStep).arg(m_totalSteps));

	m_updating = false;
}

//
void QRenderEffEditor::setSliderSpacingValue(int idx){
	if (m_updating) return;

	m_spacingLabel->setText(QString("%1").arg((double)idx/100000.0));
	emit spacingValueChanged((float)idx/100000.0);
}
void QRenderEffEditor::setEnableMainAxis(bool idx){
	if (m_updating) return;

	emit mainAxisChanged(idx);
}
void QRenderEffEditor::setEnableSideAxis(bool idx){
	if (m_updating) return;

	emit sideAxisChanged(idx);
}
void QRenderEffEditor::setEnableBoundingBox(bool idx){
	if (m_updating) return;

	emit boundingBoxChanged(idx);
}

//
void QRenderEffEditor::setEnableLightGroup(bool idx){
	if (m_updating) return;

	emit lightGroupChanged(idx);
}
void QRenderEffEditor::setSliderAmbientValue(int idx){
	if (m_updating) return;

	m_laLabel->setText(QString("%1").arg((double)idx/100.0));
	emit ambientValueChanged((float)idx/100.0);
}
void QRenderEffEditor::setSliderDiffuseValue(int idx){
	if (m_updating) return;

	m_ldLabel->setText(QString("%1").arg((float)idx/100.0));
	emit diffuseValueChanged((float)idx/100.0);
}
void QRenderEffEditor::setSliderSpecularValue(int idx){
	if (m_updating) return;

	m_lspLabel->setText(QString("%1").arg((float)idx/100.0));
	emit specularValueChanged((float)idx/100.0);
}
void QRenderEffEditor::setSliderShininessValue(int idx){
	if (m_updating) return;

	m_lshLabel->setText(QString("%1").arg((float)idx/100.0));
	emit shininessValueChanged((float)idx/100.0);
}

//
//void QRenderEffEditor::setEnableSliceGroup(bool idx){
//	if (m_updating) return;
//
//	emit sliceGroupChanged(idx);
//}
void QRenderEffEditor::setSliceCameraClicked(bool){
	if (m_updating) return;

	emit sliceMouseChanged(0);
}
void QRenderEffEditor::setSliceTrackClicked(bool){
	if (m_updating) return;

	emit sliceMouseChanged(1);
}
void QRenderEffEditor::setSliceMoveClicked(bool){
	if (m_updating) return;

	emit sliceMouseChanged(2);
}
void QRenderEffEditor::setSliceAddClicked(bool) {
	emit sliceAddChanged();
	m_sliceListDist.push_back(1000);
	m_sliceListMasking.push_back(true);
	m_sliceList->addItem(QString("Slice%1").arg(m_sliceListIdxInc++));
	m_sliceList->setCurrentIndex(m_sliceList->count()-1);
}
void QRenderEffEditor::setSliceIdxChanged(int idx) {
	if (m_updating) return;

	m_updating = true;
	emit sliceSelChanged(idx);
	if (idx >= 0) {
		m_sliceDistLabel->setText(QString("%1").arg((float)m_sliceListDist[idx]/1000.0-1.0));
		m_sliceDist->setValue(m_sliceListDist[idx]);
		m_sliceMasking->setChecked(m_sliceListMasking[idx]);
	}
	m_updating = false;
}
void QRenderEffEditor::setSliceDelClicked(bool) {
	int idx = m_sliceList->currentIndex();
	if (idx >= 0) {
		m_sliceList->removeItem(idx);
		m_sliceListDist.removeAt(idx);
		m_sliceListMasking.removeAt(idx);
		emit sliceDelChanged(idx);
		if (idx == m_sliceList->count()) idx--;
		if (idx >= 0) {
			m_sliceDistLabel->setText(QString("%1").arg((float)m_sliceListDist[idx]/1000.0-1.0));
			m_sliceDist->setValue(m_sliceListDist[idx]);
			m_sliceMasking->setChecked(m_sliceListMasking[idx]);
		}
	}
}
void QRenderEffEditor::setSliceXPClicked(bool){
	emit sliceVectorChanged(Vector3(1.0,0.0,0.0));
}
void QRenderEffEditor::setSliceXNClicked(bool ){
	emit sliceVectorChanged(Vector3(-1.0,0.0,0.0));
}
void QRenderEffEditor::setSliceYPClicked(bool ){
	emit sliceVectorChanged(Vector3(0.0,1.0,0.0));
}
void QRenderEffEditor::setSliceYNClicked(bool ){
	emit sliceVectorChanged(Vector3(0.0,-1.0,0.0));
}
void QRenderEffEditor::setSliceZPClicked(bool ){
	emit sliceVectorChanged(Vector3(0.0,0.0,1.0));
}
void QRenderEffEditor::setSliceZNClicked(bool ){
	emit sliceVectorChanged(Vector3(0.0,0.0,-1.0));
}
void QRenderEffEditor::setSliceDistValue(int idx){
	if (m_updating) return;

	if (m_sliceList->currentIndex() >= 0) m_sliceListDist[m_sliceList->currentIndex()] = idx;
	m_sliceDistLabel->setText(QString("%1").arg((float)idx/1000.0-1.0));
	emit sliceDistChanged((double)idx/1000.0-1.0);
}
void QRenderEffEditor::setSliceDistValue(double idx){
	if (m_sliceList->currentIndex() >= 0) m_sliceListDist[m_sliceList->currentIndex()] = round((idx+1.0)*1000.0);
	m_sliceDistLabel->setText(QString("%1").arg(idx));
	m_sliceDist->setValue(round((idx+1.0)*1000.0));
}
void QRenderEffEditor::setSliceEnableMasking(bool idx){
	if (m_updating) return;

	if (m_sliceList->currentIndex() >= 0) m_sliceListMasking[m_sliceList->currentIndex()] = idx;
	emit sliceMaskingChanged(idx);
}
void QRenderEffEditor::setSliceExternalMasking(bool idx){
	if (m_sliceList->currentIndex() >= 0) m_sliceListMasking[m_sliceList->currentIndex()] = idx;
	m_sliceMasking->setChecked(idx);
}
void QRenderEffEditor::setTemporalTimestepValue(int idx){
	if (m_updating) return;

	m_temporalTimestepLabel->setText(QString("%1/%2").arg(idx).arg(m_totalSteps));
	emit timestepValueChanged(idx);
}
void QRenderEffEditor::timestepValueExternalChange(int idx){
	m_temporalTimestep->setValue(idx);
	m_temporalTimestepLabel->setText(QString("%1/%2").arg(idx).arg(m_totalSteps));
}
