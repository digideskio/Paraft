#ifndef _QRENDEREFFEDITOR_H_
#define _QRENDEREFFEDITOR_H_

#include <QWidget>
#include <QTabWidget>
#include "vectors.h"
#include "slicer.h"
#include "LIGHTPARAM.h"
using namespace std;

class QSlider;
class QCheckBox;
class QGroupBox;
class QLabel;
class QRadioButton;
class QComboBox;

struct RenderEffect {
	RenderEffect() {
        m_spacing = 0.001f;
		m_axisOptions = 3;
		m_boundingBox = true;
		m_lighting = true;
		m_mouseTarget = 0;
		m_slicers.clear();
		m_totalTimeStep = 1;
		m_currentTimeStep = 1; 
		m_slice = false; // dummy
		m_sliceDist = 0.0; // dummy
		m_sliceMasking = false; //dummy
	}
	RenderEffect(const RenderEffect & re) {
		(*this) = re;
	}
	RenderEffect & operator=(const RenderEffect & src) {
		this->m_spacing = src.m_spacing;
		this->m_axisOptions = src.m_axisOptions;
		this->m_boundingBox = src.m_boundingBox;
		this->m_lighting = src.m_lighting;
		this->m_light = src.m_light;
		this->m_mouseTarget = src.m_mouseTarget;
		this->m_slicers = src.m_slicers;
		this->m_totalTimeStep = src.m_totalTimeStep;
		this->m_currentTimeStep = src.m_currentTimeStep;
		this->m_slice = src.m_slice; //dummy
		this->m_sliceDist = src.m_sliceDist; //dummy
		this->m_sliceMasking = src.m_sliceMasking; //dummy
		return *this;
	}
	// General
	float m_spacing;
	int m_axisOptions;
	bool m_boundingBox;

	// Lighting
	bool m_lighting;
	LIGHTPARAM m_light;

	// Clipping
	int m_mouseTarget;
	QList<Slicer> m_slicers;
	bool m_slice; // dummy object, to be compatible with single clipping version
	double m_sliceDist; // dummy object, to be compatible with single clipping version
	bool m_sliceMasking; // dummy object, to be compatible with single clipping version

	// Temporal
	int m_totalTimeStep;
	int m_currentTimeStep; 
};
QDataStream & operator<<(QDataStream &, const RenderEffect &);
QDataStream & operator>>(QDataStream &, RenderEffect &);

class QRenderEffEditor : public QTabWidget{
	Q_OBJECT
public:
	QRenderEffEditor(QWidget * = 0);
	QRenderEffEditor(QWidget *parent, LIGHTPARAM* lp);
	void setTotalSteps(int);
	void loadSettings(const RenderEffect &);

protected:
	QSlider		*m_spacing;
	QLabel		*m_spacingLabel;
	float		*m_pSpacingParam;
	QCheckBox	*m_mainAxis;
	QCheckBox	*m_sideAxis;
	QCheckBox	*m_boundingBox;

	LIGHTPARAM	*m_plightParam;
	QGroupBox	*m_lightGroup;
	//		Ambient     diffuse     specular       shininess
	QSlider		*m_lightKa,*m_lightKd,*m_lightKspec,*m_lightKShin; 
	QLabel 		*m_laLabel,*m_ldLabel,*m_lspLabel,*m_lshLabel;

	QRadioButton	*m_sliceMTCamera;
	QRadioButton	*m_sliceMTTrack;
	QRadioButton	*m_sliceMTMove;
	QComboBox	*m_sliceList;
	int		m_sliceListIdxInc;
	QList<int>	m_sliceListDist;
	QList<bool>	m_sliceListMasking;
	QSlider		*m_sliceDist;
	QLabel		*m_sliceDistLabel;
	QCheckBox	*m_sliceMasking;

	int		m_totalSteps;
	QLabel		*m_temporalTimestepLabel;
	QSlider		*m_temporalTimestep;

	void		initLayout();
	void		initContent();
	bool		m_updating;

public slots:
	void		setSliderSpacingValue(int);
	void		setEnableMainAxis(bool);
	void		setEnableSideAxis(bool);
	void		setEnableBoundingBox(bool);

	void		setEnableLightGroup(bool);
	void		setSliderAmbientValue(int);
	void		setSliderDiffuseValue(int);
	void		setSliderSpecularValue(int);
	void		setSliderShininessValue(int);

//	void		setEnableSliceGroup(bool);
	void		setSliceCameraClicked(bool);
	void		setSliceTrackClicked(bool);
	void		setSliceMoveClicked(bool);
	void		setSliceAddClicked(bool);
	void		setSliceIdxChanged(int);
	void		setSliceDelClicked(bool);
	void		setSliceXPClicked(bool);
	void		setSliceXNClicked(bool);
	void		setSliceYPClicked(bool);
	void		setSliceYNClicked(bool);
	void		setSliceZPClicked(bool);
	void		setSliceZNClicked(bool);
	void		setSliceDistValue(int);
	void		setSliceDistValue(double);
	void		setSliceEnableMasking(bool);
	void		setSliceExternalMasking(bool);

	void		setTemporalTimestepValue(int);
	void		timestepValueExternalChange(int);

public:
signals:
	void		spacingValueChanged(float);
	void		mainAxisChanged(bool);
	void		sideAxisChanged(bool);
	void		boundingBoxChanged(bool);

	void		lightGroupChanged(bool);
	void		ambientValueChanged(float);
	void		diffuseValueChanged(float);
	void		specularValueChanged(float);
	void		shininessValueChanged(float);

	void		sliceAddChanged();
	void		sliceDelChanged(int);
	void		sliceSelChanged(int);
	void		sliceMouseChanged(int);
	void		sliceVectorChanged(Vector3);
	void		sliceDistChanged(double);
	void		sliceMaskingChanged(bool);

	void		timestepValueChanged(int);
};

#endif
