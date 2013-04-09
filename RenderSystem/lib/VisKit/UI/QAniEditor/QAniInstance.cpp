#include "QAniInstance.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif
#ifndef round
#define round(x) (floor(x + 0.5))
#endif

QAniInstance::QAniInstance(Type type, QAniKeyframe* start, QAniKeyframe* end) : m_type(type), m_start(start), m_end(end), m_tex(-1) {
	if (m_start) m_start->setRightInstance(this);
	if (m_end) m_end->setLeftInstance(this);
}

QAniInstance::QAniInstance(QAniInstance & instance) {
	m_type = instance.m_type;
	m_start = instance.m_start;
	m_end = instance.m_end;
	m_tex = instance.m_tex;
}

QAniInstance::~QAniInstance() {
	if (m_start) m_start->setRightInstance(NULL);
	if (m_end) m_end->setLeftInstance(NULL);
}

void QAniInstance::setStart(QAniKeyframe * keyframe, bool setKeyframe) {
	m_start = keyframe;
	if (setKeyframe) m_start->setRightInstance(this);
}

void QAniInstance::setEnd(QAniKeyframe * keyframe, bool setKeyframe) {
	m_end = keyframe;
	if (setKeyframe) m_end->setLeftInstance(this);
}

void QAniInstance::generateInterpolatedFrame(QAniKeyframe * keyframe) {
	switch (keyframe->getType()) {
	case Camera:
		((QAniCameraInstance*)this)->generateInterpolatedFrame(keyframe);
		break;
	case Temporal:
		((QAniTemporalInstance*)this)->generateInterpolatedFrame(keyframe);
		break;
	case TransferFunction:
		((QAniTransferFunctionInstance*)this)->generateInterpolatedFrame(keyframe);
		break;
	case Slice:
		((QAniSliceInstance*)this)->generateInterpolatedFrame(keyframe);
		break;
	}
}

bool QAniInstance::operator==(const QAniInstance & instance) const {
	switch (instance.getType()) {
	case Camera:
		return *((QAniCameraInstance*)this) == *((QAniCameraInstance*)&instance);
		break;
	case Temporal:
		return *((QAniTemporalInstance*)this) == *((QAniTemporalInstance*)&instance);
		break;
	case TransferFunction:
		return *((QAniTransferFunctionInstance*)this) == *((QAniTransferFunctionInstance*)&instance);
		break;
	case Slice:
		return *((QAniSliceInstance*)this) == *((QAniSliceInstance*)&instance);
		break;
	default:
		return false;
	}
}

void QAniInstance::cloneInstance(QAniInstance*& instance) {
	if (instance) delete instance;
	switch (m_type) {
	case Camera:
		instance = new QAniCameraInstance(*(QAniCameraInstance*)this);
		break;
	case Temporal:
		instance = new QAniTemporalInstance(*(QAniTemporalInstance*)this);
		break;
	case TransferFunction:
		instance = new QAniTransferFunctionInstance(*(QAniTransferFunctionInstance*)this);
		break;
	case Slice:
		instance = new QAniSliceInstance(*(QAniSliceInstance*)this);
		break;
	}
}

void QAniInstance::save(QFile & file) const {
	file.write((char *)&m_type, 4);
	file.write((char *)&m_tex,4);
	switch (m_type) {
	case Camera:
		((QAniCameraInstance*)this)->save(file);
		break;
	case Temporal:
		((QAniTemporalInstance*)this)->save(file);
		break;
	case TransferFunction:
		((QAniTransferFunctionInstance*)this)->save(file);
		break;
	case Slice:
		((QAniSliceInstance*)this)->save(file);
		break;
	}
}

QAniInstance * QAniInstance::load(QFile & file) {
	QAniInstance * instance;
	Type type;
	GLint tex;
	file.read((char *)&type, 4);
	file.read((char *)&tex, 4);

	switch (type) {
	case Camera:
		instance = new QAniCameraInstance(file);
		break;
	case Temporal:
		instance = new QAniTemporalInstance(file);
		break;
	case TransferFunction:
		instance = new QAniTransferFunctionInstance(file);
		break;
	case Slice:
		instance = new QAniSliceInstance(file);
		break;
	}
	instance->setTexture(tex);
	return instance;
}

/* class QAniCameraInstance
 *
 *
 */
// This is for Spcaial Overview
QAniCameraInstance::QAniCameraInstance(QFile & file) : QAniInstance(QAniInstance::Camera) {
	load(file);	
}

QAniCameraInstance::QAniCameraInstance(double radians, Vector3 axis) : QAniInstance(QAniInstance::Camera) {
	m_lookTrans = Vector3(0.0,0.0,0.0);

	m_trackballRadians = radians;
	m_trackballAxis = axis;

	m_tiltRadians = 0.0;
	m_tiltAxis = Vector3(0.0,0.0,0.0);

	m_zoom = 0.0;
}

QAniCameraInstance::QAniCameraInstance(QAniKeyframe* start, QAniKeyframe* end) : QAniInstance(QAniInstance::Camera, start, end) {
	CameraOptions co1 = ((QAniCameraKeyframe*)start)->getOptions();
	CameraOptions co2 = ((QAniCameraKeyframe*)end)->getOptions();

	// LookTrans
	m_lookTrans = co2.l - co1.l;

	// Trackball
	m_trackballRadians = acos(co1.c.dot(co2.c));
	if (m_trackballRadians == 0) m_trackballAxis = co1.u;
	else if (m_trackballRadians == M_PI) m_trackballAxis = co1.u;
	else m_trackballAxis = co1.c * co2.c;

	// Tilt
	// first rotate co1.u by the trackball axis and radians, and compare between co1.u* and co2.u to ??derive the tilt rotation
	m_tiltRadians = acos(co1.u.dot(co2.u));
	if (m_tiltRadians == 0) m_tiltAxis = co1.c;
	else if (m_tiltRadians == M_PI) m_tiltAxis = co1.c;
	else m_tiltAxis = co1.u * co2.u;

	// Zoom
	m_zoom = co2.dist - co1.dist;
}

QAniCameraInstance::QAniCameraInstance(QAniCameraInstance & instance) : QAniInstance(instance) {
	m_lookTrans = instance.m_lookTrans;
	m_trackballRadians = instance.m_trackballRadians;
	m_trackballAxis = instance.m_trackballAxis;
	m_tiltRadians = instance.m_tiltRadians;
	m_tiltAxis = instance.m_tiltAxis;
	m_zoom = instance.m_zoom;
}

QAniCameraInstance::~QAniCameraInstance() {
}

QAniCameraKeyframe * QAniCameraInstance::generateKeyframe(QAniCameraKeyframe * start, float duration) {
	QAniCameraKeyframe * end = new QAniCameraKeyframe(start->getTime() + duration);
	(*end) = (*start);
	CameraOptions co1 = start->getOptions();
	CameraOptions& co2 = end->getOptions();

	co2.l = co1.l + m_lookTrans;

	if (m_trackballRadians != 0.0) co2.c = Matrix4x4::fromRotation(m_trackballAxis, m_trackballRadians) * co1.c;
	else co2.c = co1.c;

	if (m_tiltRadians != 0.0) {
		co2.u = Matrix4x4::fromRotation(m_tiltAxis, m_tiltRadians) * co1.u;
		co2.o = Matrix4x4::fromRotation(m_tiltAxis, m_tiltRadians) * co1.o;
	}
	else {
		co2.u = co1.u;
		co2.o = co1.o;
	}

	co2.dist = co1.dist + m_zoom;

	return end;
}

void QAniCameraInstance::generateInterpolatedFrame(QAniKeyframe * frame) {
	float itp = (frame->getTime() - getStartTime()) / getDuration();
	CameraOptions co1 = ((QAniCameraKeyframe*)m_start)->getOptions();
	CameraOptions co2 = ((QAniCameraKeyframe*)m_end)->getOptions();
	CameraOptions & co = ((QAniCameraKeyframe*)frame)->getOptions();

	co.l = co1.l + m_lookTrans * itp;

	if (m_trackballRadians != 0.0) co.c = Matrix4x4::fromRotation(m_trackballAxis, m_trackballRadians * itp) * co1.c;
	else co.c = co1.c;

	if (m_tiltRadians != 0.0) {
		co.u = Matrix4x4::fromRotation(m_tiltAxis, m_tiltRadians * itp) * co1.u;
		co.o = Matrix4x4::fromRotation(m_tiltAxis, m_tiltRadians * itp) * co1.o;
	}
	else {
		co.u = co1.u;
		co.o = co1.o;
	}

	co.dist = co1.dist + m_zoom * itp;

	co.nearclip = co1.nearclip;
	co.farclip = co1.farclip;
	co.fov = co1.fov;
	co.maxDist = co1.maxDist;
}

bool QAniCameraInstance::operator==(const QAniCameraInstance & instance) const {
	if (!(m_lookTrans == instance.m_lookTrans)) return false;
	if (!(m_trackballRadians == instance.m_trackballRadians)) return false;
	if (!(m_trackballAxis == instance.m_trackballAxis)) return false;
	if (!(m_tiltRadians == instance.m_tiltRadians)) return false;
	if (!(m_tiltAxis == instance.m_tiltAxis)) return false;
	if (!(m_zoom == instance.m_zoom)) return false;
	return true;
}

void QAniCameraInstance::save(QFile & file) const {
	double x,y,z;

	x = m_lookTrans.x();
	y = m_lookTrans.y();
	z = m_lookTrans.z();
	file.write((char *)&x, 8);
	file.write((char *)&y, 8);
	file.write((char *)&z, 8);

	file.write((char *)&m_trackballRadians, 8);

	x = m_trackballAxis.x();
	y = m_trackballAxis.y();
	z = m_trackballAxis.z();
	file.write((char *)&x, 8);
	file.write((char *)&y, 8);
	file.write((char *)&z, 8);

	file.write((char *)&m_tiltRadians, 8);

	x = m_tiltAxis.x();
	y = m_tiltAxis.y();
	z = m_tiltAxis.z();
	file.write((char *)&x, 8);
	file.write((char *)&y, 8);
	file.write((char *)&z, 8);

	file.write((char *)&m_zoom, 8);
}

void QAniCameraInstance::load(QFile & file) {
	file.read((char *)((double *)m_lookTrans), 24);
	file.read((char *)&m_trackballRadians, 8);
	file.read((char *)((double *)m_trackballAxis), 24);
	file.read((char *)&m_tiltRadians, 8);
	file.read((char *)((double *)m_tiltAxis), 24);
	file.read((char *)&m_zoom, 8);
}



/* QAniTemporalInstance
 *
 *
 */
QAniTemporalInstance::QAniTemporalInstance(QFile & file) : QAniInstance(QAniInstance::Temporal) {
	load(file);
}

QAniTemporalInstance::QAniTemporalInstance(size_t t) : QAniInstance(QAniInstance::Temporal), m_timeStepChange(static_cast<int>(t)) {
}

QAniTemporalInstance::QAniTemporalInstance(QAniKeyframe* start, QAniKeyframe* end) : QAniInstance(QAniInstance::Temporal, start, end) {
	m_timeStepChange = (int)(((QAniTemporalKeyframe*)end)->getTimeStep()) - (int)(((QAniTemporalKeyframe*)start)->getTimeStep());
}

QAniTemporalInstance::QAniTemporalInstance(QAniTemporalInstance & instance) : QAniInstance(instance) {
	m_timeStepChange = instance.m_timeStepChange;
}

QAniTemporalInstance::~QAniTemporalInstance() {
}

void QAniTemporalInstance::generateInterpolatedFrame(QAniKeyframe * frame) {
	float itp = (frame->getTime() - getStartTime()) / getDuration();
	((QAniTemporalKeyframe*)frame)->setTimeStep(round((float)((QAniTemporalKeyframe*)m_start)->getTimeStep() + (float)m_timeStepChange * itp));
}

bool QAniTemporalInstance::operator==(const QAniTemporalInstance & instance) const {
	if (m_timeStepChange != instance.m_timeStepChange) return false;
	return true;
}

void QAniTemporalInstance::save(QFile & file) const {
	file.write((char *)&m_timeStepChange, 4);
}

void QAniTemporalInstance::load(QFile & file) {
	file.read((char *)&m_timeStepChange, 4);
}


/* QAniTransferFunctionInstance
 *
 *
 */
QAniTransferFunctionInstance::QAniTransferFunctionInstance(QFile & file) : QAniInstance(QAniInstance::TransferFunction) {
	load(file);
}

QAniTransferFunctionInstance::QAniTransferFunctionInstance(QAniKeyframe* s, QAniKeyframe* e) : QAniInstance(QAniInstance::TransferFunction, s, e) {
	QAniTransferFunctionKeyframe * start = (QAniTransferFunctionKeyframe*)s;
	QAniTransferFunctionKeyframe * end = (QAniTransferFunctionKeyframe*)e;
	m_resolution = start->getResolution();
	m_tfResolutionChange = end->getResolution() - start->getResolution();
	if (m_tfResolutionChange == 0) {
		m_tfDrawArrayChange = new float[start->getResolution()];
		for (int i = 0; i < start->getResolution(); ++i) {
			m_tfDrawArrayChange[i] = end->getDrawArray()[i] - start->getDrawArray()[i];
		}
	}
	else m_tfDrawArrayChange = NULL;

	int idx = (start->getGaussians()->size() < end->getGaussians()->size() ?
			   start->getGaussians()->size() : end->getGaussians()->size());
	m_gaussianObjectChange.clear();
	for (int i = 0; i < idx; ++i) {
		GaussianObjectChange goc((*end->getGaussians())[i].m_mean - (*start->getGaussians())[i].m_mean,
					 (*end->getGaussians())[i].m_sigma - (*start->getGaussians())[i].m_sigma,
					 (*end->getGaussians())[i].m_heightFactor - (*start->getGaussians())[i].m_heightFactor);
		m_gaussianObjectChange.push_back(goc);
	}
	idx = (start->getColors()->size() < end->getColors()->size() ?
		   start->getColors()->size() : end->getColors()->size());
	m_tfColorTickChange.clear();
	for (int i = 0; i < idx; ++i) {
		Vector4 color((*end->getColors())[i].m_color.redF() - (*start->getColors())[i].m_color.redF(),
			      (*end->getColors())[i].m_color.greenF() - (*start->getColors())[i].m_color.greenF(),
			      (*end->getColors())[i].m_color.blueF() - (*start->getColors())[i].m_color.blueF(),
			      (*end->getColors())[i].m_color.alphaF() - (*start->getColors())[i].m_color.alphaF());
		ColorTickChange ctc((*end->getColors())[i].m_resX - (*start->getColors())[i].m_resX, color);
		m_tfColorTickChange.push_back(ctc);
	}
}

QAniTransferFunctionInstance::QAniTransferFunctionInstance(QAniTransferFunctionInstance & instance) : QAniInstance(instance) {
	m_resolution = instance.m_resolution;
	m_tfResolutionChange = instance.m_tfResolutionChange;
	m_tfDrawArrayChange = new float[m_resolution];
	for (int i = 0; i < m_resolution; ++i) {
		m_tfDrawArrayChange[i] = instance.m_tfDrawArrayChange[i];
	}
	m_gaussianObjectChange = instance.m_gaussianObjectChange;
	m_tfColorTickChange = instance.m_tfColorTickChange;
}

QAniTransferFunctionInstance::~QAniTransferFunctionInstance() {
	if (m_tfDrawArrayChange) delete [] m_tfDrawArrayChange;
}

void QAniTransferFunctionInstance::generateInterpolatedFrame(QAniKeyframe * f) {
	float itp = (f->getTime() - getStartTime()) / getDuration();
	QAniTransferFunctionKeyframe * start = (QAniTransferFunctionKeyframe*)m_start;
	QAniTransferFunctionKeyframe * end = (QAniTransferFunctionKeyframe*)m_end;
	QAniTransferFunctionKeyframe * frame = (QAniTransferFunctionKeyframe*)f;

	
	if (m_tfResolutionChange != 0) {
		// cannot interpolate two different resolution transfer functions
		return;
	}
	int resolution = start->getResolution();
	float * itpDrawArray = new float[resolution];
	for (int i = 0; i < resolution; ++i) {
		itpDrawArray[i] = start->getDrawArray()[i] + m_tfDrawArrayChange[i] * itp;
	}
	QVector<GaussianObject> gaussians;
	gaussians.clear();
	int idx = (start->getGaussians()->size()<end->getGaussians()->size()?start->getGaussians()->size():end->getGaussians()->size());
	for (int i = 0; i < idx; ++i) {
		GaussianObject go((*(start->getGaussians()))[i].m_mean + m_gaussianObjectChange[i].meanChange * (double)itp,
				  (*(start->getGaussians()))[i].m_sigma + m_gaussianObjectChange[i].sigmaChange * (double)itp,
				  (*(start->getGaussians()))[i].m_heightFactor + m_gaussianObjectChange[i].heightChange * (double)itp,
				  resolution);
		gaussians.push_back(go);
	}
	if (start->getGaussians()->size() > idx) {
		for (int i = idx; i < start->getGaussians()->size(); ++i) {
			GaussianObject go((*(start->getGaussians()))[i].m_mean,
					  (*(start->getGaussians()))[i].m_sigma,
					  (*(start->getGaussians()))[i].m_heightFactor * (1.0 - (double)itp),
					  resolution);
			gaussians.push_back(go);
		}
	}
	else if (end->getGaussians()->size() > idx) {
		for (int i = idx; i < end->getGaussians()->size(); ++i) {
			GaussianObject go((*(end->getGaussians()))[i].m_mean,
					  (*(end->getGaussians()))[i].m_sigma,
					  (*(end->getGaussians()))[i].m_heightFactor * (double)itp,
					  resolution);
			gaussians.push_back(go);
		}
	}
	QVector<TFColorTick> colors;
	colors.clear();
	idx = (start->getColors()->size()<end->getColors()->size()?start->getColors()->size():end->getColors()->size());
	for (int i = 0; i < idx; ++i) {
		QColor color;
		color.setRgbF((*(start->getColors()))[i].m_color.redF() + m_tfColorTickChange[i].colorChange.x() * (double)itp,
			      (*(start->getColors()))[i].m_color.greenF() + m_tfColorTickChange[i].colorChange.y() * (double)itp,
			      (*(start->getColors()))[i].m_color.blueF() + m_tfColorTickChange[i].colorChange.z() * (double)itp,
			      (*(start->getColors()))[i].m_color.alphaF() + m_tfColorTickChange[i].colorChange.d() * (double)itp);
		TFColorTick ct((*(start->getColors()))[i].m_resX + m_tfColorTickChange[i].XChange * itp, color);
		colors.push_back(ct);
	}
	if (start->getColors()->size() > idx) {
		// TODO interpolate from start tick and end interpolated color 
	}
	else if (end->getColors()->size() > idx) {
		// TODO interpolate from start interpolated color and end tick color
	}
	frame->setTransferFunction((float)resolution, itpDrawArray, &gaussians, &colors);
}

bool QAniTransferFunctionInstance::operator==(const QAniTransferFunctionInstance & instance) const {
	if (!(m_resolution == instance.m_resolution)) return false;
	if (!(m_tfResolutionChange == instance.m_tfResolutionChange)) return false;
	for (int i = 0; i < m_resolution; ++i) {
		if (m_tfDrawArrayChange[i] != instance.m_tfDrawArrayChange[i]) return false;
	}
	if (!(m_gaussianObjectChange == instance.m_gaussianObjectChange)) return false;
	if (!(m_tfColorTickChange == instance.m_tfColorTickChange)) return false;
	return true;
}

void QAniTransferFunctionInstance::save(QFile & file) const {
	file.write((char *)&m_resolution, 4);
	file.write((char *)&m_tfResolutionChange, 4);
	file.write((char *)m_tfDrawArrayChange, 4 * m_resolution);
	int size = m_gaussianObjectChange.size();
	file.write((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		file.write((char *)&(m_gaussianObjectChange[i].meanChange), 8);
		file.write((char *)&(m_gaussianObjectChange[i].sigmaChange), 8);
		file.write((char *)&(m_gaussianObjectChange[i].heightChange), 8);
	}
	size = m_tfColorTickChange.size();
	file.write((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		double r,g,b,a;
		r = m_tfColorTickChange[i].colorChange.x();
		g = m_tfColorTickChange[i].colorChange.y();
		b = m_tfColorTickChange[i].colorChange.z();
		a = m_tfColorTickChange[i].colorChange.d();
		file.write((char *)&(m_tfColorTickChange[i].XChange), 4);
		file.write((char *)&r, 8);
		file.write((char *)&g, 8);
		file.write((char *)&b, 8);
		file.write((char *)&a, 8);
	}
}

void QAniTransferFunctionInstance::load(QFile & file) {
	file.read((char *)&m_resolution, 4);
	file.read((char *)&m_tfResolutionChange, 4);
	m_tfDrawArrayChange = new float[m_resolution];
	file.read((char *)m_tfDrawArrayChange, 4 * m_resolution);

	int size;
	file.read((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		double mean,sigma,height;
		file.read((char *)&mean, 8);
		file.read((char *)&sigma, 8);
		file.read((char *)&height, 8);
		GaussianObjectChange goc(mean, sigma, height);
		m_gaussianObjectChange.push_back(goc);
	}
	file.read((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		float x;
		Vector4 color;
		file.read((char *)&x, 4);
		file.read((char *)((double*)color), 32);
		ColorTickChange ctc(x, color);
		m_tfColorTickChange.push_back(ctc);
	}
}


/* QAniSliceInstance
 *
 *
 */
QAniSliceInstance::QAniSliceInstance(QFile & file) : QAniInstance(QAniInstance::Slice) {
	load(file);
}

QAniSliceInstance::QAniSliceInstance(QAniKeyframe* start, QAniKeyframe* end) : QAniInstance(QAniInstance::Slice, start, end) {
	Slicer s1 = ((QAniSliceKeyframe*)start)->getSlicer();
	Slicer s2 = ((QAniSliceKeyframe*)end)->getSlicer();

	m_vecRadians = acos(s1.getVec().dot(s2.getVec()));
	if (m_vecRadians != 0.0 && m_vecRadians != M_PI) m_vecAxis = s1.getVec() * s2.getVec();

	m_distChange = s2.getDist() - s1.getDist();

	if (s1.isMasking() && s2.isMasking()) m_maskingChange = 2;
	else if (!s1.isMasking() && s2.isMasking()) m_maskingChange = 1;
	else if (!s1.isMasking() && !s2.isMasking()) m_maskingChange = 0;
	else if (s1.isMasking() && !s2.isMasking()) m_maskingChange = -1;
}

QAniSliceInstance::QAniSliceInstance(QAniSliceInstance & instance) : QAniInstance(instance) {
	m_vecRadians = instance.m_vecRadians;
	m_vecAxis = instance.m_vecAxis;
	m_distChange = instance.m_distChange;
	m_maskingChange = instance.m_maskingChange;
}

QAniSliceInstance::~QAniSliceInstance() {
}

void QAniSliceInstance::generateInterpolatedFrame(QAniKeyframe * frame) {
	float itp = (frame->getTime() - getStartTime()) / getDuration();
	Slicer s1 = ((QAniSliceKeyframe*)m_start)->getSlicer();
	Slicer s2 = ((QAniSliceKeyframe*)m_end)->getSlicer();
	Slicer& s = ((QAniSliceKeyframe*)frame)->getSlicer();
	if (m_vecRadians) {
		s.setVec(Matrix4x4::fromRotation(m_vecAxis, m_vecRadians * itp) * s1.getVec());
	}
	s.setDist(s1.getDist() + m_distChange * itp);
	switch (m_maskingChange) {
	case 2:
		s.setMaskingColor(QColor(255*0.75,255*0.75,255*0.75,255*0.25));
		s.setMasking(true);
		break;
	case 1:
		s.setMaskingColor(QColor(255*0.75,255*0.75,255*0.75,255*0.25*itp));
		s.setMasking(true);
		break;
	case 0:
		s.setMaskingColor(QColor(255*0.75,255*0.75,255*0.75,255*0.25));
		s.setMasking(false);
		break;
	case -1:
		s.setMaskingColor(QColor(255*0.75,255*0.75,255*0.75,255*0.25*(1-itp)));
		s.setMasking(true);
		break;
	}
}

bool QAniSliceInstance::operator==(const QAniSliceInstance & instance) const {
	if (!(m_vecRadians == instance.m_vecRadians)) return false;
	if (!(m_vecAxis == instance.m_vecAxis)) return false;
	if (!(m_distChange == instance.m_distChange)) return false;
	if (!(m_maskingChange == instance.m_maskingChange)) return false;
	return true;
}

void QAniSliceInstance::save(QFile & file) const {
	double x,y,z;

	file.write((char *)&m_vecRadians, 8);

	x = m_vecAxis.x();
	y = m_vecAxis.y();
	z = m_vecAxis.z();
	file.write((char *)&x, 8);
	file.write((char *)&y, 8);
	file.write((char *)&z, 8);

	file.write((char *)&m_distChange, 8);
	file.write((char *)&m_maskingChange, 4);
}

void QAniSliceInstance::load(QFile & file) {
	file.read((char *)&m_vecRadians, 8);
	file.read((char *)((double*)m_vecAxis), 24);
	file.read((char *)&m_distChange, 8);
	file.read((char *)&m_maskingChange, 4);
}





/* QAniInstanceManager
 *
 *
 */
QAniInstanceManager::QAniInstanceManager() {
}

QAniInstanceManager::~QAniInstanceManager() {
}

void QAniInstanceManager::addInstance(QAniInstance * instance) {
	(*this)[QAniInstance::ALL].push_back(instance);
	(*this)[instance->getType()].push_back(instance);
}

QAniInstance * QAniInstanceManager::contains(QAniInstance::Type type, float time, QList<QAniInstance*>* list) {
	if (list) list->clear();
	for (int i = 0; i < (*this)[type].size(); ++i) {
		if (time >= (*this)[type][i]->getStartTime() && time <= (*this)[type][i]->getEndTime()) {
			if (list) list->push_back((*this)[type][i]);
			else return (*this)[type][i];
		}
	}
	if (list) if (list->size() > 0) return (*list)[0];
	return NULL;
}

void QAniInstanceManager::removeInstancesAt(QAniInstance::Type type, float time) {
	for (int i = 0; i < (*this)[QAniInstance::ALL].size(); ++i) {
		if (time >= (*this)[QAniInstance::ALL][i]->getStartTime() &&
                    time <= (*this)[QAniInstance::ALL][i]->getEndTime() &&
                    (*this)[QAniInstance::ALL][i]->getType() == type) {
			(*this)[QAniInstance::ALL].removeAt(i--);
		}
	}
	for (int i = 0; i < (*this)[type].size(); ++i) {
		if (time >= (*this)[type][i]->getStartTime() &&
                    time <= (*this)[type][i]->getEndTime()) {
			delete (*this)[type][i];
			(*this)[type].removeAt(i--);
		}
	}
}

void QAniInstanceManager::removeInstancesIn(QAniInstance::Type type, float lefttime, float righttime) {
	for (int i = 0; i < (*this)[QAniInstance::ALL].size(); ++i) {
		if (lefttime <= (*this)[QAniInstance::ALL][i]->getEndTime() &&
		    righttime >= (*this)[QAniInstance::ALL][i]->getStartTime() &&
                    (*this)[QAniInstance::ALL][i]->getType() == type) {
			(*this)[QAniInstance::ALL].removeAt(i--);
		}
	}
	for (int i = 0; i < (*this)[type].size(); ++i) {
		if (lefttime <= (*this)[type][i]->getEndTime() &&
		    righttime >= (*this)[type][i]->getStartTime()) {
			delete (*this)[type][i];
			(*this)[type].removeAt(i--);
		}
	}
}

void QAniInstanceManager::removeALL() {
	for (int i = 0; i < (*this)[QAniInstance::ALL].size(); ++i) {
		delete (*this)[QAniInstance::ALL][i];
	}
	(*this)[QAniInstance::ALL].clear();
	(*this)[QAniInstance::Camera].clear();
	(*this)[QAniInstance::Temporal].clear();
	(*this)[QAniInstance::TransferFunction].clear();
	(*this)[QAniInstance::Slice].clear();
}

QAniInstance * QAniInstanceManager::generateInstancesAccordingTo(QAniKeyframe* start, QAniKeyframe* end) {
	if (start->getType() != end->getType()) return NULL;
	if (start->getTime() == end->getTime()) return NULL;
	if (*start == *end) return NULL;
	if (start->getTime() > end->getTime()) {
		QAniKeyframe * temp = end;
		end = start;
		start = temp;
	}

	QAniInstance * ni = NULL;
	switch (start->getType()) {
	case QAniKeyframe::Camera:
		ni = new QAniCameraInstance(start, end);
		(*this)[QAniInstance::ALL].push_back(ni);
		(*this)[QAniInstance::Camera].push_back(ni);
		break;
	case QAniKeyframe::Temporal:
		ni = new QAniTemporalInstance(start, end);
		(*this)[QAniInstance::ALL].push_back(ni);
		(*this)[QAniInstance::Temporal].push_back(ni);
		break;
	case QAniKeyframe::TransferFunction:
		ni = new QAniTransferFunctionInstance(start, end);
		(*this)[QAniInstance::ALL].push_back(ni);
		(*this)[QAniInstance::TransferFunction].push_back(ni);
		break;
	case QAniKeyframe::Slice:
		ni = new QAniSliceInstance(start, end);
		(*this)[QAniInstance::ALL].push_back(ni);
		(*this)[QAniInstance::Slice].push_back(ni);
		break;
	}
	return ni;
}

QList<QAniInstance*>& QAniInstanceManager::operator[](const QAniInstance::Type t) {
	return instances[t];
}
