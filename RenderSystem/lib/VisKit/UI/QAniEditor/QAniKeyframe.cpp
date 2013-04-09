#include "QAniKeyframe.h"
#include "QAniInstance.h"
#include <cmath>
#ifndef round
#define round(x) (floor(x + 0.5))
#endif



/* class QAniKeyframe
 *
 *
 */
QAniKeyframe::QAniKeyframe(float t, Type type) : m_time(t), m_type(type), m_texture(-1), m_leftInstance(NULL), m_rightInstance(NULL) {
}

QAniKeyframe::~QAniKeyframe() {
	if (m_texture > -1) glDeleteTextures(1, &m_texture);
}

void QAniKeyframe::setTexture(QImage img) {
	m_snapshot = img;
	glGenTextures(1, &m_texture);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits());
	glDisable(GL_TEXTURE_2D);
}

void QAniKeyframe::updateTexture(QImage img) {
	m_snapshot = img;
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.width(), img.height(), GL_RGBA, GL_UNSIGNED_BYTE, img.bits());
	glDisable(GL_TEXTURE_2D);
}

void QAniKeyframe::setLeftInstance(QAniInstance * i, bool setInstance) {
	m_leftInstance = i;
	if (i && setInstance) i->setEnd(this, false);
}

void QAniKeyframe::setRightInstance(QAniInstance * i, bool setInstance) {
	m_rightInstance = i;
	if (i && setInstance) i->setStart(this, false);
}

QAniKeyframe& QAniKeyframe::operator=(const QAniKeyframe &keyframe) {
	if (m_type != keyframe.getType()) return *this;
	switch (m_type) {
	case Camera:
		*((QAniCameraKeyframe*)this) = *((QAniCameraKeyframe*)&keyframe);
		break;
	case Temporal:
		*((QAniTemporalKeyframe*)this) = *((QAniTemporalKeyframe*)&keyframe);
		break;
	case TransferFunction:
		*((QAniTransferFunctionKeyframe*)this) = *((QAniTransferFunctionKeyframe*)&keyframe);
		break;
	case Slice:
		*((QAniSliceKeyframe*)this) = *((QAniSliceKeyframe*)&keyframe);
		break;
	}
	return *this;
}

bool QAniKeyframe::operator==(const QAniKeyframe &keyframe) const {
	if (m_type != keyframe.getType()) return false;
	switch (m_type) {
	case Camera:
		return *((QAniCameraKeyframe*)this) == *((QAniCameraKeyframe*)&keyframe);
		break;
	case Temporal:
		return *((QAniTemporalKeyframe*)this) == *((QAniTemporalKeyframe*)&keyframe);
		break;
	case TransferFunction:
		return *((QAniTransferFunctionKeyframe*)this) == *((QAniTransferFunctionKeyframe*)&keyframe);
		break;
	case Slice:
		return *((QAniSliceKeyframe*)this) == *((QAniSliceKeyframe*)&keyframe);
		break;
	}
	return false;
}

void QAniKeyframe::cloneKeyframe(QAniKeyframe*& keyframe) {
	if (keyframe) delete keyframe;
	switch (m_type) {
	case Camera:
		keyframe = new QAniCameraKeyframe(m_time);
		break;
	case Temporal:
		keyframe = new QAniTemporalKeyframe(m_time);
		break;
	case TransferFunction:
		keyframe = new QAniTransferFunctionKeyframe(m_time);
		break;
	case Slice:
		keyframe = new QAniSliceKeyframe(m_time);
		break;
	}
	*keyframe = *this;
	keyframe->m_snapshot = m_snapshot;
//	keyframe->m_texture = m_texture;
}

bool QAniKeyframe::totallyEqual(const QAniKeyframe *keyframe) const {
	if (!(m_type == keyframe->getType())) return false;
	if (!(*this == *keyframe)) return false;
	if (!(m_time == keyframe->getTime())) return false;
	return true;
}

void QAniKeyframe::save(QFile & file) const {
	file.write((char *)&m_time, 4);
	file.write((char *)&m_type, 4);
	int w = m_snapshot.width();
	int h = m_snapshot.height();
	file.write((char *)&w, 4);
	file.write((char *)&h, 4);
	file.write((char *)(m_snapshot.bits()), m_snapshot.numBytes());
	char left;
	if (m_leftInstance) left = 1;
	else left = 0;
	file.write(&left, 1);

	switch (m_type) {
	case Camera:
		((QAniCameraKeyframe*)this)->save(file);
		break;
	case Temporal:
		((QAniTemporalKeyframe*)this)->save(file);
		break;
	case TransferFunction:
		((QAniTransferFunctionKeyframe*)this)->save(file);
		break;
	case Slice:
		((QAniSliceKeyframe*)this)->save(file);
		break;
	}	
}

QAniKeyframe * QAniKeyframe::load(QFile & file) {
	QAniKeyframe * keyframe;
	float time;
	Type type;
	int w, h;
	file.read((char *)&time, 4);
	file.read((char *)&type, 4);
	file.read((char *)&w, 4);
	file.read((char *)&h, 4);
	unsigned char * img = new unsigned char[w*h*4];
	file.read((char*)img, w*h*4);
	QImage snapshot(img, w, h, QImage::Format_ARGB32);

	char left;
	file.read(&left, 1);

	switch (type) {
	case Camera:
		keyframe = new QAniCameraKeyframe(time);
		((QAniCameraKeyframe*)keyframe)->load(file);
		break;
	case Temporal:
		keyframe = new QAniTemporalKeyframe(time);
		((QAniTemporalKeyframe*)keyframe)->load(file);
		break;
	case TransferFunction:
		keyframe = new QAniTransferFunctionKeyframe(time);
		((QAniTransferFunctionKeyframe*)keyframe)->load(file);
		break;
	case Slice:
		keyframe = new QAniSliceKeyframe(time);
		((QAniSliceKeyframe*)keyframe)->load(file);
		break;
	}
	keyframe->setTexture(snapshot);
	if (left == 1) keyframe->setLeftInstance((QAniInstance*)&left, false);
	else keyframe->setLeftInstance(NULL);
	return keyframe;
}

/* class QAniCameraKeyframe
 *
 *
 */
QAniCameraKeyframe::QAniCameraKeyframe(float t) : QAniKeyframe(t, Camera) {
}

QAniCameraKeyframe::~QAniCameraKeyframe() {
}

QAniCameraKeyframe& QAniCameraKeyframe::operator=(const QAniCameraKeyframe &keyframe) {
	options.l = keyframe.getOptions().l;
	options.c = keyframe.getOptions().c;
	options.u = keyframe.getOptions().u;
	options.o = keyframe.getOptions().o;
	options.dist = keyframe.getOptions().dist;
	options.nearclip = keyframe.getOptions().nearclip;
	options.farclip = keyframe.getOptions().farclip;
	options.fov = keyframe.getOptions().fov;
	options.maxDist = keyframe.getOptions().maxDist;
	return *this;
}

bool QAniCameraKeyframe::operator==(const QAniCameraKeyframe &keyframe) const {
	bool ret = true;
	if (!(options.l == keyframe.options.l)) ret = false;
	if (!(options.c == keyframe.options.c)) ret = false;
	if (!(options.u == keyframe.options.u)) ret = false;
	if (!(options.o == keyframe.options.o)) ret = false;
	if (!(options.dist == keyframe.options.dist)) ret = false;
	if (!(options.nearclip == keyframe.options.nearclip)) ret = false;
	if (!(options.farclip == keyframe.options.farclip)) ret = false;
	if (!(options.fov == keyframe.options.fov)) ret = false;
	if (!(options.maxDist == keyframe.options.maxDist)) ret = false;
	return ret;
}

void QAniCameraKeyframe::save(QFile & file) const {
	double x,y,z;
	// l
	x = options.l.x();
	y = options.l.y();
	z = options.l.z();
	file.write((char *)&x, 8);
	file.write((char *)&y, 8);
	file.write((char *)&z, 8);

	// c
	x = options.c.x();
	y = options.c.y();
	z = options.c.z();
	file.write((char *)&x, 8);
	file.write((char *)&y, 8);
	file.write((char *)&z, 8);

	// u
	x = options.u.x();
	y = options.u.y();
	z = options.u.z();
	file.write((char *)&x, 8);
	file.write((char *)&y, 8);
	file.write((char *)&z, 8);

	// o
	x = options.o.x();
	y = options.o.y();
	z = options.o.z();
	file.write((char *)&x, 8);
	file.write((char *)&y, 8);
	file.write((char *)&z, 8);

	file.write((char *)&(options.dist), 8);
	file.write((char *)&(options.nearclip), 8);
	file.write((char *)&(options.farclip), 8);
	file.write((char *)&(options.fov), 8);
	file.write((char *)&(options.maxDist), 8);
}

void QAniCameraKeyframe::load(QFile & file) {
	file.read((char *)((double*)(options.l)), 24);
	file.read((char *)((double*)(options.c)), 24);
	file.read((char *)((double*)(options.u)), 24);
	file.read((char *)((double*)(options.o)), 24);
	file.read((char *)&(options.dist), 8);
	file.read((char *)&(options.nearclip), 8);
	file.read((char *)&(options.farclip), 8);
	file.read((char *)&(options.fov), 8);
	file.read((char *)&(options.maxDist), 8);
}




/* class QAniTemporalKeyframe
 *
 *
 */
QAniTemporalKeyframe::QAniTemporalKeyframe(float t, size_t timestep) : QAniKeyframe(t, Temporal), m_timeStep(timestep) {
}

QAniTemporalKeyframe::~QAniTemporalKeyframe() {
}

QAniTemporalKeyframe& QAniTemporalKeyframe::operator=(const QAniTemporalKeyframe &keyframe) {
	m_timeStep = keyframe.getTimeStep();
	return *this;
}

bool QAniTemporalKeyframe::operator==(const QAniTemporalKeyframe &keyframe) const {
	return getTimeStep()==keyframe.getTimeStep();
}

void QAniTemporalKeyframe::save(QFile & file) const {
	file.write((char *)&m_timeStep, sizeof(size_t));
}

void QAniTemporalKeyframe::load(QFile & file) {
	file.read((char *)&m_timeStep, sizeof(size_t));
}




/* class QAniTransferFunctionKeyframe
 *
 *
 */
QAniTransferFunctionKeyframe::QAniTransferFunctionKeyframe(float t, float w, float * draw, QVector<GaussianObject> * gaussians, QVector<TFColorTick> * colors) : QAniKeyframe(t, TransferFunction), m_tfResolution(w) {
	m_tfDrawArray = new float[(int)m_tfResolution];
	memcpy(m_tfDrawArray, draw, sizeof(float) * (int)m_tfResolution);
	m_gaussianObjectArray.clear();
	for (int i = 0; i < gaussians->size(); ++i) {
		GaussianObject go((*gaussians)[i].m_mean, (*gaussians)[i].m_sigma, (*gaussians)[i].m_heightFactor, (*gaussians)[i].m_resolution);
		m_gaussianObjectArray.push_back(go);
	}
	m_tfColorTick.clear();
	for (int i = 0; i < colors->size(); ++i) {
		TFColorTick ct((*colors)[i].m_resX, (*colors)[i].m_color);
		m_tfColorTick.push_back(ct);
	}
}

QAniTransferFunctionKeyframe::~QAniTransferFunctionKeyframe() {
	if (m_tfDrawArray) delete [] m_tfDrawArray;
}

void QAniTransferFunctionKeyframe::setTransferFunction(float w, float * draw, const QVector<GaussianObject> * gaussians, const QVector<TFColorTick> * colors) {
	if (m_tfResolution != w) {
		if (m_tfDrawArray) delete [] m_tfDrawArray;
		m_tfResolution = w;
		m_tfDrawArray = new float[(int)m_tfResolution];
	}
	if (!m_tfDrawArray) m_tfDrawArray = new float[(int)m_tfResolution];
	memcpy(m_tfDrawArray, draw, sizeof(float) * (int)m_tfResolution);
	m_gaussianObjectArray.clear();
	for (int i = 0; i < gaussians->size(); ++i) {
		GaussianObject go((*gaussians)[i].m_mean, (*gaussians)[i].m_sigma, (*gaussians)[i].m_heightFactor, (*gaussians)[i].m_resolution);
		m_gaussianObjectArray.push_back(go);
	}
	m_tfColorTick.clear();
	for (int i = 0; i < colors->size(); ++i) {
		TFColorTick ct((*colors)[i].m_resX, (*colors)[i].m_color);
		m_tfColorTick.push_back(ct);
	}
}

QAniTransferFunctionKeyframe& QAniTransferFunctionKeyframe::operator=(const QAniTransferFunctionKeyframe &keyframe) {
	setTransferFunction(keyframe.m_tfResolution, keyframe.m_tfDrawArray, &(keyframe.m_gaussianObjectArray), &(keyframe.m_tfColorTick));
	return *this;
}

bool QAniTransferFunctionKeyframe::operator==(const QAniTransferFunctionKeyframe &keyframe) const {
	if (m_tfResolution != keyframe.m_tfResolution) return false;
	for (int i = 0; i < (int)m_tfResolution; ++i) if (m_tfDrawArray[i] != keyframe.m_tfDrawArray[i]) return false;
	if (m_gaussianObjectArray.size() != keyframe.m_gaussianObjectArray.size()) return false;
	for (int i = 0; i < m_gaussianObjectArray.size(); ++i) {
		if (m_gaussianObjectArray[i].m_mean != keyframe.m_gaussianObjectArray[i].m_mean) return false;
		if (m_gaussianObjectArray[i].m_sigma != keyframe.m_gaussianObjectArray[i].m_sigma) return false;
		if (m_gaussianObjectArray[i].m_heightFactor != keyframe.m_gaussianObjectArray[i].m_heightFactor) return false;
	}
	if (m_tfColorTick.size() != keyframe.m_tfColorTick.size()) return false;
	for (int i = 0; i < m_tfColorTick.size(); ++i) {
		if (m_tfColorTick[i].m_resX != keyframe.m_tfColorTick[i].m_resX) return false;
		if (!(m_tfColorTick[i].m_color == keyframe.m_tfColorTick[i].m_color)) return false;
	}
	return true;
}

void QAniTransferFunctionKeyframe::save(QFile & file) const {
	file.write((char *)&m_tfResolution, 4);
	file.write((char *)m_tfDrawArray, 4 * m_tfResolution);
	int size = m_gaussianObjectArray.size();
	file.write((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		file.write((char *)&(m_gaussianObjectArray[i].m_mean),8);
		file.write((char *)&(m_gaussianObjectArray[i].m_sigma),8);
		file.write((char *)&(m_gaussianObjectArray[i].m_heightFactor),8);
	}
	size = m_tfColorTick.size();
	file.write((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		double t[3];
		file.write((char *)&(m_tfColorTick[i].m_resX),4);
		m_tfColorTick[i].m_color.getRgbF(t, t+1, t+2);
		file.write((char*)t, 24);
	}
}

void QAniTransferFunctionKeyframe::load(QFile & file) {
	file.read((char *)&m_tfResolution, 4);
	m_tfDrawArray = new float[(int)m_tfResolution];
	file.read((char *)m_tfDrawArray, 4 * m_tfResolution);
	int size;
	file.read((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		double mean, sigma, height;
		file.read((char *)&mean, 8);
		file.read((char *)&sigma, 8);
		file.read((char *)&height, 8);
		GaussianObject go(mean, sigma, height, m_tfResolution);
		m_gaussianObjectArray.push_back(go);
	}
	file.read((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		double t[3];
		QColor color;
		float x;
		file.read((char *)&x,4);
		file.read((char*)t, 24);
		color.setRedF(t[0]);
		color.setGreenF(t[1]);
		color.setBlueF(t[2]);
		TFColorTick ct(x, color);
		m_tfColorTick.push_back(ct);
	}
}



/* class QAniSliceKeyframe
 *
 *
 */
QAniSliceKeyframe::QAniSliceKeyframe(float t, Slicer & s) : QAniKeyframe(t, Slice) {
	setSlicer(s);
}

QAniSliceKeyframe::~QAniSliceKeyframe() {
}

void QAniSliceKeyframe::setSlicer(const Slicer slicer) {
	m_slicer.setVec(slicer.getVec());
	m_slicer.setDist(slicer.getDist());
	m_slicer.setMasking(slicer.isMasking());
	m_slicer.setMaskingColor(slicer.getMaskingColor());
}

QAniSliceKeyframe& QAniSliceKeyframe::operator=(const QAniSliceKeyframe &keyframe) {
	setSlicer(keyframe.getSlicer());
	return *this;
}

bool QAniSliceKeyframe::operator==(const QAniSliceKeyframe &keyframe) const {
	bool ret = true;
	if (!(m_slicer.getVec() == keyframe.getSlicer().getVec())) ret = false;
	if (!(m_slicer.getDist() == keyframe.getSlicer().getDist())) ret = false;
	if (!(m_slicer.isMasking() == keyframe.getSlicer().isMasking())) ret = false;
	return ret;
}

void QAniSliceKeyframe::save(QFile & file) const {
	file.write((char *)&m_slicer, sizeof(m_slicer));
}

void QAniSliceKeyframe::load(QFile & file) {
	file.read((char *)&m_slicer, sizeof(m_slicer));
}


/* class QAniKeyframeManager
 *
 *
 */
QAniKeyframeManager::QAniKeyframeManager() {
	currentframes[QAniKeyframe::Camera] = new QAniCameraKeyframe(0.0);
	currentframes[QAniKeyframe::Temporal] = new QAniTemporalKeyframe(0.0);
	currentframes[QAniKeyframe::TransferFunction] = new QAniTransferFunctionKeyframe(0.0);
	currentframes[QAniKeyframe::Slice] = new QAniSliceKeyframe(0.0);
}

QAniKeyframeManager::~QAniKeyframeManager() {
}

QList<QAniKeyframe*>& QAniKeyframeManager::operator[](const QAniKeyframe::Type t) {
	return keyframes[t];
}

QAniKeyframe* QAniKeyframeManager::operator()(const QAniKeyframe::Type t) {
	return currentframes[t];
}

QAniKeyframe* QAniKeyframeManager::findKeyframeAt(float time, QAniKeyframe::Type type) {
	for (int i = 0; i < (*this)[type].size(); ++i) {
		if ((*this)[type][i]->getTime() == time)
			return (*this)[type][i];
	}
	return NULL;
}

QAniKeyframe* QAniKeyframeManager::findNextKeyframe(float time, QAniKeyframe::Type type) {
	for (int i = 0; i < (*this)[type].size(); ++i) {
		if ((*this)[type][i]->getTime() > time)
			return (*this)[type][i];
	}
	return NULL;
}

QAniKeyframe* QAniKeyframeManager::findPrevKeyframe(float time, QAniKeyframe::Type type) {
	for (int i = (*this)[type].size() - 1; i >= 0; --i) {
		if ((*this)[type][i]->getTime() < time)
			return (*this)[type][i];
	}
	return NULL;
}

void QAniKeyframeManager::addKeyframe(QAniKeyframe* keyframe) {
	for (int i = 0; i <= (*this)[QAniKeyframe::ALL].size(); ++i) {
		if (i == (*this)[QAniKeyframe::ALL].size()) {
			(*this)[QAniKeyframe::ALL].push_back(keyframe);
			break;
		}
		else if ((*this)[QAniKeyframe::ALL][i]->getTime() > keyframe->getTime()) {
			(*this)[QAniKeyframe::ALL].insert(i, keyframe);
			break;
		}
	}

	for (int i = 0; i <= (*this)[keyframe->getType()].size(); ++i) {
		if (i == (*this)[keyframe->getType()].size()){
			 (*this)[keyframe->getType()].push_back(keyframe);
			break;
		}
		if ((*this)[keyframe->getType()][i]->getTime() > keyframe->getTime()) {
			(*this)[keyframe->getType()].insert(i, keyframe);
			break;
		}
	}
}

void QAniKeyframeManager::removeKeyframesIn(QAniKeyframe::Type type, float lefttime, float righttime) {
	for (int i = 0; i < (*this)[QAniKeyframe::ALL].size(); ++i) {
		if (lefttime <= (*this)[QAniKeyframe::ALL][i]->getTime() &&
		    righttime >= (*this)[QAniKeyframe::ALL][i]->getTime() &&
                    (*this)[QAniKeyframe::ALL][i]->getType() == type) {
			(*this)[QAniKeyframe::ALL].removeAt(i--);
		}
	}
	for (int i = 0; i < (*this)[type].size(); ++i) {
		if (lefttime <= (*this)[type][i]->getTime() &&
		    righttime >= (*this)[type][i]->getTime()) {
			if ((*this)[type][i]->getLeftInstance() || (*this)[type][i]->getRightInstance())
				itm->removeInstancesAt((QAniInstance::Type)type, (*this)[type][i]->getTime());
			delete (*this)[type][i];
			(*this)[type].removeAt(i--);
		}
	}
}

void QAniKeyframeManager::removeALL() {
	for (int i = 0; i < (*this)[QAniKeyframe::ALL].size(); ++i) {
		delete (*this)[QAniKeyframe::ALL][i];
	}
	(*this)[QAniKeyframe::ALL].clear();
	(*this)[QAniKeyframe::Camera].clear();
	(*this)[QAniKeyframe::Temporal].clear();
	(*this)[QAniKeyframe::TransferFunction].clear();
	(*this)[QAniKeyframe::Slice].clear();

}

void QAniKeyframeManager::translateAfter(QAniKeyframe::Type type, float start, float trans) {
	for (int i = 0; i < (*this)[type].size(); ++i) {
		if ((*this)[type][i]->getTime() >= start)
			(*this)[type][i]->setTime((*this)[type][i]->getTime() + trans);
	}
}
