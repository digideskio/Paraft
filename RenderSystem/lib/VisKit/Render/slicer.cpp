#include "slicer.h"
#include <cmath>
#include <QDataStream>
#define MIN(x,y) (x<=y?x:y)

Slicer::Slicer() {
	m_sliceVec = Vector3(0.0,0.0,1.0);
	m_sliceDist = 0.0;
	m_sliceMasking = true;
	m_sliceMaskingColor.setRgbF(0.75,0.75,0.75,0.25);
	width = 0;
	height = 0;
}

Slicer::Slicer(const Slicer & slicer) {
	(*this) = slicer;
}

Slicer::Slicer(Vector3 vec, double dist, bool mask) {
	m_sliceVec = vec;
	m_sliceDist = dist;
	m_sliceMasking = mask;
	m_sliceMaskingColor.setRgbF(0.75,0.75,0.75,0.25);
	width = 0;
	height = 0;
}

void Slicer::resize(int w, int h) {
	width = w;
	height = h;
}

void Slicer::start(QMouseEvent * e, Vector3 & up, Vector3 & right) {
	start(Vector2(e->x(), e->y()), up, right);
}

void Slicer::start(const Vector2 & e, Vector3 & up, Vector3 & right) {
	prev = e;
	double radius = ((double)MIN(width,height))/2.0;
	center = e;
	center -= Vector2(m_sliceVec.dot(right), -m_sliceVec.dot(up)) * radius;
}

void Slicer::track(QMouseEvent * e, Vector3 & view, Vector3 & up, Vector3 & right) {
	track(Vector2(e->x(), e->y()), view, up, right);
}

void Slicer::track(const Vector2 & e, Vector3 & view, Vector3 & up, Vector3 & right) {
	if (width == 0 || height == 0) return;

	double radius = ((double)MIN(width,height))/2.0;
	Vector2 pos((e - center)/radius);
	if (pos.length() > 1.0) pos.normalize();
	double viewportion = cos(asin(pos.length()));
	if (viewportion != viewportion) viewportion = 0.0;
	m_sliceVec = view * viewportion + up * -pos.y() + right * pos.x();
	m_sliceVec.normalize();

	prev = e;
}

void Slicer::move(QMouseEvent * e, Vector3 & up, Vector3 & right) {
	move(Vector2(e->x(), e->y()), up, right);
}

void Slicer::move(const Vector2 & e, Vector3 & up, Vector3 & right) {
	if (width == 0 || height == 0) return;
	double radius = ((double)MIN(width,height))/2.0;
	Vector2 pos((e - prev)/radius);
	Vector3 realpos = up * -pos.y() + right * pos.x();
	if (realpos.dot(m_sliceVec) >= 0.0) m_sliceDist += pos.length();
	else m_sliceDist -= pos.length();

	prev = e;
}

Slicer & Slicer::operator=(const Slicer & src) {
	this->m_sliceVec = src.m_sliceVec;
	this->m_sliceDist = src.m_sliceDist;
	this->m_sliceMasking = src.m_sliceMasking;
	this->m_sliceMaskingColor = src.m_sliceMaskingColor;
	this->width = src.width;
	this->height = src.height;
	return *this;
}

bool Slicer::operator==(const Slicer & src) const {
	if (!(this->m_sliceVec == src.m_sliceVec)) return false;
	if (this->m_sliceDist != src.m_sliceDist) return false;
	if (this->m_sliceMasking != src.m_sliceMasking) return false;
	return true;
}

QDataStream & operator<<(QDataStream & out, const Slicer & slicer) {
qDebug("slice1");
	out << slicer.m_sliceVec.x() << slicer.m_sliceVec.y() << slicer.m_sliceVec.z();
qDebug("slice2");
	out << slicer.m_sliceDist;
qDebug("slice3");
	out << slicer.m_sliceMasking;
qDebug("slice4");
	return out;
}
QDataStream & operator>>(QDataStream & in, Slicer & slicer) {
	in >> slicer.m_sliceVec.x() >> slicer.m_sliceVec.y() >> slicer.m_sliceVec.z();
	in >> slicer.m_sliceDist;
	in >> slicer.m_sliceMasking;
	return in;
}
