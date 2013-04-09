#ifndef _SLICER_H_
#define _SLICER_H_

#include <QMouseEvent>
#include <QColor>
#include "vectors.h"

class QDataStream;

struct Slicer;
QDataStream & operator<<(QDataStream &, const Slicer &);
QDataStream & operator>>(QDataStream &, Slicer &);

struct Slicer {
	Vector3 m_sliceVec;
	double m_sliceDist;
	bool m_sliceMasking;
	QColor m_sliceMaskingColor;
	int width;
	int height;
	Vector2 prev;
	Vector2 center;

	Slicer();
	Slicer(const Slicer &);
	Slicer(Vector3, double, bool);

	Vector3 getVec() const { return m_sliceVec;  }
	double getDist() const { return m_sliceDist; }
	bool isMasking() const { return m_sliceMasking; }
	QColor getMaskingColor() const { return m_sliceMaskingColor; }
	void setVec(const Vector3 & v) { m_sliceVec = v; }
	void setDist(const double & d) { m_sliceDist = d; }
	void setMasking(bool m) { m_sliceMasking = m; }
	void setMaskingColor(QColor c) { m_sliceMaskingColor = c; }
	void resize(int, int);

	void start(QMouseEvent*, Vector3&, Vector3&);
	void start(const Vector2&, Vector3&, Vector3&);

	void track(QMouseEvent*, Vector3&, Vector3&, Vector3&);
	void track(const Vector2&, Vector3&, Vector3&, Vector3&);

	void move(QMouseEvent*, Vector3&, Vector3&);
	void move(const Vector2&, Vector3&, Vector3&);

	Slicer & operator=(const Slicer&);
	bool operator==(const Slicer&) const;
};

#endif
