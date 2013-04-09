#ifndef _QANIKEYFRAME_H_
#define _QANIKEYFRAME_H_

#include <QList>
#include <QHash>
#include <GL/glew.h>
#include "camera.h"
#include "slicer.h"
#include "QTFPanel.h"
//#include "QAniInstance.h"

class QAniInstance;
class QAniInstanceManager;

class QAniKeyframe {
public:
	enum Type { ALL, Camera, Temporal, TransferFunction, Slice };
	QAniKeyframe(float, Type);
	~QAniKeyframe();

	float getTime() const { return m_time; }
	void setTime(float t) { m_time = t; }
	Type getType() const { return m_type; }
	void setTexture(QImage img);
	void updateTexture(QImage img);
	GLuint getTexture() const { return m_texture; }
	void setSnapshot(QImage img) { m_snapshot = img; }
	QImage getSnapshot() const { return m_snapshot; }

	void setLeftInstance(QAniInstance *, bool = true);
	QAniInstance * getLeftInstance() const { return m_leftInstance; }
	void setRightInstance(QAniInstance *, bool = true);
	QAniInstance * getRightInstance() const { return m_rightInstance; }
	
	QAniKeyframe& operator=(const QAniKeyframe&);
	bool operator==(const QAniKeyframe&) const;
	void cloneKeyframe(QAniKeyframe*&);
	bool totallyEqual(const QAniKeyframe *) const;

	void save(QFile &) const;
	static QAniKeyframe * load(QFile &);

protected:
	float m_time;
	Type m_type;
	QImage m_snapshot;
	GLuint m_texture;

	QAniInstance * m_leftInstance;
	QAniInstance * m_rightInstance;
};

/* class QAniCameraKeyframe
 *
 *
 */
class QAniCameraKeyframe : public QAniKeyframe {
public:
	QAniCameraKeyframe(float);
	~QAniCameraKeyframe();

	CameraOptions  getOptions() const { return options; }
	CameraOptions& getOptions() { return options; }

	QAniCameraKeyframe& operator=(const QAniCameraKeyframe&);
	bool operator==(const QAniCameraKeyframe&) const;

	void save(QFile &) const ;
	void load(QFile &);
protected:
	CameraOptions options;
};

/* class QAniTemporalKeyframe
 *
 *
 */
class QAniTemporalKeyframe : public QAniKeyframe {
public:
	QAniTemporalKeyframe(float, size_t = 0);
	~QAniTemporalKeyframe();

	void setTimeStep(size_t t) { m_timeStep = t; }
	size_t getTimeStep() const { return m_timeStep; }

	QAniTemporalKeyframe& operator=(const QAniTemporalKeyframe&);
	bool operator==(const QAniTemporalKeyframe&) const;

	void save(QFile &) const;
	void load(QFile &);
protected:
	size_t m_timeStep;
};

/* class QAniTransferFunctionKeyframe
 *
 *
 */
class QAniTransferFunctionKeyframe : public QAniKeyframe {
public:
	QAniTransferFunctionKeyframe(float t) : QAniKeyframe(t, TransferFunction) { m_tfDrawArray = NULL; }
	QAniTransferFunctionKeyframe(float, float, float *, QVector<GaussianObject> *, QVector<TFColorTick> *);
	~QAniTransferFunctionKeyframe();

	void setTransferFunction(float, float *, const QVector<GaussianObject> *, const QVector<TFColorTick> *);
	int getResolution() { return (int)m_tfResolution; }
	float * getDrawArray() { return m_tfDrawArray; }
	QVector<GaussianObject> * getGaussians() { return &m_gaussianObjectArray; }
	QVector<TFColorTick> * getColors() { return &m_tfColorTick; }

	QAniTransferFunctionKeyframe& operator=(const QAniTransferFunctionKeyframe&);
	bool operator==(const QAniTransferFunctionKeyframe&) const;

	void save(QFile &) const;
	void load(QFile &);
protected:
	float m_tfResolution;
	float * m_tfDrawArray;
	QVector<GaussianObject>	m_gaussianObjectArray;
	QVector<TFColorTick>	m_tfColorTick;
};

/* class QAniSliceKeyframe
 *
 *
 */
class QAniSliceKeyframe : public QAniKeyframe {
public:
	QAniSliceKeyframe(float t) : QAniKeyframe(t, Slice) { m_slicer.setVec(Vector3(0.0,0.0,0.0));}
	QAniSliceKeyframe(float, Slicer &);
	~QAniSliceKeyframe();

	void setSlicer(const Slicer);
	Slicer getSlicer() const { return m_slicer; }
	Slicer& getSlicer() { return m_slicer; }

	QAniSliceKeyframe& operator=(const QAniSliceKeyframe&);
	bool operator==(const QAniSliceKeyframe&) const;

	void save(QFile &) const;
	void load(QFile &);
protected:
	Slicer m_slicer;
};

/* class QAniKeyframeManager
 *
 *
 */
class QAniKeyframeManager {
public:
	QAniKeyframeManager();
	~QAniKeyframeManager();

	void setInstanceManager(QAniInstanceManager * m) { itm = m; }

	void addKeyframe(QAniKeyframe*);
	void removeKeyframesIn(QAniKeyframe::Type, float, float);
	void removeALL();
	void translateAfter(QAniKeyframe::Type, float, float);

	QAniKeyframe* findKeyframeAt(float, QAniKeyframe::Type = QAniKeyframe::ALL);
	QAniKeyframe* findNextKeyframe(float, QAniKeyframe::Type = QAniKeyframe::ALL);
	QAniKeyframe* findPrevKeyframe(float, QAniKeyframe::Type = QAniKeyframe::ALL);

	QList<QAniKeyframe*>& operator[](const QAniKeyframe::Type);
	QAniKeyframe* operator()(const QAniKeyframe::Type);

protected:
	QHash<QAniKeyframe::Type, QList<QAniKeyframe*> > keyframes;
	QHash<QAniKeyframe::Type, QAniKeyframe*> currentframes;
	QAniInstanceManager * itm;
};

#endif
