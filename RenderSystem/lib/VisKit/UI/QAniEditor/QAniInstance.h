#ifndef _QANIINSTANCE_H_
#define _QANIINSTANCE_H_

#include <GL/glew.h>
#include "QAniKeyframe.h"

class QAniKeyframe;

class QAniInstance {
public:
	enum Type { ALL, Camera, Temporal, TransferFunction, Slice };
	QAniInstance(Type, QAniKeyframe * = NULL, QAniKeyframe * = NULL);
	QAniInstance(QAniInstance &);
	~QAniInstance();

	Type getType() const { return m_type; }
	float getStartTime() const { return m_start->getTime(); }
	float getEndTime() const { return m_end->getTime(); }
	float getDuration() const { return getEndTime() - getStartTime(); }

	void setStart(QAniKeyframe* keyframe, bool setKeyframe = true);
	void setEnd(QAniKeyframe* keyframe, bool setKeyframe = true);
	QAniKeyframe * getStart() { return m_start; }
	QAniKeyframe * getEnd() { return m_end; }

	void setTexture(GLint t) { m_tex = t; }
	GLint getTexture() { return m_tex; }

	bool operator==(const QAniInstance&) const;
	void cloneInstance(QAniInstance*&);

	virtual void generateInterpolatedFrame(QAniKeyframe *);

	void save(QFile &) const;
	static QAniInstance * load(QFile &);

protected:
	Type m_type;
	QAniKeyframe * m_start;
	QAniKeyframe * m_end;
	GLint m_tex;
};

/* class QAniCameraInstance
 *
 *
 */
class QAniCameraInstance : public QAniInstance {
public:
	// LookTrans: Translate CameraOptions::l
	// Trackball: Rotate CameraOptions::c along an axis
	// Tilt:      Rotate CameraOptions::u,o along viewing vector
	QAniCameraInstance(QFile &);
	QAniCameraInstance(double, Vector3);
	QAniCameraInstance(QAniKeyframe*, QAniKeyframe*);
	QAniCameraInstance(QAniCameraInstance&);
	~QAniCameraInstance();

	QAniCameraKeyframe * generateKeyframe(QAniCameraKeyframe *, float);
	void generateInterpolatedFrame(QAniKeyframe *);

	bool operator==(const QAniCameraInstance&) const;

	void save(QFile &) const;
	void load(QFile &);
private:
	Vector3 m_lookTrans;

	double m_trackballRadians;
	Vector3 m_trackballAxis;

	double m_tiltRadians;
	Vector3 m_tiltAxis;

	double m_zoom;
};

/* class QAniTemporalInstance
 *
 *
 */
class QAniTemporalInstance : public QAniInstance {
public:
	QAniTemporalInstance(QFile &);
	QAniTemporalInstance(size_t);
	QAniTemporalInstance(QAniKeyframe*, QAniKeyframe*);
	QAniTemporalInstance(QAniTemporalInstance&);
	~QAniTemporalInstance();

	void generateInterpolatedFrame(QAniKeyframe *);	

	bool operator==(const QAniTemporalInstance&) const;

	void save(QFile &) const;
	void load(QFile &);
private:
	int m_timeStepChange;
};

/* class QAniTransferFunctionInstance
 *
 *
 */
class QAniTransferFunctionInstance : public QAniInstance {
public:
	struct GaussianObjectChange {
		GaussianObjectChange() {}
		GaussianObjectChange(double m, double s, double h) {
			meanChange = m;
			sigmaChange = s;
			heightChange = h;
		}
		GaussianObjectChange& operator=(const GaussianObjectChange& g) {
			meanChange = g.meanChange;
			sigmaChange = g.sigmaChange;
			heightChange = g.heightChange;
			return *this;
		}
		bool operator==(const GaussianObjectChange& g) const {
			if (meanChange != g.meanChange) return false;
			if (sigmaChange != g.sigmaChange) return false;
			if (heightChange != g.heightChange) return false;
			return true;
		}
		double meanChange;
		double sigmaChange;
		double heightChange;
	};
	struct ColorTickChange {
		ColorTickChange() {}
		ColorTickChange(float x, Vector4 color) {
			XChange = x;
			colorChange = color;
		}
		ColorTickChange& operator=(const ColorTickChange& c) {
			XChange = c.XChange;
			colorChange = c.colorChange;
			return *this;
		}
		bool operator==(const ColorTickChange& c) const {
			if (!(XChange == c.XChange)) return false;
			if (!(colorChange == c.colorChange)) return false;
			return true;
		}
		float XChange;
		Vector4 colorChange;
	};

	QAniTransferFunctionInstance(QFile &);
	QAniTransferFunctionInstance(QAniKeyframe*, QAniKeyframe*);
	QAniTransferFunctionInstance(QAniTransferFunctionInstance&);
	~QAniTransferFunctionInstance();

	void generateInterpolatedFrame(QAniKeyframe *);

	bool operator==(const QAniTransferFunctionInstance&) const;

	void save(QFile &) const;
	void load(QFile &);
private:
	int m_resolution;
	int m_tfResolutionChange;
	float * m_tfDrawArrayChange;
	QVector<GaussianObjectChange> m_gaussianObjectChange;
	QVector<ColorTickChange> m_tfColorTickChange;
	
};

/* class QAniSliceInstance
 *
 *
 */
class QAniSliceInstance : public QAniInstance {
public:
	QAniSliceInstance(QFile &);
	QAniSliceInstance(QAniKeyframe*, QAniKeyframe*);
	QAniSliceInstance(QAniSliceInstance&);
	~QAniSliceInstance();

	void generateInterpolatedFrame(QAniKeyframe *);

	bool operator==(const QAniSliceInstance&) const;

	void save(QFile &) const;
	void load(QFile &);
private:
	double m_vecRadians;
	Vector3 m_vecAxis;

	double m_distChange;

	int m_maskingChange; // 2: ON->ON, 1: OFF->ON, 0: OFF->OFF, -1: ON->OFF
};

/* class QAniInstanceManager
 *
 *
 */
class QAniInstanceManager {
public:
	QAniInstanceManager();
	~QAniInstanceManager();

	void addInstance(QAniInstance *);
	QAniInstance * contains(QAniInstance::Type, float, QList<QAniInstance*>* = NULL);
	void removeInstancesAt(QAniInstance::Type, float);
	void removeInstancesIn(QAniInstance::Type, float, float);
	void removeALL();

	QAniInstance * generateInstancesAccordingTo(QAniKeyframe*, QAniKeyframe*);

	QList<QAniInstance*>& operator[](const QAniInstance::Type);

protected:
	QHash<QAniInstance::Type, QList<QAniInstance*> > instances;
};

#endif
