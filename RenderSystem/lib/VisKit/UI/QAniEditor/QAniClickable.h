#ifndef _QANICLICKABLE_H_
#define _QANICLICKABLE_H_

#include <GL/glew.h>
#include <Qt>
#include <QWidget>
#include <QGLWidget>
#include <QPixmap>
#include <QTime>
#include <QTimer>
#include "QAniKeyframe.h"
#include "QAniInstance.h"

using namespace std;

class QAniEditor;
class QAniToolBox;
class QAniTemplateBox;
class QAniTemplate;
class QAniTimeline;
class QAniTimelineScaler;
class QAniTimelineInstance;

class QAniClickable : public QWidget {
public:
	float x, y;
	float w, h;
	void setRect(int _x, int _y, int _w, int _h) {x = _x; y = _y; w = _w; h = _h;}
	bool selected;
	bool visible;

	bool dragable;
	bool dragging;

	virtual void draw();
	bool isSelected();
	void updateGL();

	QAniEditor *timelineGL;

	void setCursor(Qt::CursorShape q);

	QAniClickable(QWidget*);
	~QAniClickable() {}

	virtual void setSelected(bool value) { selected = value; }
	virtual void drag(float dx, float dy);
	virtual void hover(float, float) {}
	virtual void press(float, float) {}
	virtual void release(float, float) {}
	virtual void wheel(float, float, int) {}
	virtual bool encloses(float px, float py);
	virtual bool isVisible();
	virtual void setVisible(bool value);
	virtual void click() {}
	virtual void resize() {}
};

/* class ToolBox
 *
 *
 */

class QAniToolBox : public QAniClickable {
	Q_OBJECT
public:
	QAniToolBox(QWidget * parent = 0);
	~QAniToolBox();

	virtual void draw();
	virtual void press(float px, float py);
	virtual void hover(float px, float py);
	virtual void release(float px, float py);
	virtual void resize();

private:
	void drawBackground();
	void drawButtons();

	bool m_playing;
	bool m_hoverPlaying;
	bool m_hoverRecord;

public:
signals:
	void playButtonHit();
	void recordButtonHit();
	void menuButtonHit();
};

/* class TemplateBox
 *
 *
 */

class QAniTemplateBox : public QAniClickable {
	Q_OBJECT
public:
	QAniTemplateBox(QWidget * parent = 0);
	~QAniTemplateBox();

	QAniTimelineScaler *scaler;
	void setScaler(QAniTimelineScaler *s) { scaler = s; }

	vector<QAniTemplate*> templates;
	float offsety;
	int scroll; // 0: stop  1: up  -1: down
	void add(QAniTemplate*);
	void drawBackground();
	bool checkOffsety();

	int selectedIdx;

	int templateDragIdx;
	QAniTemplate* getTemplateDrag() { return (*this)[templateDragIdx]; }
	bool isDragging() { return templateDragIdx >= 0; }
	void setDragging(int idx) { templateDragIdx = idx; }
	void clearDragging() { templateDragIdx = -1; }
	QAniTemplate*& operator[](int & index) { return templates[index]; }

	virtual void draw();
	virtual void press(float px, float py);
	virtual void drag(float px, float py);
	virtual void hover(float px, float py);
	virtual void release(float px, float py);
	virtual void wheel(float px, float py, int delta);
	virtual void resize();

public slots:
	void scrollTimer();
};

/* class Template
 *
 *
 */

class QAniTemplate : public QAniClickable {
	Q_OBJECT
public:
	enum Type {SpacialOverview, TemporalOverview, TransferFunctionOverview};
	QAniTemplate(Type t, QWidget * parent = 0);
	Type type;
	GLint tex;
	QAniTemplateBox *tbox;
	
	virtual void draw(float offset = 35.0);
	virtual void drag(float px, float py);
	virtual void release(float px, float py);
	void drawDragging();
	Type getType() { return type; }

	float fx;
	bool first;
	bool dragInTimeline;
	void resetDragging(float px, float py);
	bool isFirstSet() { return first; }
	void setFirstPoint(float px);
	void setSecondPoint(float px, float py);

public:
signals:
	void setTemplate(QAniTemplate::Type, float, float, GLint);
};

/* class Timeline
 *
 *
 */

class QAniTimeline : public QAniClickable {
	Q_OBJECT
public:
	QAniTimeline(QWidget * parent = 0);

	float opx;
	float opy;
	float cursorx; // window space x
	float currentx; // time space x

	QAniTimelineScaler *scaler;
	QAniTemplateBox *tbox;
	void setScaler(QAniTimelineScaler *s) { scaler = s; }
	void setTemplateBox(QAniTemplateBox *b) { tbox = b; }

	vector<QAniTimelineInstance*> instances;

	virtual void draw();
	virtual void resize();
	virtual void press(float px, float py);
	virtual void hover(float px, float py);
	virtual void drag(float px, float py);
	virtual void release(float px, float py);

	QTimer *playingTimer;
private:
	void drawBackground();
	void drawCursor();
	void drawCurrent();
	void drawInstance(QAniInstance *);
	void drawKeyframe(QAniKeyframe *);

	int m_lastTime;
	bool draggingCurrentTime;
	bool draggingKeyframe;
	int  draggingKeyframeIdx;
	bool draggingInstance;
	float draggingStartTime;
	float draggingInstanceTime;
	int  draggingInstanceIdx;
	bool hoverCameraKeyframeKey;
	bool hoverTemporalKeyframeKey;
	bool hoverTransferFunctionKeyframeKey;
	bool hoverSliceKeyframeKey;
	bool pressedCameraKeyframeKey;
	bool pressedTemporalKeyframeKey;
	bool pressedTransferFunctionKeyframeKey;
	bool pressedSliceKeyframeKey;

	QList<float> m_snapshotPos;

public slots:
	void playButtonHit();
	void recordButtonHit();
	void playingTimeout();

public:
signals:
	void setKeyframe(float, QAniKeyframe::Type = QAniKeyframe::ALL);
	void currentTimeChange();
};

/* class TimelineScaler
 *
 *
 */

class QAniTimelineScaler : public QAniClickable {
	Q_OBJECT
public:
	QAniTimelineScaler(QWidget * parent = 0);
	float offset;
	float scale;
	float drag_ox, drag_oy;

	float WtoT(float wx) { return offset + (wx - 51.0) / scale; } // Transform Coord from Window Space to Timeline Space
	float TtoW(float tx) { return (tx - offset) * scale + 51.0; } // Transform Coord from Timeline Space to Window Space

	virtual void draw();
	virtual void wheel(float px, float py, int delta);
	virtual void press(float px, float py);
	virtual void drag(float px, float py);
	virtual void release(float px, float py);
	virtual void resize();

	QTimer *offsetTimer;
	float scrollDx;

public slots:
	void offsetTimeout();
};

/* class TimelineInstance
 *
 *
 */

class QAniTimelineInstance : public QAniClickable {
public:
	QAniTimelineInstance(float tx, float ty, QAniTemplate::Type t, QAniTimeline * tl, QWidget * parent = 0);

	QAniTimeline* timeline;

	virtual void draw();
	virtual void press(float px, float py);
	virtual void drag(float dx, float dy);
	virtual void release(float px, float py);
	virtual bool encloses(float px, float py);
};
#endif
