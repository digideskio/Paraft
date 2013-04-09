#ifndef _QANIGRAPH_H_
#define _QANIGRAPH_H_
#include <GL/glew.h>
#include <Qt>
#include <QGLWidget>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QList>
#include <QFile>

class QAniEditor;
class QAniKeyframe;

class QAniGraphNode {
public:
	QAniGraphNode();
	~QAniGraphNode();

	bool contains(double x, double y) { return m_rect.contains(x, y); }
	double x() { return m_rect.x(); }
	double y() { return m_rect.y(); }
	double width() { return m_rect.width(); }
	double height() { return m_rect.height(); }

protected:
	friend class QAniGraph;
	QAniKeyframe * m_keyframe;
	QRectF m_rect;
	bool m_hovered;
	bool m_highlighted;
	bool m_current;
	QAniGraphNode * parent;
	QList<QAniGraphNode *> children;
};

/* class QAniGraph
 *
 *
 */

class QAniGraph : public QGLWidget {
	Q_OBJECT
public:
	QAniGraph(QWidget *, const QGLWidget *);
	~QAniGraph();


protected:
	virtual void initializeGL();
	virtual void resizeGL(int, int);
	virtual void paintGL();

	void drawRecursiveGraph(QList<QAniGraphNode *> *);
	void hoverRecursiveGraph(QList<QAniGraphNode *> *, double, double);
	void clearCurrentRecurrsiveGraph(QList<QAniGraphNode *> *);
	void saveRecursiveGraph(QFile &, QList<QAniGraphNode *> *);
	void loadRecursiveGraph(QFile &, QAniGraphNode *, int);
	int getBreadth(QAniGraphNode *);
	void updatePosition(QAniGraphNode *);
	void removeALL();
	void removeRecursiveGraph(QList<QAniGraphNode *> *);

	virtual void mousePressEvent(QMouseEvent *);
	virtual void mouseMoveEvent(QMouseEvent *);
	virtual void mouseReleaseEvent(QMouseEvent *);
	virtual void wheelEvent(QWheelEvent *);

	QAniEditor * m_aniEditor;
	QAniGraphNode * root;
	QList<QAniGraphNode *> * head;
	QList<int> m_currentNodes;

public:
	void buildGraph(QList<QAniKeyframe*> *);
	void saveGraph(QFile &);
	void loadGraph(QFile &);

private:
	int m_width;
	int m_height;
	double m_offsetx;
	double m_offsety;
	double m_zoom;
	int m_oldx;
	int m_oldy;
	bool m_mousePressed;
	QAniGraphNode * m_hoveredNode;
	

public slots:
	void aniUpdate();

public:
signals:
	void pickedTimelineFromGraph(QList<QAniKeyframe*> *);
};

#endif
