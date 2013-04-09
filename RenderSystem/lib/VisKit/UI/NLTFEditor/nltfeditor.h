#ifndef NLTFEDITOR_H
#define NLTFEDITOR_H

#include <QtGui/QWidget>


struct TFNode;

class NLTFEditor : public QWidget
{
    Q_OBJECT
	TFNode* head;
	TFNode* tail;
	TFNode* selected;
	TFNode* highlighted;
	bool moved;
	QList<QPair<float,float> > mapping;

public:
    NLTFEditor(QWidget *parent = 0);
	virtual ~NLTFEditor();
	const QList<QPair<float,float> >& getNodes() const;
protected:
	void mousePressEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent *);
	void mouseDoubleClickEvent(QMouseEvent *);
	void paintEvent(QPaintEvent *);
	void keyReleaseEvent(QKeyEvent *);
	TFNode* insertNode(float position, float value);
	void functionChanged();
signals:
	void mappingChanged();
};

#endif // NLTFEDITOR_H
