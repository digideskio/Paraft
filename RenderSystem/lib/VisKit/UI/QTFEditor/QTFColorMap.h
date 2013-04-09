#ifndef _QTFCOLORMAP_H_
#define _QTFCOLORMAP_H_

#include <QWidget>
#include "QTFAbstractPanel.h"
class QTFEditor;
class QMenu;

class QTFColorMap : public QTFAbstractPanel
{
	Q_OBJECT
public:
	QTFColorMap(QWidget *parent=0);
	QColor getBackgroundColor() { return m_colorBackground; }
	void	changeBGColor(QColor);
	//void	updateTFColorMap();
private:
	void	initLayout();

	void	updatePanelImage();
	QColor	m_colorBackground;
	
	QMenu*	m_backgroundMenu;
	QAction* changeBG2WhiteAct;
	QAction* changeBG2BlackAct;
	QAction* changeBG2SelectedAct;
	
	
protected:
	void realPaintEvent(QPaintEvent *);
//	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
//	void mouseReleaseEvent(QMouseEvent*);
	
public slots:
	void updateTFColorMap();
	void changeBG2White();
	void changeBG2Black();
	void changeBG2Selected();

signals:
	void bgColorChanged(const QColor&);	
};
#endif

