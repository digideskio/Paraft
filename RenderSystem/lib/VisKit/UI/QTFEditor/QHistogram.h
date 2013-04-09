#ifndef _QHISTOGRAM_H_
#define _QHISTOGRAM_H_

#include <QWidget>
#include "QTFAbstractPanel.h"
class QTFEditor;
class QImage;
class QMenu;

#include <QList>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
struct Range {
	float min, max;
	Range(float min=0, float max=1):min(min), max(max) {}
	bool inside(float value) {
		return (value >= min && value <= max);
	}
	bool isOnEdge(float value);
	void merge(const Range& other) {
		if(other.min < min)
			min = other.min;
		if(other.max > max)
			max = other.max;
	}
};

class QHistogram : public QTFAbstractPanel
{
	Q_OBJECT
public:
	QHistogram(QWidget *parent=0);

private:
	void	initLayout();
	void	initMenu();
	void	updatePanelImage();
	void	startZero(int v);
	void	endZero(int v);
	void	drawZero(int);
	bool	inZero(int value);
	bool	isOnZeroEdge(int value);
	float	convertX(int value); //convert from absolute coordinates to relative
	
	float	m_histogramMax,m_histogramMin;
	float	m_binWidth,m_binMaxHeight;
	QMenu	*m_optionMenu;
	QMenu	*m_zeroMenu;
	QAction	*changeStyle2LineAct,*changeStyle2BarAct;
	QAction	*changeBM2NoneAct,*changeBM2DotLineAct;
	QAction *changeYT2LinearAct,*changeYT2LogAct;
	QAction *removeZeroRangeAct, *toggleZeroLockAct;
	QList<Range> m_zeros;
	QList<Range>::iterator m_currentZero;
	int			m_histogramType; // 1:curve, 2:bar
	int			m_backgroundMesh; // 0: none, 1: dot mesh
	int			m_yTransform; // 0:linear, 1: log
	bool	m_drawingZero;
	Range	m_tempZero; //used for drawing the current zero

private slots:
	void removeZeroRange();
	void toggleZeroLock();

protected:
	void realPaintEvent(QPaintEvent *);
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);
	
public slots:
	void changeStyle2Line();
	void changeStyle2Bar();
	void changeBM2None();
	void changeBM2DotLine();
	void changeYT2Linear();
	void changeYT2Log();
	void updateHistogram();
};
#endif

