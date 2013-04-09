#ifndef _QCOLORPALETTE_H_
#define _QCOLORPALETTE_H_
#include <QWidget>
#include <QColor>
#include <QPushButton>
#include <QPaintEvent>
#include <QVector>
class QColorButton:public QPushButton
{
	Q_OBJECT
public:
	QColorButton(QWidget *parent, QColor cr=QColor::fromRgbF(1.0,1.0,1.0,1.0));
	void	setColor(QColor &cr)
	{		m_color = cr;	repaint();}
	QColor&	getColor(){return m_color;}
	
private:
	QColor m_color;

protected:
	void paintEvent(QPaintEvent *);
	void mouseReleaseEvent(QMouseEvent*);	
};

class QColorPicker;
class QColorButton;
class QSignalMapper;

class QColorPalette:public QWidget
{
Q_OBJECT
public:
	QColorPalette(QWidget *parent);
	QColorPicker	*m_pColorPicker;
	QColorButton	*m_nowColor;
	QVector<QColorButton*>	m_colorArray;
	
	int			getColorCounter(){return m_colorCounter;}
	void		setColorCounter(int cr){m_colorCounter = cr;}
	
private:
	void			initLayout();
	QSignalMapper	*signalMapper;
	int				m_colorCounter;
	int				m_nosColorBlockLayout;
	int				m_nosColorPoolButton;
	int				m_colorBlockWidth;
	int				m_colorBlockHeight;
	int				m_colorBlockSpacing;
	
public slots:
	void		updateColor(int);
	void		colorArrayClicked(const QString &);

protected:
	void		mouseReleaseEvent(QMouseEvent*);
};

#endif
