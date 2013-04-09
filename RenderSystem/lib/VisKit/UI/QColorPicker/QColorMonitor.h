#ifndef _QCOLORMONITOR_H_
#define _QCOLORMONITOR_H_

#include <QWidget>

class QColorPicker;
class QSlider;
class QSpinBox;

class QColorMonitor : public QWidget
{
	Q_OBJECT
public:
	QColorMonitor(QWidget *parent);
	QColorPicker	*m_pColorPicker;

private:
	void		initLayout();

	QSlider		*m_HSlider, *m_SSlider, *m_VSlider, 
				*m_RSlider, *m_GSlider, *m_BSlider;
	QSpinBox	*m_HBox, *m_SBox, *m_VBox, 
				*m_RBox, *m_GBox, *m_BBox;
public slots:
	void	adjustColorSwitches(int);
	void	updatePanelColorHSV();
	void	updatePanelColorRGB();	
};

#endif
