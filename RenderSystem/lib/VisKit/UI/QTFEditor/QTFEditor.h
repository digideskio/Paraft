#ifndef _QTFEDITOR_H_
#define _QTFEDITOR_H_

#include <QWidget>
#include <QIODevice>
#include <QString>
#include "QHistogram.h"
#include "QTFPanel.h"
#include "QTFColorMap.h"
#include "../QColorPicker/QColorPicker.h"
#include "histogram.h"
#include <QByteArray>

class Histogram;

class QTFEditor : public QWidget{
	Q_OBJECT
public:
	QTFEditor(int transferWidth=1024, QWidget *parent=0, const char *file=NULL);
	~QTFEditor();
	
	int				getTFColorMapResolution(){ return m_tfPanel->getTFColorMapResolution();}
	float*			getTFColorMap(){return m_tfPanel->getTFColorMap();}
	float*			getTFMappingMap(){return m_tfPanel->getXYMappingMap();}
	QColor			getColor(float pos);
	Histogram*		getHistogram() { return m_histogramData; }
	QHistogram*		getQHistogram() { return m_histogramPanel; }
	QTFPanel*		getTFPanel() { return m_tfPanel; }
	QTFColorMap*	getColorMap() { return m_tfColorMapPanel; }
	QColorPicker*	getColorPicker() { return m_colorPicker; }
	void			incrementHistogram(double);
	void			setHistogramMinMax(double, double);
	void			setHistogram(Histogram*);
	void			updateHistogram(Histogram*);
	QIODevice&		saveSettings(QIODevice&);
	QIODevice&		loadSettings(QIODevice&);
	void			saveSettings(TF&);
	void			loadSettings(TF&);

	// get complete TF information
	float				getTFResolution() {return m_tfPanel->getTFResolution();}
	float*				getTFDrawArray() {return m_tfPanel->getTFDrawArray();}
	QVector<GaussianObject>*	getGaussians() {return m_tfPanel->getGaussians();}
	QVector<TFColorTick>*		getColors() {return m_tfPanel->getColors();}

	void		readDefaultSettings();
	void		writeDefaultSettings();
	
public slots:
	void			newHistogram(Histogram*);
	void			loadByteArray(QByteArray);

private:
	void	initLayout(int);
	QHistogram		*m_histogramPanel;
	QTFPanel		*m_tfPanel;
	QTFColorMap		*m_tfColorMapPanel;
	QColorPicker	*m_colorPicker;
	
	Histogram		*m_histogramData;
	QSize			m_tfEditorSize;
	QSize			m_histogramPanelSize,m_tfPanelSize,m_tfColorMapPanelSize,m_colorPickerSize;

	QString			m_tfefilename;

protected slots:
	void	changeTickColor(int);
	void	changedTFColorMap();
};

#endif

