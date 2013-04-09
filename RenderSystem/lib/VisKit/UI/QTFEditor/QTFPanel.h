#ifndef _QTFPANEL_H_
#define _QTFPANEL_H_

#include <QWidget>
#include <QVector>
#include <cmath>
#include <QFile>
#include "QTFAbstractPanel.h"
#include <QLabel>

class NLTFEditor;
class QTFEditor;
class QSlider;
class QPushButton;
class QMenu;
class QLineEdit;

class TFColorTick{
public:
	TFColorTick():m_resX(0.0f),m_color(Qt::white){}
	TFColorTick(float posX,QColor cr):m_resX(posX),m_color(cr){}
	float		m_resX;
	QColor		m_color;

	bool operator==(const TFColorTick & src)
	{
		if (m_resX != src.m_resX) return false;
		if (m_color != src.m_color) return false;
		return true;
	}
	TFColorTick& operator=(const TFColorTick & src)
	{
		this->m_resX = src.m_resX;
		this->m_color= src.m_color;
		return *this;
	}
};
struct ZeroRange {
	ZeroRange():start(0.f), end(100.f){}
	ZeroRange(float s, float e):start(s), end(e){}
	float start;
	float end;
};
class GaussianObject{
public:
	GaussianObject():m_mean(0.0),m_sigma(1.0),m_heightFactor(0.3),m_resolution(100),m_distribution(NULL){}
	GaussianObject(const GaussianObject & go) { (*this) = go; }
	GaussianObject(double m,double s,double h,int res):m_mean(m),m_sigma(s),
	m_heightFactor(h),m_resolution(res),m_distribution(NULL){}
	~GaussianObject() { if (m_distribution) delete [] m_distribution; }
	double	m_mean;
	double	m_sigma;
	double	m_heightFactor; // 0.1~1
	int		m_resolution;
	float	*m_distribution;
	void	update()
	{
		if(m_distribution == NULL)
			m_distribution = new float[m_resolution];

		double meanPt = m_mean * m_resolution;
		double sigmaPt = m_sigma * m_resolution;
		double heightPt = m_heightFactor*m_resolution;
		for(int x=0;x<m_resolution;++x)
		{
			m_distribution[x] = heightPt*gaussianGen(x,meanPt,sigmaPt);
		}
	}
	double value(double x) {
		return m_heightFactor*gaussianGen(x,m_mean,m_sigma);

	}
	static double	gaussianGen(double x,double mean,double sigma)
	{
		double pp = x-mean;
		//double val = 1.0/(sigma*sqrt(2.0*3.1415926)) * exp(-(pp)*(pp)/(2.0*sigma*sigma));
		return 1.0/(sigma*sqrt(2.0*3.1415926)) * exp(-(pp)*(pp)/(2.0*sigma*sigma));
	}
	bool operator==(const GaussianObject & src)
	{
		if (m_mean != src.m_mean) return false;
		if (m_sigma != src.m_sigma) return false;
		if (m_heightFactor != src.m_heightFactor) return false;
		return true;
	}
	GaussianObject& operator=(const GaussianObject & src)
	{
		this->m_resolution = src.m_resolution;
		this->m_mean = src.m_mean;
		this->m_sigma = src.m_sigma;
		this->m_heightFactor = src.m_heightFactor;
		this->m_distribution = NULL;
		this->update();
		return *this;
	}
};
struct TF {
	TF() {
		tfResolution = 1024;
		tfDrawArray = new float[(int)tfResolution];
		for (int i = 0; i < tfResolution; ++i) tfDrawArray[i] = 0.0f;
		combineMode = 0;
		tranSliderValue = 5;
		backgroundMesh = 1;
		backgroundColor = QColor(0,0,0);
		tfColorTick << TFColorTick(0.0f, QColor::fromHsvF(0.611,1.0,1.0))
		            << TFColorTick(tfResolution * 1 / 6, QColor::fromHsvF(0.500,1.0,1.0))
		            << TFColorTick(tfResolution * 2 / 6, QColor::fromHsvF(0.353,1.0,1.0))
		            << TFColorTick(tfResolution * 3 / 6, QColor::fromHsvF(0.176,1.0,1.0))
		            << TFColorTick(tfResolution * 4 / 6, QColor::fromHsvF(0.086,1.0,1.0))
		            << TFColorTick(tfResolution * 5 / 6, QColor::fromHsvF(0.000,1.0,1.0))
		            << TFColorTick(tfResolution * 6 / 6, QColor::fromHsvF(0.784,1.0,1.0));
		gaussianObjectArray << GaussianObject(0.5,0.133,0.166,tfResolution);
	}
	TF(const TF & tf) {
		tfDrawArray = NULL;
		(*this) = tf;
	}
	TF(const QString & filename) {
		tfDrawArray = NULL;
		load(filename);
	}
	~TF() {
		if (tfDrawArray) delete [] tfDrawArray;
	}
	bool operator==(const TF & src) const {
		if (tfResolution != src.tfResolution) return false;
		for (int i = 0; i < tfResolution; ++i) if (tfDrawArray[i] != src.tfDrawArray[i]) return false;
		if (!(gaussianObjectArray == src.gaussianObjectArray)) return false;
		if (!(tfColorTick == src.tfColorTick)) return false;
		if (combineMode != src.combineMode) return false;
		if (tranSliderValue != src.tranSliderValue) return false;
		if (backgroundMesh != src.backgroundMesh) return false;
		if (backgroundColor != src.backgroundColor) return false;
		return true;
	}
	TF& operator=(const TF & src) {
		this->tfResolution = src.tfResolution;
		if (this->tfDrawArray) delete [] this->tfDrawArray;
		this->tfDrawArray = new float[(int)(this->tfResolution)];
		for (int i = 0; i < (int)(this->tfResolution); ++i) this->tfDrawArray[i] = src.tfDrawArray[i];
		this->gaussianObjectArray = src.gaussianObjectArray;
		this->tfColorTick = src.tfColorTick;
		this->combineMode = src.combineMode;
		this->tranSliderValue = src.tranSliderValue;
		this->backgroundMesh = src.backgroundMesh;
		this->backgroundColor = src.backgroundColor;
		return *this;
	}
	void load(const QString & filename) {
		if (filename.isEmpty() || filename.isNull()) return;
		QFile file(filename);
		if(!file.open(QIODevice::ReadOnly)) return;
		if(!file.isReadable()) return;
		load(file);
		file.close();
	}
	void load(QFile & file) {
		file.read((char*)&tfResolution, 4);
		if (tfDrawArray) delete [] tfDrawArray;
		tfDrawArray = new float[(int)tfResolution];
		file.read((char*)tfDrawArray, (int)tfResolution*4);
		int size;
		file.read((char*)&size, 4);
		gaussianObjectArray.clear();
		double t[3];
		for(int i = 0; i < size; i++) {
			file.read((char*)t, 24);
			GaussianObject obj(t[0], t[1], t[2], tfResolution);
			obj.update();
			gaussianObjectArray.push_back(obj);
		}
		file.read((char*)&size, 4);
		float resX;
		tfColorTick.clear();
		QColor c;
		for(int i = 0; i < size; i++) {
			file.read((char*)&resX, 4);
			file.read((char*)t, 24);
			c.setRgbF(t[0], t[1], t[2]);
			TFColorTick tick(resX, c);
			tfColorTick.push_back(tick);
		}
		file.read((char*)&combineMode, 4);
		file.read((char*)&tranSliderValue, 4);
		file.read((char*)&backgroundMesh, 4);
		file.read((char*)t, 24);
		backgroundColor.setRgbF(t[0], t[1], t[2]);
	}
	void save(const QString & filename) const {
		if (filename.isEmpty() || filename.isNull()) return;
		QFile file(filename);
		if(!file.open(QIODevice::WriteOnly)) return;
		if(!file.isWritable()) return;
		file.close();
	}
	void save(QFile & file) const {
		file.write((char*)&tfResolution, 4);
		file.write((char*)tfDrawArray, tfResolution*4);
		int size = gaussianObjectArray.size();
		file.write((char*)&size, 4);
		for(int i = 0; i < size; i++) {
			file.write((char*)&(gaussianObjectArray[i].m_mean), 8);
			file.write((char*)&(gaussianObjectArray[i].m_sigma), 8);
			file.write((char*)&(gaussianObjectArray[i].m_heightFactor), 8);
		}
		size = tfColorTick.size();
		file.write((char*)&size, 4);
		double t[3];
		for(int i = 0; i < size; i++) {
			file.write((char*)&(tfColorTick[i].m_resX), 4);
			tfColorTick[i].m_color.getRgbF(t, t+1, t+2);
			file.write((char*)t, 24);
		}
		file.write((char*)&combineMode, 4);
		size = tranSliderValue;
		file.write((char*)&size, 4);
		file.write((char*)&backgroundMesh, 4);
		backgroundColor.getRgbF(t, t+1, t+2);
		file.write((char*)t, 24);
	}
	float tfResolution;
	float * tfDrawArray;
	QVector<GaussianObject> gaussianObjectArray;
	QVector<TFColorTick> tfColorTick;
	int combineMode;
	int tranSliderValue;
	int backgroundMesh;
	QColor backgroundColor;
};
QDataStream & operator<<(QDataStream &, const GaussianObject &);
QDataStream & operator>>(QDataStream &, GaussianObject &);
QDataStream & operator<<(QDataStream &, const TFColorTick &);
QDataStream & operator>>(QDataStream &, TFColorTick &);
QDataStream & operator<<(QDataStream &, const TF &);
QDataStream & operator>>(QDataStream &, TF &);


class TFMappingEditor : public QWidget{
	Q_OBJECT
public:
	TFMappingEditor(QWidget *parent = 0);
	~TFMappingEditor();

	int x1,x2;
	float y1,y2;

	QLineEdit	*x1edit,*x2edit,*y1edit,*y2edit;
public slots:
	void updateButtonPressed();

signals:
	void mappingChanged(int x1,float y1, int x2,float y2);
};


class QTFPanel : public QTFAbstractPanel
{
	Q_OBJECT
public:
	QTFPanel(int transferWidth, QWidget *parent=0);
	~QTFPanel();

	int			getTFColorMapResolution() {return m_tfColorMapResoultion;}
	float*		getTFColorMap() {return m_tfColorMap;}
	float*		getXYMappingMap(){return (turnOnNonlinearXYMapping)?m_xyMappingMap:m_xyLinearMappingMap;}
	bool		getIsTurnOnMapping(){return turnOnNonlinearXYMapping;}

	float		getTFResolution() {return m_tfResolution;}
	float*		getTFDrawArray() {return m_tfDrawArray;}
	QVector<GaussianObject>* getGaussians() {return &m_gaussianObjectArray;}
	QVector<TFColorTick>*	getColors() {return &m_tfColorTick;}
	QVector<ZeroRange>*	getZeros() { return &m_zeroRangesArray; }
	void		changeSelectedTickColor(QColor &cr);


	void		saveSettings(TF&);
	void		loadSettings(TF&);
	QIODevice&	saveFile(QIODevice&);
	QIODevice&	openFile(QIODevice&);
	QFile&		saveFileVisTransport(QFile&);
	QFile&		openFileVisTransport(QFile&);
	void		setRange(float rmin, float rmax);

	QImage * m_tfImage;
	float * computeColorMapFromTF(TF &, float * = NULL);

private:
	int			m_colorTickHeight;
	void		initLayout();
	bool		m_isLeftClick,m_isRightClick;

	QVector<TFColorTick>	m_tfColorTick; // at the beginning, 2 colors (left and right)
	QPoint					m_nowPoint,m_lastPoint;

	QRect		m_panelArea, m_panelAreaClick, m_colorTickArea;
	float		m_tfResolution;
	float		m_tfPt2ResFactor,m_tfRes2PtFactor;
	float		*m_tfDrawArray;
	bool		m_ifOnDrawing,m_ifOnDrawColorTick;
	float		yval, lastyval;
	int		m_nowResIndex,m_lastResIndex;
	void		interpolateResPoint(bool ctrl=false);
	int		clickOnTicks(QPoint	&pt);
	int		m_clickedTicks;
	int		m_backgroundMesh; // 0: none, 1: dot mesh

	QSlider		*m_vertTranSlider;
	QString		m_vertTranPercetangeString;
	bool		m_vertStringNeedUpdate;
	int		m_vertTranSliderRange;

	QSize		m_buttonSize;
	QPushButton	*m_settingButton,*m_functionButton,*m_mappingButton;
	QMenu		*m_functionMenu,*m_settingMenu;
	QAction		*changeBM2NoneAct,*changeBM2DotLineAct,*changeCombine2OrAct,*changeCombine2AndAct,
				*openFileAct,*saveFileAct,*toggleInstantAct, *toggleDrawLabels,
				*toggleDrawXYMapping;

	//double		gaussianGen(double x,double mean,double sigma);
	QVector<GaussianObject>	m_gaussianObjectArray;
	QVector<ZeroRange> m_zeroRangesArray;
	int		m_clickedObjectControlBox;
	int		clickOnObjctControlBox(QPoint	&pt);
	int		clickOnObjctControlBox(QPoint	&pt,int &side);
	bool		m_ifOnMovingObjectControlBox, m_instant;
	int		m_objectControlBoxSide; // 1:right, 2:left

	bool		m_ifCTRLPressed, m_ifSHIFTPressed;

	int		m_tfColorMapResoultion;
	int		m_combineMode;
	float		*m_tfColorMap;
	void		updateTFColorMap();
	void		updatePanelImage();
	void		generateZeroRanges();
	float		alphaValue(int);
	float		rangemin, rangemax;
	bool		drawLabels;
	bool		turnOnNonlinearXYMapping;

	QVector<float>	m_xyMapping,m_xyLinearMapping;
	float	*m_xyMappingMap,*m_xyLinearMappingMap;
	void	updateXYLinearMappingMap();
	void	updateXYMappingMap();
	TFMappingEditor *m_mappingEditor;
	//NLTFEditor		*m_nltfEditor;
	int     m_mappingRes;


protected:
	void realPaintEvent(QPaintEvent *);
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
	void mouseDoubleClickEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);
	void resizeEvent(QResizeEvent *);
//	void keyPressEvent(QKeyEvent *);
//	void keyReleaseEvent(QKeyEvent *);

public slots:
	void vertTranSliderChange(int);
	void functionButtonPressed();
	void settingButtonPressed();
	void mappingButtonPressed();

	void changeBM2None();
	void changeBM2DotLine();
	void changeCombine2And();
	void changeCombine2Or();
	void toggleInstant();
	void togglePositionLabel(bool);
	void toggleXYMapping();

	void openFile();
	void saveFile();

	void tfExternalChanged(float, float*, QVector<GaussianObject>*, QVector<TFColorTick>*, bool);
	void generateMapping(int x1,float y1,int x2, float y2);

signals:
	void tfColorMapChange();
	void tfChanged(float*, bool = true);
	void tfMappingChanged(float*, float*, bool = true);
};

#endif

