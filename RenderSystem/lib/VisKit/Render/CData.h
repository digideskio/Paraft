#ifndef _CDATA_H_
#define _CDATA_H_

#include <QMap>
#include <QVector>
#include <QString>

enum BYTEORDER	{LITTEENDIAN, BIGENDIAN/*LITTLE_ENDIAN, BIG_ENDIAN*/};
enum FILETYPE	{BINARY, TEXT};
enum MESHFORMAT	{X,XY,XYZ}; // 1,2,3D
enum MESHATT	{UNSIGNED_8BIT, SIGNED_8BIT,  // 0
				 UNSIGNED_16BIT, SIGNED_16BIT,
				 UNSIGNED_32BIT, SIGNED_32BIT,
				 FLOATT, DOUBLEE};
struct CFileConfig{
	CFileConfig()
	{
		m_endian		= BIGENDIAN;
		m_fileType		= BINARY;
		m_meshFormat	= XYZ;
		m_meshAtt		= UNSIGNED_8BIT; // Byte
		m_totalStep		= 1;
		m_stepFileStart	= 1; // filename0001, 0002, 0003...-> file starts at 1
		m_stepDelta		= 1; // filename0001, 0003, 0005...-> delta = 2
		m_numDigits		= -1;
		m_ifClampMax	= false;
		m_ifClampMin	= false;
		m_clampMaxVal	= 1e+20;
		m_clampMinVal	= -1e+20;
		m_ifSetRange	= false;
		m_dataMaxVal	= 1e+20;
		m_dataMinVal	= -1E+20;
	}
	QString		m_filename;
	QString		m_filepattern;
	BYTEORDER	m_endian;
	FILETYPE	m_fileType;
	MESHFORMAT	m_meshFormat;
	MESHATT		m_meshAtt;
	size_t		m_totalStep;
	size_t		m_stepDelta;	
	size_t		m_stepFileStart;
	int			m_numDigits; // -1: no prefix digits for number
	bool		m_ifClampMax,m_ifClampMin;
	double		m_clampMaxVal,m_clampMinVal;
	bool		m_ifSetRange;
	double		m_dataMaxVal,m_dataMinVal;
};

class CData
{
public:
	CData(){m_currentStep=1;}
	~CData(){}

	size_t	m_filesize;

	CFileConfig		m_fileConfig;
	void			setFileConfig(const CFileConfig &c){m_fileConfig = c;}
	CFileConfig&	getFileConfig(){return m_fileConfig;}

	// scalar range
	float	m_scalarMin, m_scalarMax;
	float	*m_scalar; // everything is turned into float

	QString		getFilename(){return m_fileConfig.m_filename;}
	void		setFilename(QString fn){m_fileConfig.m_filename = fn;}
	QString		getFilePattern(){return m_fileConfig.m_filepattern;}
	void		setFilePattern(QString pn){m_fileConfig.m_filepattern = pn;}
	QString		getFilenameN(size_t s);
	QString		getNextFilenameN();
	QString		getPreviousFilenameN();
	void		readNextData();
	void		readPreviousData();
	void		setClampMax(bool a,double xv);
	void		setClampMin(bool a,double yv);

	virtual bool readDataN(size_t tstep=1);
	virtual bool readData(QString fn=QString(), int offset=0)=0;
	virtual int  getTotalDataEntry(int idx)=0;
	
	// multi steps
	size_t	m_currentStep; // start from 1
	virtual void	setCurrentStep(size_t s);
	size_t	getCurrentStep(){return m_currentStep;}
	size_t	getTotalSteps(){return m_fileConfig.m_totalStep;}
};


#endif

