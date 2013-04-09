#ifndef _CSTRUCTUREDMESHDATA_H_
#define _CSTRUCTUREDMESHDATA_H_

#include "CData.h"
#include <QString>

class CStructuredMeshData : public CData
{
public:
	CStructuredMeshData();
	~CStructuredMeshData();

	virtual bool readData(QString fn=QString(), int offset=0);
	virtual int  getTotalDataEntry(int idx=0);
	// for structural mesh
	size_t  m_orgDim[3];	// mesh dimensions (x,y,z)
	size_t	m_sf[3];		// sampling factors
	size_t	m_newDim[3];	// mesh dimensions after sampling
	float	m_scaledDim[3]; 
	float	m_scale[3];

	float	*m_rawData;

	virtual void	setOrgDimensions(int dx,int dy,int dz,float sx=1.f, float sy=1.f, float sz=1.f)
	{	m_orgDim[0]=dx;	m_orgDim[1]=dy;	m_orgDim[2]=dz; m_scale[0]=sx; m_scale[1]=sy; m_scale[2]=sz;}
	virtual void	setSamplingFactor(int sx,int sy,int sz)
	{	m_sf[0]=sx;m_sf[1]=sy;m_sf[2]=sz;}

	size_t*	getOrgDataDimensions(){return m_orgDim;}
	size_t*	getSamplingFactor(){return m_sf;}
	size_t*	getNewDataDimensions(){return m_newDim;}
	
	void	outputData(char *filename,int type = 0);

	virtual void	normalilze();
};

class CStructuredMeshData2D : public CStructuredMeshData
{
public:
	CStructuredMeshData2D();
	~CStructuredMeshData2D();
	
	size_t  m_realDim[3];
	size_t  m_sliceAxis; // 0-x,1-y,2-z
	size_t  m_sliceDist; 
	
	virtual void setSliceInfo(size_t ax, size_t dist){
		m_sliceAxis = ax;
		m_sliceDist = dist;
	}
	virtual bool readData(QString fn=QString(), int offset=0);
	
	virtual void setOrgRealDimensions(int dx,int dy,int dz,
									  int rx,int ry,int rz,
									  float sx=1.f, float sy=1.f, float sz=1.f){	
		m_orgDim[0]=dx;	m_orgDim[1]=dy;	m_orgDim[2]=dz; 
		m_realDim[0] = rx;m_realDim[1] = ry;m_realDim[2] = rz;
		m_scale[0]=sx; m_scale[1]=sy; m_scale[2]=sz;}
};

#endif
