#ifndef _CIMAGEDATA_H_
#define _CIMAGEDATA_H_

#include "CStructuredMeshData.h"
#include <QString>

class CImageData : public CStructuredMeshData
{
public:
	CImageData();
	~CImageData();

	virtual bool readData(QString fn=QString(), int offset=0);
	
	virtual void setOrgDimensions(int dx,int dy,int dz=1,float sx=1.f, float sy=1.f, float sz=1.f)
	{	m_orgDim[0]=dx;	m_orgDim[1]=dy;	m_orgDim[2]=dz; m_scale[0]=sx; m_scale[1]=sy; m_scale[2]=sz;}
	virtual void setSamplingFactor(int sx,int sy,int sz=1)
	{	m_sf[0]=sx;m_sf[1]=sy;m_sf[2]=sz;}
};

#endif
