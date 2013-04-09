#include "CImageData.h"
#include <QString>
#include <QFile>
#include <QDataStream>
#include <QFileInfo>
#include <QImage>
#include <cmath>
#include <iostream>
#include <fstream>
using namespace std;

CImageData::CImageData()
{
	m_rawData = NULL;
	m_sf[0] = m_sf[1] = m_sf[2] = 1;
}
CImageData::~CImageData()
{
}

bool CImageData::readData(QString fn, int offset){

	if(!fn.isEmpty()) // if fn== NULL, use m_filename, if not, replace m_filename
		m_fileConfig.m_filename = fn;

	if(m_rawData) delete [] m_rawData;

	int m_dimX,m_dimY,m_dimZ;
	QImage rawImage;
	if(rawImage.load(fn)){
		m_scale[0] = 1.0;m_scale[1] = 1.0;m_scale[2] = 1.0;
		
		m_orgDim[0] = rawImage.width();
		m_orgDim[1] = rawImage.height();
		m_orgDim[2] = 1.0;

		m_newDim[0] = m_orgDim[0]/m_sf[0];
		m_newDim[1] = m_orgDim[1]/m_sf[1];
		m_newDim[2] = 1.0;
		
		m_dimX = m_newDim[0];
		m_dimY = m_newDim[1];
		m_dimZ = 1.0;

		int iwidth = m_orgDim[0];
		int iheight= m_orgDim[1];
		
		m_rawData = new float[m_dimX*m_dimY*m_dimZ*3]; // RGB
		
		//cout << "Color depth =" << m_rawData.depth() << endl;
		
		float maxx = -1e20;
		float minn = 1e20;
		int counter = 0;
		for(int y=(iheight-1);y>=0;--y){
			for(int x=0;x<iwidth;++x){
				m_rawData[3*counter  ] = (float)qRed(rawImage.pixel(x,y))/255.0f;
				m_rawData[3*counter+1] = (float)qGreen(rawImage.pixel(x,y))/255.0f;
				m_rawData[3*counter+2] = (float)qBlue(rawImage.pixel(x,y))/255.0f;
				
				if(maxx < m_rawData[3*counter]) maxx = m_rawData[3*counter];
				if(maxx < m_rawData[3*counter+1]) maxx = m_rawData[3*counter+1];
				if(maxx < m_rawData[3*counter+2]) maxx = m_rawData[3*counter+2];

				if(minn > m_rawData[3*counter]) minn = m_rawData[3*counter];
				if(minn > m_rawData[3*counter+1]) minn = m_rawData[3*counter+1];
				if(minn > m_rawData[3*counter+2]) minn = m_rawData[3*counter+2];
				
				counter++;
			}
		}	
		cout << rawImage.depth() << endl;
	//	cout << rawImage.numColors() << endl;
		cout << maxx << "," << minn << endl;
		// find out the max dim
		size_t maxDim = max(m_dimX,m_dimY);
		m_scaledDim[0] = (float)m_dimX/(float)maxDim;
		m_scaledDim[1] = (float)m_dimY/(float)maxDim;
		m_scaledDim[2] = 1;
		m_scaledDim[0] *= m_scale[0];
		m_scaledDim[1] *= m_scale[1];
		m_scaledDim[2] *= 1;
		
		return true;
	}

	return false;
}