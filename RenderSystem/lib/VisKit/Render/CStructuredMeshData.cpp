#include "CStructuredMeshData.h"
#include <QString>
#include <QFile>
#include <QDataStream>
#include <QFileInfo>
#include <cmath>
#include <iostream>
#include <fstream>
using namespace std;

CStructuredMeshData::CStructuredMeshData()
{
	m_rawData = NULL;
	m_sf[0] = m_sf[1] = m_sf[2] = 1;
}
CStructuredMeshData::~CStructuredMeshData()
{
	if(m_rawData) delete [] m_rawData;
}
bool CStructuredMeshData::readData(QString fn, int offset){

	if(!fn.isEmpty()) // if fn== NULL, use m_filename, if not, replace m_filename
		m_fileConfig.m_filename = fn;

	if(m_rawData) delete [] m_rawData;

	if(!QFile::exists(m_fileConfig.m_filename)) {
		qDebug("Missing Volume Data File: %s", m_fileConfig.m_filename.toLocal8Bit().constData());
		return false;
//		exit(1);
	}

	QFile dataFile(m_fileConfig.m_filename);
	if(!dataFile.open(QIODevice::ReadOnly)) 
		return false;

	QDataStream	inpData(&dataFile);
	dataFile.seek(offset);

	if(m_fileConfig.m_endian == BIGENDIAN)
		inpData.setByteOrder(QDataStream::BigEndian);
	else
		inpData.setByteOrder(QDataStream::LittleEndian);

	// sampled dimensions
	m_newDim[0] = m_orgDim[0]/m_sf[0];
	m_newDim[1] = m_orgDim[1]/m_sf[1];
	m_newDim[2] = m_orgDim[2]/m_sf[2];
	// 
	size_t m_dimX = m_newDim[0];
	size_t m_dimY = m_newDim[1];
	size_t m_dimZ = m_newDim[2];

	//// traditional c style (no flexibility)
	//m_rawData = new float[m_dimX*m_dimY*m_dimZ];
	//FILE *ww = fopen(m_fileConfig.m_filename.toAscii(),"rb");
	//fread(m_rawData,sizeof(short),m_orgDim[0]*m_orgDim[1]*m_orgDim[2],ww);

	void*	rawData;	
	size_t	realFileSizeInBytes;
	//size_t	realFileSizeInBytes = m_dimX*m_dimY*m_dimZ*sizeof(char);//
	if(m_fileConfig.m_meshAtt == UNSIGNED_8BIT || m_fileConfig.m_meshAtt == SIGNED_8BIT)	
		realFileSizeInBytes = m_dimX*m_dimY*m_dimZ*sizeof(char);
	else if(m_fileConfig.m_meshAtt == UNSIGNED_16BIT || m_fileConfig.m_meshAtt == SIGNED_16BIT)	
		realFileSizeInBytes = m_dimX*m_dimY*m_dimZ*2*sizeof(short); // 2 bytes
	else if(m_fileConfig.m_meshAtt == UNSIGNED_32BIT || m_fileConfig.m_meshAtt == SIGNED_32BIT)	
		realFileSizeInBytes = m_dimX*m_dimY*m_dimZ*4*sizeof(char); // 4 bytes
	else if(m_fileConfig.m_meshAtt == FLOATT)	
		realFileSizeInBytes = m_dimX*m_dimY*m_dimZ*sizeof(float);
	else if(m_fileConfig.m_meshAtt == DOUBLEE)	
		realFileSizeInBytes = m_dimX*m_dimY*m_dimZ*sizeof(double);

	rawData = (void*)new char[realFileSizeInBytes];
	inpData.readRawData((char*)rawData,realFileSizeInBytes);
	dataFile.close();

	m_rawData = new float[m_dimX*m_dimY*m_dimZ];
	unsigned int index = 0;

	if(m_fileConfig.m_meshAtt == FLOATT){
		for(size_t a=0;a<m_dimZ;a=a+m_sf[2])
		{
			for(size_t b=0;b<m_dimY;b=b+m_sf[1])
			{
				for(size_t c=0;c<m_dimX;c=c+m_sf[0])
				{
					index = c + b*m_dimX + a*m_dimX*m_dimY;
					m_rawData[index] = (float)((float*)rawData)[index];
				}
			}
		}
	}
	else if(m_fileConfig.m_meshAtt == UNSIGNED_8BIT || m_fileConfig.m_meshAtt == SIGNED_8BIT){
		for(size_t a=0;a<m_dimZ;a=a+m_sf[2])
		{
			for(size_t b=0;b<m_dimY;b=b+m_sf[1])
			{
				for(size_t c=0;c<m_dimX;c=c+m_sf[0])
				{
					index = c + b*m_dimX + a*m_dimX*m_dimY;
					m_rawData[index] = (float)((unsigned char*)rawData)[index];
				}
			}
		}
	}
	else if(m_fileConfig.m_meshAtt == UNSIGNED_16BIT || m_fileConfig.m_meshAtt == SIGNED_16BIT){
		for(size_t a=0;a<m_dimZ;a=a+m_sf[2])
		{
			for(size_t b=0;b<m_dimY;b=b+m_sf[1])
			{
				for(size_t c=0;c<m_dimX;c=c+m_sf[0])
				{
					index = c + b*m_dimX + a*m_dimX*m_dimY;
					if(m_fileConfig.m_meshAtt == UNSIGNED_16BIT) {
						unsigned short value;
						if(m_fileConfig.m_endian == BIGENDIAN) {
							*((char*)(&value)+0) = ((char*)rawData)[index*2+0];
							*((char*)(&value)+1) = ((char*)rawData)[index*2+1];
						}
						else if(m_fileConfig.m_endian == LITTEENDIAN) {
							*((char*)(&value)+0) = ((char*)rawData)[index*2+1];
							*((char*)(&value)+1) = ((char*)rawData)[index*2+0];
						}
						m_rawData[index] = (float)(value);
					}
					else {
						short value;
						if(m_fileConfig.m_endian == BIGENDIAN) {
							*((char*)(&value)+0) = ((char*)rawData)[index*2+0];
							*((char*)(&value)+1) = ((char*)rawData)[index*2+1];
						}
						else if(m_fileConfig.m_endian == LITTEENDIAN) {
							*((char*)(&value)+0) = ((char*)rawData)[index*2+1];
							*((char*)(&value)+1) = ((char*)rawData)[index*2+0];
						}
						m_rawData[index] = (float)(value);
					}
//					if(m_fileConfig.m_meshAtt == UNSIGNED_16BIT)
//						m_rawData[index] = (float)((unsigned short*)rawData)[index];
//					else 
//						m_rawData[index] = (float)((short*)rawData)[index];
				}
			}
		}
	}

	delete [] (float*)rawData;

	// normalize data
	normalilze();

	// find out the max dim
	size_t maxDim = max(m_dimX,max(m_dimY,m_dimZ));
	m_scaledDim[0] = (float)m_dimX/maxDim;
	m_scaledDim[1] = (float)m_dimY/maxDim;
	m_scaledDim[2] = (float)m_dimZ/maxDim;
	m_scaledDim[0] *= m_scale[0];
	m_scaledDim[1] *= m_scale[1];
	m_scaledDim[2] *= m_scale[2];
	return true;
}
int CStructuredMeshData::getTotalDataEntry(int idx)
{
	size_t m_dimX = m_newDim[0];
	size_t m_dimY = m_newDim[1];
	size_t m_dimZ = m_newDim[2];

	int total = m_orgDim[0]*m_orgDim[1]*m_orgDim[2];
	if(idx == 0)
		; // original total entry
	else if(idx == 1) // sampled data
		total = m_dimX*m_dimY*m_dimZ;

	return total;
}
void CStructuredMeshData::normalilze()
{
	size_t m_dimX = m_newDim[0];
	size_t m_dimY = m_newDim[1];
	size_t m_dimZ = m_newDim[2];

	size_t dataSize = m_dimX*m_dimY*m_dimZ;

	if(m_fileConfig.m_ifClampMax){
		for(size_t x=0;x<dataSize;++x){
			if(m_rawData[x] > m_fileConfig.m_clampMaxVal)
				m_rawData[x] = m_fileConfig.m_clampMaxVal;
		}
	}
	if(m_fileConfig.m_ifClampMin){
		for(size_t x=0;x<dataSize;++x){
			if(m_rawData[x] < m_fileConfig.m_clampMinVal)
				m_rawData[x] = m_fileConfig.m_clampMinVal;
		}
	}	

	if(!m_fileConfig.m_ifSetRange){	
		m_fileConfig.m_dataMaxVal = -1E20;
		m_fileConfig.m_dataMinVal = 1E20;
		for(size_t x=0;x<dataSize;x++)
		{	
			if(m_fileConfig.m_dataMaxVal < m_rawData[x])
				m_fileConfig.m_dataMaxVal = m_rawData[x];
			if(m_fileConfig.m_dataMinVal > m_rawData[x])
				m_fileConfig.m_dataMinVal = m_rawData[x];
		}
	}
	double range = m_fileConfig.m_dataMaxVal-m_fileConfig.m_dataMinVal;
	for(size_t s=0;s<dataSize;s++)
		m_rawData[s] = (m_rawData[s]-m_fileConfig.m_dataMinVal)/range;
}
void CStructuredMeshData::outputData(char *filename,int type){
	if(type == 0){ // binary raw
		ofstream outD(filename,ios::out | ios::binary);
		outD.write((char*)m_rawData,m_newDim[0]*m_newDim[1]*m_newDim[2]*sizeof(float));		
		outD.close();		
	}
	else if(type == 1){ // vtk structure rect
	}
	else{
		cout << "Unrecognized data type" << endl;
	}
}
CStructuredMeshData2D::CStructuredMeshData2D(){
	// z axis = 2, middle z = 50
	setSliceInfo(2,50);
}
CStructuredMeshData2D::~CStructuredMeshData2D(){
}
bool CStructuredMeshData2D::readData(QString fn,int offset){
	if(!fn.isEmpty()) // if fn== NULL, use m_filename, if not, replace m_filename
		m_fileConfig.m_filename = fn;
	
	if(m_rawData) delete [] m_rawData;
	
	if(!QFile::exists(m_fileConfig.m_filename)) {
		qDebug("Missing Volume Data File: %s", m_fileConfig.m_filename.toLocal8Bit().constData());
		return false;
//		exit(1);
	}
	
	QFile dataFile(m_fileConfig.m_filename);
	if(!dataFile.open(QIODevice::ReadOnly)) 
		return false;
	
	QDataStream	inpData(&dataFile);
	dataFile.seek(offset);
	
	if(m_fileConfig.m_endian == BIGENDIAN)
		inpData.setByteOrder(QDataStream::BigEndian);
	else
		inpData.setByteOrder(QDataStream::LittleEndian);
	
	// sampled dimensions
	m_newDim[0] = m_orgDim[0]/m_sf[0];
	m_newDim[1] = m_orgDim[1]/m_sf[1];
	m_newDim[2] = m_orgDim[2]/m_sf[2];
	// 
	size_t m_dimX = m_newDim[0];
	size_t m_dimY = m_newDim[1];
	size_t m_dimZ = m_newDim[2]; // always = 1
	
	size_t realx = m_realDim[0];
	size_t realy = m_realDim[1];
	size_t realz = m_realDim[2];
	
	//// traditional c style (no flexibility)
	//m_rawData = new float[m_dimX*m_dimY*m_dimZ];
	//FILE *ww = fopen(m_fileConfig.m_filename.toAscii(),"rb");
	//fread(m_rawData,sizeof(short),m_orgDim[0]*m_orgDim[1]*m_orgDim[2],ww);
	
	void*	rawData;	
	size_t	realFileSizeInBytes;
	//size_t	realFileSizeInBytes = m_dimX*m_dimY*m_dimZ*sizeof(char);//
	if(m_fileConfig.m_meshAtt == UNSIGNED_8BIT || m_fileConfig.m_meshAtt == SIGNED_8BIT)	
		realFileSizeInBytes = realx*realy*realz*sizeof(char);
	else if(m_fileConfig.m_meshAtt == UNSIGNED_16BIT || m_fileConfig.m_meshAtt == SIGNED_16BIT)	
		realFileSizeInBytes = realx*realy*realz*2*sizeof(short); // 2 bytes
	else if(m_fileConfig.m_meshAtt == UNSIGNED_32BIT || m_fileConfig.m_meshAtt == SIGNED_32BIT)	
		realFileSizeInBytes = realx*realy*realz*4*sizeof(char); // 4 bytes
	else if(m_fileConfig.m_meshAtt == FLOATT)	
		realFileSizeInBytes = realx*realy*realz*sizeof(float);
	else if(m_fileConfig.m_meshAtt == DOUBLEE)	
		realFileSizeInBytes = realx*realy*realz*sizeof(double);
	
	rawData = (void*)new char[realFileSizeInBytes];
	inpData.readRawData((char*)rawData,realFileSizeInBytes);
	dataFile.close();
	
	m_rawData = new float[m_dimX*m_dimY*m_dimZ];
	unsigned int index = 0;
	// only z slice (xy plane)
	unsigned int indexbase = m_sliceDist*m_dimX*m_dimY;
	unsigned int orgIndex = 0;
	if(m_fileConfig.m_meshAtt == FLOATT){
		for(size_t b=0;b<m_dimY;b=b+m_sf[1]){
			for(size_t c=0;c<m_dimX;c=c+m_sf[0]){
				orgIndex = c + b*m_dimX;
				index = orgIndex + indexbase;
				m_rawData[orgIndex] = (float)((float*)rawData)[index];
			}
		}
	}
	else if(m_fileConfig.m_meshAtt == UNSIGNED_8BIT || m_fileConfig.m_meshAtt == SIGNED_8BIT){
		for(size_t b=0;b<m_dimY;b=b+m_sf[1]){
			for(size_t c=0;c<m_dimX;c=c+m_sf[0]){
				orgIndex = c + b*m_dimX;
				index = orgIndex + indexbase;
				m_rawData[orgIndex] = (float)((unsigned char*)rawData)[index];
			}
		}
	}
	else if(m_fileConfig.m_meshAtt == UNSIGNED_16BIT || m_fileConfig.m_meshAtt == SIGNED_16BIT){
		for(size_t b=0;b<m_dimY;b=b+m_sf[1]){
			for(size_t c=0;c<m_dimX;c=c+m_sf[0]){
				orgIndex = c + b*m_dimX;
				index = orgIndex + indexbase;
				if(m_fileConfig.m_meshAtt == UNSIGNED_16BIT)
					m_rawData[orgIndex] = (float)((unsigned short*)rawData)[index];
				else 
					m_rawData[orgIndex] = (float)((short*)rawData)[index];
			}
		}
	}
	
	delete [] (float*)rawData;
	
	// normalize data
	normalilze();
	
	// find out the max dim
	size_t maxDim = max(m_dimX,max(m_dimY,m_dimZ));
	m_scaledDim[0] = (float)m_dimX/maxDim;
	m_scaledDim[1] = (float)m_dimY/maxDim;
	m_scaledDim[2] = (float)m_dimZ/maxDim;
	m_scaledDim[0] *= m_scale[0];
	m_scaledDim[1] *= m_scale[1];
	m_scaledDim[2] *= m_scale[2];
	return true;
}

