#include "CData.h"

void CData::setCurrentStep(size_t s){
	if(s <= m_fileConfig.m_totalStep && s > 0){
		m_currentStep = s;
	}
}
bool CData::readDataN(size_t tstep){
	if(tstep > m_fileConfig.m_totalStep || tstep < 1)
		return false;

	if(m_fileConfig.m_totalStep == 1)
		return readData();
	else{
		size_t fileIdx = m_fileConfig.m_stepFileStart+(tstep-1)*m_fileConfig.m_stepDelta;

		if(m_fileConfig.m_numDigits == -1)
			return readData(QString(m_fileConfig.m_filepattern).arg(fileIdx));
		else
			return readData(QString(m_fileConfig.m_filepattern).arg(fileIdx,m_fileConfig.m_numDigits,10,QLatin1Char('0')));
	}
}
QString CData::getFilenameN(size_t s){
	setCurrentStep(s);

	if(m_fileConfig.m_totalStep == 1)
		return m_fileConfig.m_filename;
	else{
		size_t fileIdx = m_fileConfig.m_stepFileStart+(s-1)*m_fileConfig.m_stepDelta;
		if(m_fileConfig.m_numDigits == -1)
			return QString(m_fileConfig.m_filepattern).arg(fileIdx);
		else
			return QString(m_fileConfig.m_filepattern).arg(fileIdx,m_fileConfig.m_numDigits,10,QLatin1Char('0'));
	}
}
QString CData::getNextFilenameN(){
	size_t temp = m_currentStep + 1;
	if(temp <= m_fileConfig.m_totalStep)
		return getFilenameN(temp);	
	return m_fileConfig.m_filename;
}
QString CData::getPreviousFilenameN(){
	size_t temp = m_currentStep - 1;
	if(temp >= 1)
		return getFilenameN(temp);
	return m_fileConfig.m_filename;
}
void CData::readNextData(){
	QString tempname = getNextFilenameN();
	readData(tempname);
}
void CData::readPreviousData(){
	QString tempname = getPreviousFilenameN();
	readData(tempname);
}
void CData::setClampMax(bool a,double xv){
	m_fileConfig.m_ifClampMax = a;
	m_fileConfig.m_clampMaxVal = xv;
}
void CData::setClampMin(bool a,double yv){
	m_fileConfig.m_ifClampMin = a;
	m_fileConfig.m_clampMinVal = yv;
}