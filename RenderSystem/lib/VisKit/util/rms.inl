//
// C++ Implementation: rms
//
// Description: 
//
//
// Author: Chris Ho <csho@ucdavis.edu>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <cmath>
#include <limits>

template <typename T>
RMS<T>::RMS(): total(0), count(0), rms(0), std(0), rmstotal(0) {
	min = std::numeric_limits<T>::max();
	max = std::numeric_limits<T>::min();
}

template <typename T>
void RMS<T>::reset() {
	total = 0;
	count = 0;
	rms = 0;
	std = 0;
	rmstotal = 0;
	min = std::numeric_limits<T>::max();
	max = std::numeric_limits<T>::min();
}

template <typename T>
double RMS<T>::getSigma() {
	mean = total/count;
	rms = rmstotal/count;
	std = rms - mean*mean;
	sigma = sqrt(std);
	return sigma;
}

template <typename T>
double RMS<T>::getMean() {
	mean = total/count;
	return mean;
}

template <typename T>
T RMS<T>::getMin() {
	return min;
}

template <typename T>
T RMS<T>::getMax() {
	return max;
}

template <typename T>
double RMS<T>::getRMS() {
	rms = sqrt(rmstotal/count);
	return rms;
}

template <typename T>
void RMS<T>::addData(T* data, int c, int stride) {
	count += c;
	for(int i = 0; i < c*stride; i += stride) {
		total += data[i];
		rmstotal += data[i]*data[i];
		min = min < data[i] ? min : data[i];
		max = max > data[i] ? max : data[i];
	}
}

template <typename T>
QFile& RMS<T>::saveInfo(QFile& file) {
	file.write((char*)&min, sizeof(T));
	file.write((char*)&max, sizeof(T));
	file.write((char*)&mean, 8);
	file.write((char*)&rms, 8);
	file.write((char*)&std, 8);
	file.write((char*)&total, 8);
	file.write((char*)&count, 8);
	file.write((char*)&rmstotal, 8);
	return file;
}
template <typename T>
QFile& RMS<T>::readInfo(QFile& file) {
	file.read((char*)&min, sizeof(T));
	file.read((char*)&max, sizeof(T));
	file.read((char*)&mean, 8);
	file.read((char*)&rms, 8);
	file.read((char*)&std, 8);
	file.read((char*)&total, 8);
	file.read((char*)&count, 8);
	file.read((char*)&rmstotal, 8);
	return file;
}
template <typename T>
QString RMS<T>::toString() {
	return QString("min: %4 max: %5 mean: %6 rms: %7 sigma: %8").arg(
			QString::number(getMin()), QString::number(getMax()),
			QString::number(getMean()), QString::number(getRMS()),
			QString::number(getSigma()));
}





