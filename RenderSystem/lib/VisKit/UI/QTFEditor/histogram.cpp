#include "histogram.h"
Histogram::Histogram(size_t l):n(l > 0 ? l : 200), bin(new unsigned int[l > 0 ? l : 200]) {
	memset(bin, 0, n*4);
}

void Histogram::setMinMax(double minrange, double maxrange) {
	min = minrange < maxrange ? minrange : maxrange;
	max = maxrange > minrange ? maxrange : minrange;
	
	memset(bin, 0, n*4);
}

Histogram::~Histogram() {
	delete [] bin;
}

void Histogram::setLength(size_t l) {
	if(l > n) {
		delete [] bin;
		bin = new unsigned int[l];
	}
	n = l;
	memset(bin, 0, l*4);
}

size_t Histogram::getLength() {
	return n;
}

double Histogram::getMax() const {
	return max;
}

double Histogram::getMin() const {
	return min;
}

unsigned int& Histogram::operator[](const unsigned int& i) const {
	return bin[i];
}

void Histogram::increment(double i) {
	if( i < min || i > max )
		return;
	
	bin[(size_t)((i - min)/(max - min)*((double)n - 1) + 0.5)]++;
	emit updated();
}

void Histogram::clear() {
	memset(bin, 0, n*4);
}

void Histogram::save(QFile& file) {
	if(!file.isWritable()) {
		return;
	}
	long long l = static_cast<long long>(n);
	file.write((char*)&l, 8);
	file.write((char*)&min, 8);
	file.write((char*)&max, 8);
	file.write((char*)bin, n*4);
}


void Histogram::load(QFile& file) {
	if(!file.isReadable()) {
		return;
	}
	long long l;
	file.read((char*)&l, 8);
	file.read((char*)&min, 8);
	file.read((char*)&max, 8);
	if(l != n) {
		delete [] bin;
		bin = new unsigned int[l];
		n = l;
	}
	file.read((char*)bin, n*4);
	emit updated();
}

Histogram& Histogram::operator+=(const Histogram& rhs) {
	if(n != rhs.n || min != rhs.min || max != rhs.max) {
		return *this;
	}
	for(size_t i = 0; i < n; i++) {
		bin[i] += rhs.bin[i];
	}
	return *this;
}

Histogram& Histogram::operator=(const Histogram& rhs) {
	this->setLength(rhs.n);
	this->setMinMax(rhs.min, rhs.max);
	memcpy(this->bin, rhs.bin, sizeof(unsigned int) * n);
	return *this;
}
