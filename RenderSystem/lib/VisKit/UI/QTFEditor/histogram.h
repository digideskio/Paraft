#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

#include <iostream>

using namespace std;

#include <QObject>
#include <QFile>

class Histogram : public QObject {
	Q_OBJECT
	
	double min, max;
	size_t n;
	unsigned int* bin;
	
	public:
		Histogram(size_t l=200);
		~Histogram();
		
		void setMinMax(double,double);
		
		double getMin() const;
		double getMax() const;
		void clear();
		
		void setLength(size_t l);
		size_t getLength();
		
		void increment(double);
		unsigned int& operator[](const unsigned int& i) const;

		void save(QFile& file);
		void load(QFile& file);

		Histogram& operator+=(const Histogram& rhs);
		Histogram& operator=(const Histogram& rhs);
		
	signals:
		void updated();
};


#endif
