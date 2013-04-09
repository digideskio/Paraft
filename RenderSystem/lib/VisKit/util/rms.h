//
// C++ Interface: rms
//
// Description: 
//
//
// Author: Chris Ho <csho@ucdavis.edu>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef _RMS_H_
#define _RMS_H_

/*! RootMeanSquare
 * Does some simple statistics info for a set of data (single-precision floats)
 * Gives stuff like the min, max, mean, RMS, and STD.
 */

#include <QFile>
#include <QString>

template <typename T>
class RMS {
	double mean;
	T min;
	T max;
	double sigma;
	float cutoff;
	double total;
	long long count;
	double rms;
	double std;
	double rmstotal;
	
	public:

		RMS();
		
		/*! adds data to the total
		 * \param d data set
		 * \param c the count
		 * \param stride how far apart each value is (default=1)
		 */
		void addData(T* d, int c, int stride=1);
		
		//! retuns the standard deviation
		double getSigma();
		
		//! returns the mean
		double getMean();
		
		//! returns the minimum of the data set
		T getMin();
		
		//! returns the maximum of the data set
		T getMax();
		
		//! returns the RMS of the data set
		double getRMS();
		
		void reset();
		
		QFile& saveInfo(QFile&);
		QFile& readInfo(QFile&);
		QString toString();
};

#include "rms.inl"

#endif


