//
// C++ Interface: shaderslider
//
// Description: 
//
//
// Author: Chris Ho <csho@ucdavis.edu>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef _SHADERSLIDER_H_
#define _SHADERSLIDER_H_

#include <QSlider>

class ShaderSlider : public QSlider {
	Q_OBJECT

	double rMin, rMax;
	int steps;
	public:
		ShaderSlider(QWidget* parent=0);
		ShaderSlider(double min, double max, int steps);

		void setMin(double min);
		void setMax(double max);
		void setSteps(int s);
	public slots:
		void setNewValue(double);
	protected:
		void sliderChange(SliderChange);
	signals:
		void newValue(double=1.);
};




#endif



