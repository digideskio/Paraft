//
// C++ Implementation: shaderslider
//
// Description: 
//
//
// Author: Chris Ho <csho@ucdavis.edu>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include "shaderslider.h"

ShaderSlider::ShaderSlider(QWidget* parent):QSlider(Qt::Horizontal, parent),
rMin(0), rMax(1), steps(20) {
	setRange(0, steps);
	setTickInterval(1);
}

ShaderSlider::ShaderSlider(double min, double max, int steps):QSlider(Qt::Horizontal), rMin(min), rMax(max - min), steps(steps) {
	setRange(0, steps);
	setTickInterval(1);
}

void ShaderSlider::setMin(double min) {
	rMax = rMax + rMin - min;
	rMin = min;
}

void ShaderSlider::setMax(double max) {
	rMax = max - rMin;
}

void ShaderSlider::setSteps(int s) {
	setRange(0, s);
	steps = s;
}


void ShaderSlider::sliderChange(SliderChange sliderchange) {
	if(!isSliderDown())
		return;
	emit newValue(rMin + (value()/(double)steps)*rMax);
	QSlider::sliderChange(sliderchange);
}

void ShaderSlider::setNewValue(double v) {
	setValue((int)((v - rMin)/rMax*steps));
}


