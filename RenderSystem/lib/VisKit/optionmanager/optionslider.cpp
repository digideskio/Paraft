#include "optionslider.h"

OptionSlider::OptionSlider(double min, double max, int steps):QSlider(Qt::Horizontal), rMin(min), rMax(max - min), steps(steps) {
	setRange(0, steps);
	setTickInterval(1);
}

void OptionSlider::sliderChange(SliderChange sliderchange) {
	if(!isSliderDown())
		return;
	emit newValue(rMin + (value()/(double)steps)*rMax);
	QSlider::sliderChange(sliderchange);
}

void OptionSlider::setNewValue(double v) {
	setValue((int)((v - rMin)/rMax*steps));
}


