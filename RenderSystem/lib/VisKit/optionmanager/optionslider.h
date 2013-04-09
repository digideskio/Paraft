#ifndef _OPTIONSLIDER_H_
#define _OPTIONSLIDER_H_

#include <QSlider>

class OptionSlider : public QSlider {
	Q_OBJECT

	double rMin, rMax;
	int steps;
	public:
		OptionSlider(double min, double max, int steps);
	public slots:
		void setNewValue(double);
	protected:
		void sliderChange(SliderChange);
	signals:
		void newValue(double);
};

#endif

