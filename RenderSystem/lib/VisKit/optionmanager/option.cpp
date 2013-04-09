#include "option.h"
#include <cstdarg>
#include <QGridLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QCheckBox>
#include "optionslider.h"

Option::Option(int numParams, const QString& name, OptionType otype)
	:numParams(numParams), name(name), otype(otype), options(0), sliders(numParams), modifiable(false) {
	for(int i = 0; i < numParams; i++) {
		sliders[i] = 0;
	}
}

void Option::setFieldNames(const char* first, ...) {
	fields.clear();
	fields.push_back(tr(first));
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		fields.push_back(tr(va_arg(ap, const char*)));
	}
	va_end(ap);
}

void Option::setSteps(int s) { //TODO
	steps = s;
}

void OptionInt::setRanges(int first, ...) {
	va_list ap;
	va_start(ap, first);
	ranges[0] = first;
	ranges[1] = va_arg(ap, int);
	for(int i = 1; i < numParams; i++) {
		ranges[i*2] = va_arg(ap, int);
		ranges[i*2 + 1] = va_arg(ap, int);
	}
	va_end(ap);
}

void OptionFloat::setRanges(float first, ...) {
	va_list ap;
	va_start(ap, first);
	ranges[0] = first;
	ranges[1] = (float)va_arg(ap, double);
	for(int i = 1; i < numParams; i++) {
		ranges[i*2] = (float)va_arg(ap, double);
		ranges[i*2 + 1] = (float)va_arg(ap, double);
	}
	va_end(ap);
}

void OptionDouble::setRanges(double first, ...) {
	va_list ap;
	va_start(ap, first);
	ranges[0] = first;
	ranges[1] = va_arg(ap, double);
	for(int i = 1; i < numParams; i++) {
		ranges[i*2] = va_arg(ap, double);
		ranges[i*2 + 1] = va_arg(ap, double);
	}
	va_end(ap);
}

void OptionInt::setValues(int first, ...) {
	vars[0] = first;
	if(numParams == 1)
		return;
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		vars[i] = va_arg(ap, int);
	}
	va_end(ap);
}

void OptionFloat::setValues(float first, ...) {
	vars[0] = first;
	if(numParams == 1)
		return;
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		vars[i] = (float)va_arg(ap, double);
	}
	va_end(ap);
}

void OptionDouble::setValues(double first, ...) {
	vars[0] = first;
	if(numParams == 1)
		return;
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		vars[i] = (float)va_arg(ap, double);
	}
	va_end(ap);
}


QGroupBox* OptionInt::getOptions() {
	if(!modifiable)
		return 0;
	
	if(options) {
		return options;
	}
	
	options = new OptionGroupBox(getName());
	QGridLayout *layout = new QGridLayout();
	for(int i = 0; i < numParams; i++) {
		if(fields.size() > i) {
			layout->addWidget(new QLabel(fields[i]), i, 0);
		}
		spins[i] = new QSpinBox();
		if(ranges[i*2] != ranges[i*2 + 1]) {
			spins[i]->setRange(ranges[2*i], ranges[2*i + 1]);
		}
		if(otype & SpinBox)
			layout->addWidget(spins[i], i, 2);
		
		if(otype & Slider) {
			if(ranges[i*2] != ranges[i*2+1]) {
				sliders[i] = new QSlider(Qt::Horizontal);
				sliders[i]->setRange(ranges[i*2], ranges[i*2 + 1]);
				connect(sliders[i], SIGNAL(sliderMoved(int)),
						spins[i], SLOT(setValue(int)));
				connect(spins[i], SIGNAL(valueChanged(int)),
						sliders[i], SLOT(setValue(int)));
				layout->addWidget(sliders[i], i, 1);
			}
		}
	}
	options->setLayout(layout);
	connect(options, SIGNAL(shown()),
			this, SLOT(setOptions()));
	return options;
}

QGroupBox* OptionFloat::getOptions() {
	if(!modifiable)
		return 0;
	
	if(options) {
		return options;
	}
	
	options = new OptionGroupBox(getName());
	QGridLayout *layout = new QGridLayout();
	for(int i = 0; i < numParams; i++) {
		if(fields.size() > i) {
			layout->addWidget(new QLabel(fields[i]), i, 0);
		}
		spins[i] = new QDoubleSpinBox();
		if(ranges[i*2] != ranges[i*2 + 1]) {
			spins[i]->setRange(ranges[2*i], ranges[2*i + 1]);
		}
		if(otype & SpinBox) 
			layout->addWidget(spins[i], i, 2);
		if(otype & Slider) {
			if(ranges[i*2] != ranges[i*2+1]) { //range required for slider
				sliders[i] = new OptionSlider(ranges[i*2], ranges[i*2+1], steps);
				connect((OptionSlider*)sliders[i], SIGNAL(newValue(double)),
							spins[i], SLOT(setValue(double)));
				connect(spins[i], SIGNAL(valueChanged(double)),
						(OptionSlider*)sliders[i], SLOT(setNewValue(double)));
				layout->addWidget(sliders[i], i, 1);
			}
		}
	}
	options->setLayout(layout);
	connect(options, SIGNAL(shown()),
			this, SLOT(setOptions()));
	return options;
}

QGroupBox* OptionDouble::getOptions() {
	if(!modifiable)
		return 0;
	
	if(options) {
		return options;
	}
	
	options = new OptionGroupBox(getName());
	QGridLayout *layout = new QGridLayout();
	for(int i = 0; i < numParams; i++) {
		if(fields.size() > i) {
			layout->addWidget(new QLabel(fields[i]), i, 0);
		}
		spins[i] = new QDoubleSpinBox();
		if(ranges[i*2] != ranges[i*2 + 1]) {
			spins[i]->setRange(ranges[2*i], ranges[2*i + 1]);
		}
		if(otype & SpinBox) 
			layout->addWidget(spins[i], i, 2);
		if(otype & Slider) {
			if(ranges[i*2] != ranges[i*2+1]) { //range required for slider
				sliders[i] = new OptionSlider(ranges[i*2], ranges[i*2+1], steps);
				connect((OptionSlider*)sliders[i], SIGNAL(newValue(double)),
							spins[i], SLOT(setValue(double)));
				connect(spins[i], SIGNAL(valueChanged(double)),
						(OptionSlider*)sliders[i], SLOT(setNewValue(double)));
				layout->addWidget(sliders[i], i, 1);
			}
		}
	}
	options->setLayout(layout);
	connect(options, SIGNAL(shown()),
			this, SLOT(setOptions()));
	return options;
}

QGroupBox* OptionBool::getOptions() {
	if(!modifiable)
		return 0;
	
	if(options) {
		return options;
	}
	
	options = new OptionGroupBox(getName());
	QGridLayout *layout = new QGridLayout(); /*
	for(int i = 0; i < numParams; i++) {
		if(fields.size() > i) {
			layout->addWidget(new QLabel(fields[i]), i, 0);
		}
		spins[i] = new QDoubleSpinBox();
		if(ranges[i*2] != ranges[i*2 + 1]) {
			spins[i]->setRange(ranges[2*i], ranges[2*i + 1]);
		}
		if(otype & SpinBox)
			layout->addWidget(spins[i], i, 2);
		
		if(otype & Slider) {
			if(ranges[i*2] != ranges[i*2+1]) {
				sliders[i] = new QSlider(Qt::Horizontal);
				sliders[i]->setRange(ranges[i*2], ranges[i*2 + 1]);
				connect(sliders[i], SIGNAL(sliderMoved(int)),
						spins[i], SLOT(setValue(int)));
				connect(spins[i], SIGNAL(valueChanged(int)),
						sliders[i], SLOT(setValue(int)));
				layout->addWidget(sliders[i], i, 1);
			}
		}
	} */
	cb = new QCheckBox();
	cb->setChecked(value);
	layout->addWidget(cb);
	options->setLayout(layout);
	connect(options, SIGNAL(shown()),
			this, SLOT(setOptions()));
	return options;
}

void OptionBool::accepted() {
	if(modifiable) {
		for(int i = 0; i < numParams; i++) {
			value = cb->isChecked();
		}
	}
}

void OptionBool::setOptions() {
	if(!modifiable)
		return;
	
	for(int i = 0; i < numParams; i++) {
		cb->setChecked(value);
	}
}

void OptionInt::accepted() {
	if(modifiable) {
		for(int i = 0; i < numParams; i++) {
			vars[i] = spins[i]->value();
		}
	}
}

void OptionFloat::setOptions() {
	if(!modifiable)
		return;
	
	for(int i = 0; i < numParams; i++) {
		spins[i]->setValue(vars[i]);
	}
}

void OptionDouble::setOptions() {
	if(!modifiable)
		return;
	
	for(int i = 0; i < numParams; i++) {
		spins[i]->setValue(vars[i]);
	}
}

void OptionInt::setOptions() {
	if(!modifiable)
		return;
	
	for(int i = 0; i < numParams; i++) {
		spins[i]->setValue(vars[i]);
	}
}

void OptionFloat::accepted() {
	if(modifiable) {
		for(int i = 0; i < numParams; i++) {
			vars[i] = (float)spins[i]->value();
		}
	}
}

void OptionDouble::accepted() {
	if(modifiable) {
		for(int i = 0; i < numParams; i++) {
			vars[i] = (float)spins[i]->value();
		}
	}
}
