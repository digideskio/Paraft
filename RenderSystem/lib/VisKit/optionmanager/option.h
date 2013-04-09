#ifndef _OPTION_H_
#define _OPTION_H_

#include <QObject>
#include <QVector>
#include <QStringList>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QCheckBox>

#include "optiongroupbox.h"
//! Option
/*! The Option base class holds all the things things that all options share in common. Since
	most options are numerical in nature, they have the option to be a slider, spinbox or both.

	They also generate their own option box.
*/
class Option : public QObject {
	Q_OBJECT
public:
	//! OptionType
	//! An enum for what kinds of options there are.
	enum OptionType { SpinBox = 1, Slider = 2, Both = 3 };
protected:
	//! numParams
	/*! int for the number of parameters we have. The reason it's written this way to mimic the
		the style of the shadermanager class
	*/
	int numParams;

	//! the name given to the option upon creation, by the manager
	QString name;
	
	//! the display name
	QString display;

	//! the type of option, spinbox slider or both
	OptionType otype;

	//! names of the fields
	QStringList fields;

	//! our special option group box, enures the correct values on display
	OptionGroupBox* options;

	//! sliders for the groupbox
	QVector<QSlider*> sliders;

	//! whether or not this will generate an external group box when asked
	bool modifiable;

	//! number of steps for the slider
	int steps;
public:
	//! constructor
	/*! \param numParams the number of paramters the option has
		\param name the name of the option
		\param otype the type of option this will generate
	*/
	Option(int numParams, const QString& name, OptionType otype);
	
	//! returns the name given or the display name, if set
	const QString& getName() const { return display.isEmpty() ? name : display; }

	//! sets whether or not this option is modifiable
	void setModifiable(bool mod) { modifiable = mod; }

	//! vararg function to set the field names, just enter them in order
	void setFieldNames(const char* first, ...);

	//! gets whether or not this option generates a optionbox
	bool isModifiable() const { return modifiable; }

	//! sets the display name
	void setDisplayName(const QString& d) { display = d; }

	//! sets the slider steps
	void setSteps(int s);

	virtual QGroupBox* getOptions()=0;
	virtual void accepted()=0;
	virtual ~Option() {}
};


//! OptionBool
/*! The simplest option class, option bool basically encapsulates a checkbox.
*/
class OptionBool : public Option {
	Q_OBJECT
	bool value;
	QCheckBox* cb;
	
public:
	OptionBool(const QString& name):Option(1, name, Both), value(true) {}
	QGroupBox* getOptions();
	void accepted();
	bool toggle() { value = !value; return value; }
	void setValue(bool v) { value = v; }
	bool& v() { return value; }
public slots:
	void setOptions();
};

class OptionInt : public Option {
	Q_OBJECT
	
	QVector<int> vars;
	QVector<int> ranges;
	QVector<QSpinBox*> spins;
public:
	OptionInt(int numParams, const QString& name=QString(), OptionType otype=Both)
		:Option(numParams, name, otype), vars(numParams), ranges(numParams*2), spins(numParams) {
		for(int i = 0; i < numParams; i++) {
			vars[i] = 0;
			ranges[i*2] = 0;
			ranges[i*2 + 1] = 0;
			spins[i] = 0;
		}
	}
	
	void setRanges(int first, ...);
	void setValues(int first, ...);
	void accepted();
	QVector<int>& v() { return vars; }
	QGroupBox* getOptions();
public slots:
	void setOptions();
};

class OptionFloat : public Option {
	Q_OBJECT
	
	QVector<float> vars;
	QVector<float> ranges;
	QVector<QDoubleSpinBox*> spins;
public:
	OptionFloat(int numParams, const QString& name=QString(), OptionType otype=Both)
		:Option(numParams, name, otype), vars(numParams), ranges(numParams*2), spins(numParams) {
		for(int i = 0; i < numParams; i++) {
			vars[i] = 0;
			ranges[i*2] = 0;
			ranges[i*2 + 1] = 0;
			spins[i] = 0;
		}
	}
	
	void setRanges(float first, ...);
	void setValues(float first, ...);
	void accepted();
	QVector<float>& v() { return vars; }
	QGroupBox* getOptions();
public slots:
	void setOptions();
};

class OptionDouble : public Option {
	Q_OBJECT
	
	QVector<double> vars;
	QVector<double> ranges;
	QVector<QDoubleSpinBox*> spins;
public:
	OptionDouble(int numParams, const QString& name=QString(), OptionType otype=Both)
		:Option(numParams, name, otype), vars(numParams), ranges(numParams*2), spins(numParams) {
		for(int i = 0; i < numParams; i++) {
			vars[i] = 0;
			ranges[i*2] = 0;
			ranges[i*2 + 1] = 0;
			spins[i] = 0;
		}
	}
	
	void setRanges(double first, ...);	
	void setValues(double first, ...);
	void accepted();
	QVector<double>& v() { return vars; }
	QGroupBox* getOptions();
public slots:
	void setOptions();
};

#endif
