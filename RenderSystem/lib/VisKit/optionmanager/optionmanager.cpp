#include "optionmanager.h"
#include "option.h"
#include <QVBoxLayout>

void OptionManager::addOptionf(const QString& name, int numFields) {
	if(!options.contains(name)) {
		floats[name] = new OptionFloat(numFields, name);
		options[name] = floats[name];
	}
}

void OptionManager::addOptiond(const QString& name, int numFields) {
	if(!options.contains(name)) {
		doubles[name] = new OptionDouble(numFields, name);
		options[name] = doubles[name];
	}
	
}
void OptionManager::addOptioni(const QString& name, int numFields) {
	if(!options.contains(name)) {
		ints[name] = new OptionInt(numFields, name);
		options[name] = ints[name];
	}
	
}
void OptionManager::addOptionb(const QString& name) {
	if(!options.contains(name)) {
		bools[name] = new OptionBool(name);
		options[name] = bools[name];
	}
	
}

OptionFloat* OptionManager::f(const QString& name) {
	if(!floats.contains(name))
		return 0;
	return floats[name];
}

OptionDouble* OptionManager::d(const QString& name) {
	if(!doubles.contains(name))
		return 0;
	return doubles[name];
}

OptionInt* OptionManager::i(const QString& name) {
	if(!ints.contains(name))
		return 0;
	return ints[name];
}

OptionBool* OptionManager::b(const QString& name) {
	if(!bools.contains(name))
		return 0;
	return bools[name];
}

Option* OptionManager::operator [](const QString& index) {
	if(!options.contains(index))
		return 0;
	return options[index];
}

QGroupBox* OptionManager::getOptions() {
	if(options.isEmpty())
		return 0;
	
	if(groupbox) {
		return groupbox;
	}
	groupbox = new QGroupBox(tr("Options"));
	QVBoxLayout* layout = new QVBoxLayout;
	for(QHash<QString, Option*>::iterator it = options.begin(); it != options.end(); it++) {
		if((*it)->isModifiable())
			layout->addWidget((*it)->getOptions());
	}
	groupbox->setLayout(layout);
	return groupbox;
}

void OptionManager::accepted() {
	for(QHash<QString, Option*>::iterator it = options.begin(); it != options.end(); it++) {
		(*it)->accepted();
	}
}
