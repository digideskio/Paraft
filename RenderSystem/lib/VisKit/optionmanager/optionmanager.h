#ifndef _OPTIONMANAGER_H_
#define _OPTIONMANAGER_H_

#include "option.h"

#include <QHash>
#include <QString>

class OptionManager : public QObject {
	Q_OBJECT
	
	QHash<QString, Option*> options;
	QHash<QString, OptionFloat*> floats;
	QHash<QString, OptionInt*> ints;
	QHash<QString, OptionBool*> bools;
	QHash<QString, OptionDouble*> doubles;
	
	QGroupBox* groupbox;
public:
	OptionManager():groupbox(0) {}
	void addOptionf(const QString& name, int numFields);
	void addOptiond(const QString& name, int numFields);
	void addOptioni(const QString& name, int numFields);
	void addOptionb(const QString& name);
	
	OptionFloat* f(const QString& name);
	OptionInt* i(const QString& name);
	OptionBool* b(const QString& name);
	OptionDouble* d(const QString& name);
	
	Option* operator[](const QString& index);
	QGroupBox* getOptions();
public slots:
	void accepted();
};


#endif
