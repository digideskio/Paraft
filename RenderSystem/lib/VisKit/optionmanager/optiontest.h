#ifndef _OPTIONTEST_H_
#define _OPTIONTEST_H_

#include <QWidget>
#include "optionmanager.h"

class QDialog;
class OptionTest : public QWidget {
	Q_OBJECT
	OptionManager om;
	QDialog* options;
	void showOptions();
public:
	OptionTest();
	
	void keyReleaseEvent(QKeyEvent* e);
	~OptionTest();
};


#endif
