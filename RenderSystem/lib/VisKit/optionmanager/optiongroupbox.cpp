#include "optiongroupbox.h"
#include <QShowEvent>

void OptionGroupBox::showEvent(QShowEvent*) {
	emit shown();
}
