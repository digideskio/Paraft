#include <QApplication>
#include <QDialog>

#include "optiontest.h"

int main(int argc, char* argv[]) {
	QApplication app(argc, argv);
	
	OptionTest w;
	w.show();
	
	return app.exec();
}
