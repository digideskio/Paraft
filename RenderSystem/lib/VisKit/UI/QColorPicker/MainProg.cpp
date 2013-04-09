#include <QApplication>
#include <QPushButton>

#include "QColorPicker.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QColorPicker	cp;
	QObject::connect(&cp, SIGNAL(clicked()),&app, SLOT(quit()));
	cp.show();
	return app.exec();
}
