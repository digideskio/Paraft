#include <QApplication>
#include "QTFEditor.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QTFEditor	tf(1024);
	tf.show();
	tf.getTFPanel()->setRange(-0.1, 0.1);
	return app.exec();
}

