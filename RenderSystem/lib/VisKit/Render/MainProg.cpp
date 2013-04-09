#include <QApplication>

#include "QVolumeRenderWindow.h"

bool verbose = false;
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QVolumeRenderWindow *renderWin = new QVolumeRenderWindow;
    renderWin->show();

	int retInt = app.exec();
	delete renderWin;
    return retInt;
} 
