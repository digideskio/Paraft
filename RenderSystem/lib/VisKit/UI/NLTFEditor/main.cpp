#include <QtGui/QApplication>
#include "nltfeditor.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    NLTFEditor w;
    w.show();

    return a.exec();
}
