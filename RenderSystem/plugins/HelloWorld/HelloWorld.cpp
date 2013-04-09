#include "HelloWorld.h"

void HelloWorld::init()
{
    setPluginName("HelloWorld");
    setPluginVersion("0.9.0");
    QMenu *menuTools = getMenu("Tools");
    QAction *actSayHello = menuTools->addAction(tr("Say Hello"));
    connect(actSayHello, SIGNAL(triggered()), this, SLOT(sayHello()));
    QPushButton *btnSayHello = new QPushButton(tr("Say Hello"));
    addDockWidget(tr("Hello World"), btnSayHello);
    connect(btnSayHello, SIGNAL(clicked()), this, SLOT(sayHello()));
}

void HelloWorld::sayHello()
{
    qDebug("Hello world!");
    QLabel *lblHello = new QLabel(tr("Hello, world!"));
    QMdiSubWindow *win = addSubWindow(lblHello);
    win->resize(320, 120);
    win->setMinimumSize(320, 120);
    win->show();
}

Q_EXPORT_PLUGIN2(HelloWorldPlugin, HelloWorld);
