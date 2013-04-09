#ifndef HELLOWORLD_H
#define HELLOWORLD_H

#include "PluginInterface.h"

class HelloWorld : public PluginInterface
{
    Q_OBJECT
    Q_INTERFACES(PluginInterface)

public:
    void init();

public slots:
    void sayHello();
};

#endif // HELLOWORLD_H
