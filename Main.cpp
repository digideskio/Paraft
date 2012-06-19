#include <QCoreApplication>
#include <cassert>
#include "MultiCoreController.h"
#include <iostream>

int main (int argc, char** argv) {
    if (argc != 5) {
        std::cout << "argc.0: " << argv[0] << endl;
        std::cout << "argc.1: " << argv[1] << endl;
        std::cout << "argc.2: " << argv[2] << endl;
        std::cout << "argc.3: " << argv[3] << endl;
        std::cout << "argc.4: " << argv[4] << endl;
        cerr << "Usage : " << argv[0] << " npx npy npz #dataset(0/1)" << endl;
        return 0;
    }

    QCoreApplication app(argc, argv);
    MultiCoreController mcc;
    mcc.Init(argc, argv);
    mcc.Start();

    return app.exec();
}
