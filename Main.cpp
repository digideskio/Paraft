#include <QCoreApplication>
#include <cassert>
#include "MultiCoreController.h"
#include <iostream>

int main (int argc, char** argv) {
    if (argc != 5) {
        std::cout << "argv[0]: " << argv[0] << endl;
        std::cout << "argv[1]: " << argv[1] << endl;
        std::cout << "argv[2]: " << argv[2] << endl;
        std::cout << "argv[3]: " << argv[3] << endl;
        std::cout << "argv[4]: " << argv[4] << endl;
        cerr << "Usage : " << argv[0] << " npx npy npz #dataset(0/1)" << endl;
        return 0;
    }

    QCoreApplication app(argc, argv);
    MultiCoreController mcc;
    mcc.Init(argc, argv);
    mcc.Start();

    return app.exec();
}
