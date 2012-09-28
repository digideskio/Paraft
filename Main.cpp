#include "MpiController.h"
#include <iostream>

using namespace std;

int main (int argc, char** argv) {
    if (argc != 4) {
        cout << "argv[0]: " << argv[0] << endl;
        cout << "argv[1]: " << argv[1] << endl;
        cout << "argv[2]: " << argv[2] << endl;
        cout << "argv[3]: " << argv[3] << endl;
        cerr << "Usage : " << argv[0] << " npx npy npz" << endl;
        return 0;
    }

    MpiController *mc = new MpiController;
    mc->InitWith(argc, argv);
    mc->Start();
    delete mc;
    return 0;
}
