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

//    ifstream inf("/Users/Yang/Develop/Data/jet_vort/jet_vort_0001.dat");
//    if (!inf) cout << "file not found;" << endl;

//    float value;
//    for (int i = 0; i < 10000; i++) {
//        inf.read(reinterpret_cast<char*>(&value), sizeof(float));
//        if (i % 100 == 0)
//            cout << value << "\t";
//        } cout << endl;
//    inf.close();

    return 0;
}
