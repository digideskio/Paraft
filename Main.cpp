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

//-----------------

//    int world_rank;
//    int num_proc;
//    int neighbors[3];

//    MPI_Init(&argc,&argv);
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

//    MPI_Comm MPI_COMM_ORIG, MPI_COMM_LOCAL, MPI_COMM_TEMP;
//    MPI_Group world_group, local_group;

//    MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_ORIG);
//    MPI_Comm_group(MPI_COMM_ORIG, &world_group);

//    cout << "111" << endl;

//    if (world_rank == 0) {
//        neighbors[0] = 0;
//        neighbors[1] = 1;
//        neighbors[2] = 2;
//    } else if (world_rank == 1) {
//        neighbors[0] = 0;
//        neighbors[1] = 1;
//        neighbors[2] = 3;
//    } else if (world_rank == 2) {
//        neighbors[0] = 0;
//        neighbors[1] = 2;
//        neighbors[2] = 3;
//    } else {
//        neighbors[0] = 1;
//        neighbors[1] = 2;
//        neighbors[2] = 3;
//    }

//    MPI_Group_incl(world_group, 3, neighbors, &local_group);
//    MPI_Comm_create(MPI_COMM_ORIG, local_group, &MPI_COMM_TEMP);
//    MPI_Barrier(MPI_COMM_TEMP);

//    MPI_Comm_dup(MPI_COMM_TEMP, &MPI_COMM_LOCAL);

//    cout << "+[" << world_rank << "]: ";
//    int result = -1;
//    MPI_Allreduce(&world_rank, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_LOCAL);
//    cout << result;
//    cout << "-" << endl;

//    MPI_Group_free(&world_group);
//    MPI_Group_free(&local_group);

//    MPI_Finalize();
//    return 0;
}
