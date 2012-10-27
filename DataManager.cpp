#include "DataManager.h"

DataManager::DataManager() {
    tfResolution = -1;
    pMaskVolume = NULL;
    dataSequence.clear();
}

DataManager::~DataManager() {
    if (!dataSequence.empty()) {
        DataSequence::iterator it;
        for (it = dataSequence.begin(); it != dataSequence.end(); it++) {
            delete [] it->second;   // pointer to each timestep
        }
    }

    if (pMaskVolume != NULL) {
        delete [] pMaskVolume;
    }
}

void DataManager::CreateNewMaskVolume() {
    pMaskVolume = new float[volumeSize];
    std::fill(pMaskVolume, pMaskVolume+volumeSize, 0);
}

void DataManager::InitTFSettings(string filename) {
    ifstream inf(filename.c_str(), ios::binary);
    if (!inf) { cout << "cannot read tf setting: " + filename << endl; exit(1); }

    float tfResF = 0.0f;
    inf.read(reinterpret_cast<char*>(&tfResF), sizeof(float));
    if (IS_BIG_ENDIAN) ReverseEndian(&tfResF);
    if (tfResF < 1) { cout << "tfResolution = " << tfResF << endl; exit(2); }

    tfResolution = (int)tfResF;
    pTFOpacityMap = new float[tfResolution];
    inf.read(reinterpret_cast<char*>(pTFOpacityMap), tfResolution*sizeof(float));
    inf.close();

    if (IS_BIG_ENDIAN) {
        for (int i = 0; i < tfResolution; i++) {
            ReverseEndian(&pTFOpacityMap[i]);
        }
    }
}

void DataManager::PreloadDataSequence(Vector3i partition, Vector3i blockCoord, DataSet ds, int timestep) {
    blockSize = ds.size / partition;
    volumeSize = blockSize.volume();

    // delete if exsiting data is not within 2-neighbor of current timestep
    DataSequence::iterator it;
    for (it = dataSequence.begin(); it != dataSequence.end(); it++) {
        if (it->first < timestep-2 || it->first > timestep+2) {
            delete [] it->second;
            dataSequence.erase(it);
        }
    }

    for (int t = timestep-2; t <= timestep+3; t++) {
        if (t < ds.start || t > ds.end) {   // only [t-2, t-1, t, t+1, t+2]
            continue;
        }

        // 1. allocate new data buffer then add pointer to map
        if (dataSequence[t] == NULL) {
            dataSequence[t] = new float[volumeSize];
        }

        // 2. get file name from time index
        string filePath;
        char timestep[21]; // hold up to 64-bits
        sprintf(timestep, "%8d", t);
        for (int i = 0; i < 8; i++) {
            timestep[i] = timestep[i] == ' ' ? '0' : timestep[i];
        }
        filePath = ds.path + ds.prefix + timestep + ds.surfix;

        char *filename = new char[filePath.size() + 1];
        std::copy(filePath.begin(), filePath.end(), filename);
        filename[filePath.size()] = '\0';

        // 3. read corresponding file using mpi collective io
        int *gsizes = (blockSize * partition).toArray();
        int *subsizes = blockSize.toArray();
        int *starts = (blockSize * blockCoord).toArray();

        MPI_Datatype filetype;
        MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_FORTRAN,
                                 MPI_FLOAT, &filetype);
        MPI_Type_commit(&filetype);

        MPI_File file;
        int not_exist = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                                      MPI_INFO_NULL, &file);
        if (not_exist) printf("%s not exist.\n", filename);

        MPI_File_set_view(file, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
        MPI_File_read_all(file, dataSequence[t], volumeSize, MPI_FLOAT, MPI_STATUS_IGNORE);

        normalizeData(dataSequence[t], ds);

        MPI_File_close(&file);
        MPI_Type_free(&filetype);
        delete[] filename;
    }
}

void DataManager::normalizeData(float *pData, DataSet ds) {
    for (int i = 0; i < volumeSize; i++) {
        if (IS_BIG_ENDIAN) ReverseEndian(&pData[i]);
        pData[i] -= ds.min;
        pData[i] /= ds.max - ds.min;
    }
}
