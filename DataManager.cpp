#include "DataManager.h"

DataManager::DataManager() {
    tfResolution = -1;
    pDataBuffer = NULL;
    pMaskMatrix = NULL;
    dataSequence.clear();;
    minMaxSequence.clear();
    featureVectors.clear();
}

DataManager::~DataManager() {
    if (!dataSequence.empty()) {
        for (uint i = 0; i < dataSequence.size(); i++) {
            delete [] dataSequence.at(i);
        }
    }

    minMaxSequence.clear();

    if (featureVectors.size() != 0) {
        for (uint i = 0; i < featureVectors.size(); i++) {
            for (uint j = 0; j < featureVectors.at(i).size(); j++) {
                featureVectors.at(i).at(j).SurfacePoints.clear();
                featureVectors.at(i).at(j).InnerPoints.clear();
                featureVectors.at(i).at(j).Uncertainty.clear();
            }
        }
    }

    if (pMaskMatrix != NULL) {
        delete [] pMaskMatrix;
    }
}

void DataManager::CreateNewMaskMatrix() {
    pMaskMatrix = new float[volumeSize];
    memset(pMaskMatrix, 0, sizeof(float)*volumeSize);
}

void DataManager::InitTFSettings(string filename) {
    ifstream inf(filename.c_str(), ios::binary);
    if (!inf) { cout << "cannot read tf setting: " + filename << endl; exit(1); }

    float tfResF = 0.0f;
    inf.read(reinterpret_cast<char*>(&tfResF), sizeof(float));
    if (tfResF <= 0) { cout << "tfResolution = " << tfResF << endl; exit(2); }

    tfResolution = (int)tfResF;
    pTFOpacityMap = new float[tfResolution];
    inf.read(reinterpret_cast<char*>(pTFOpacityMap), tfResolution * sizeof(float));
    inf.close();

    if (IS_BIG_ENDIAN) {  // reverse endian
        union {
            float f;
            unsigned char b[4];
        } bigen, littlen;

        for (int i = 0; i < tfResolution; i++) {
            bigen.f = pTFOpacityMap[i];
            littlen.b[0] = bigen.b[3];
            littlen.b[1] = bigen.b[2];
            littlen.b[2] = bigen.b[1];
            littlen.b[3] = bigen.b[0];
            pTFOpacityMap[i] = littlen.f;
        }
    }

    // debug tf
    //    for (int i = 0; i < tfResolution; i++) {
    //        printf("%2.4f\t", pTFOpacityMap[i]);
    //    } cout << endl; exit(0);
}

void DataManager::MpiReadDataSequence(Vector3i blockCoord, Vector3i partition,
                                      DataSet ds) {
    volumeDim = ds.dim / partition;
    volumeSize = volumeDim.volume();

    for (int time = ds.start; time < ds.end; time++) {
        // 1. allocate new data buffer then add pointer to map
        pDataBuffer = new float[volumeSize];
        if (pDataBuffer == NULL) {
            cerr << "Allocate memory failed" << endl; exit(1);
        }
        dataSequence.push_back(pDataBuffer); // index offset = ds.start

        // 2. get file name from time index
        string filePath;
        char timestep[21]; // hold up to 64-bits
        sprintf(timestep, "%8d", time);
        for (int i = 0; i < 8; i++) {
            timestep[i] = timestep[i] == ' ' ? '0' : timestep[i];
        }
        filePath = ds.path + ds.prefix + timestep + ds.surfix;

        char *filename = new char[filePath.size() + 1];
        copy(filePath.begin(), filePath.end(), filename);
        filename[filePath.size()] = '\0';

        // 3. read corresponding file using mpi collective io
        int *gsizes = (volumeDim * partition).toArray();
        int *subsizes = volumeDim.toArray();
        int *starts = (volumeDim * blockCoord).toArray();

        MPI_Datatype filetype;
        MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_FORTRAN,
                                 MPI_FLOAT, &filetype);
        MPI_Type_commit(&filetype);

        MPI_File file;
        int not_exist = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                                      MPI_INFO_NULL, &file);
        if (not_exist) printf("%s not exist.\n", filename);

        MPI_File_set_view(file, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
        MPI_File_read_all(file, pDataBuffer, volumeSize, MPI_FLOAT,
                          MPI_STATUS_IGNORE);

        if (IS_BIG_ENDIAN) { // reverse endian
            union {
                float f;
                unsigned char b[4];
            } bigen, littlen;

            for (int i = 0; i < volumeSize; i++) {
                bigen.f = pDataBuffer[i];
                littlen.b[0] = bigen.b[3];
                littlen.b[1] = bigen.b[2];
                littlen.b[2] = bigen.b[1];
                littlen.b[3] = bigen.b[0];
                pDataBuffer[i] = littlen.f;
            }
        }


        MPI_File_close(&file);
        MPI_Type_free(&filetype);
        delete[] filename;

        calculateLocalMinMax();
    }

    normalizeData(ds);
}

void DataManager::calculateLocalMinMax() {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();

    for (int i = 0; i < volumeSize; i++) {
        min = min < pDataBuffer[i] ? min : pDataBuffer[i];
        max = max > pDataBuffer[i] ? max : pDataBuffer[i];
    }

    minMaxSequence.push_back(MinMax(min, max));
}

void DataManager::normalizeData(DataSet ds) {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();

    // 1. get local min-max for the whole data sequence
    for (int i = 0; i < ds.end-ds.start; i++) {
        min = min < minMaxSequence.at(i).min ? min : minMaxSequence.at(i).min;
        max = max > minMaxSequence.at(i).max ? max : minMaxSequence.at(i).max;
    }

    // 2. get global min-max for the whole data sequence
    float gmin, gmax;
    MPI_Allreduce(&min, &gmin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max, &gmax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    cout << "global min: " << gmin << " max: " << gmax << endl;

    // 2. normalize data sequence
    for (int i = 0; i < ds.end-ds.start; i++) {
        for (int j = 0; j < volumeSize; j++) {
            dataSequence.at(i)[j] -= gmin;          // global min -> 0
            dataSequence.at(i)[j] /= (gmax-gmin);   // global max -> 1
        }
    }
}
