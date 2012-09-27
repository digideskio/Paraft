#include "DataManager.h"

DataManager::DataManager() {
    pAllocatedBuffer = NULL;
    pMaskMatrix = NULL;
    pDataVector.clear();
    pMinMaxVector.clear();
    pFeatureVectors.clear();
}

DataManager::~DataManager() {
    if (pDataVector.size() != 0) {
        for (unsigned int i = 0; i < pDataVector.size(); i++) {
            delete [] pDataVector.at(i);
        }
    }
    pMinMaxVector.clear();

    if (pFeatureVectors.size() != 0) {
        for (unsigned int i = 0; i < pFeatureVectors.size(); i++) {
            for (unsigned int j = 0; j < pFeatureVectors.at(i).size(); j++) {
                pFeatureVectors.at(i).at(j).SurfacePoints.clear();
                pFeatureVectors.at(i).at(j).InnerPoints.clear();
                pFeatureVectors.at(i).at(j).Uncertainty.clear();
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

void DataManager::ReadSphDataSequence(DataSet ds) {
    SphReader *sph = new SphReader;

    for (int i = ds.start; i <= ds.end; i++) {
        string filePath;
        char timestep[21]; // hold up to 64 bit integer
        sprintf(timestep, "%8d", i);
        for (int i = 0; i < 8; i++) {
            timestep[i] = timestep[i] == ' ' ? '0' : timestep[i];
        }

        // prs_0000001500_id000000.sph
        // ds.index_start = 1;
        // ds.index_end   = 9;
        // ds.prefix      = "prs_";
        // ds.surfix      = "00_id000000.sph";
        // ds.data_path   = "/Users/Yang/Develop/Data/Sim_128_128_128";
        filePath = ds.path + ds.prefix + timestep + ds.surfix;

        cout << filePath << endl;

        pAllocatedBuffer = sph->loadData(filePath);
        pDataVector.push_back(pAllocatedBuffer);

        volumeSize = sph->getVolumeSize();
        calculateLocalMinMax();
    }
    normalizeData();

    delete sph;
}

void DataManager::MpiReadDataSequence(Vector3i blockCoord, Vector3i partition, DataSet ds) {

    volumeDim = ds.dim / partition;
    volumeSize = volumeDim.volume();

    for (int i = ds.start; i <= ds.end; i++) {
        string fileName;
        char timestep[21]; // enough to hold all numbers up to 64-bits
        sprintf(timestep, "%4d", i);
        for (int i = 0; i < 4; i++) {
            timestep[i] = timestep[i] == ' ' ? '0' : timestep[i];
        }
        fileName = ds.path + "/" + ds.prefix + timestep + ds.surfix;

        mpiReadOneDataFile(blockCoord, partition, fileName);
    }
    normalizeData();
}

// Read one file from disk and save it to the end of the data vector
bool DataManager::mpiReadOneDataFile(Vector3i blockCoord, Vector3i partition,
                                  string filePath) {

    float *pBuffer = allocateNewDataBuffer(volumeSize);

    int *gsizes = (volumeDim * partition).toArray();
    int *subsizes = volumeDim.toArray();
    int *starts = (volumeDim * blockCoord).toArray();

    MPI_Datatype filetype;
    MPI_Type_create_subarray(3, gsizes, subsizes, starts,
                             MPI_ORDER_FORTRAN, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);

    char *filename = new char[filePath.size() + 1];
    copy(filePath.begin(), filePath.end(), filename);
    filename[filePath.size()] = '\0';

    MPI_File file;
    int not_exist = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                                  MPI_INFO_NULL, &file);
    if (not_exist) printf("%s not exist.\n", filename);

    MPI_File_set_view(file, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(file, pBuffer, volumeSize, MPI_FLOAT, MPI_STATUS_IGNORE);

    MPI_File_close(&file);
    MPI_Type_free(&filetype);

    calculateLocalMinMax();

    delete[] filename;
    return true;
}

void DataManager::normalizeData() {
    // Get the first local min and max

    float min = pMinMaxVector.at(0).min;
    float max = pMinMaxVector.at(0).max;

    for (uint i = 0 ; i < pDataVector.size(); i++) {
        min = min < pMinMaxVector.at(i).min ? min : pMinMaxVector.at(i).min;
        max = max > pMinMaxVector.at(i).max ? max : pMinMaxVector.at(i).max;
    }

    for (uint j = 0; j < pDataVector.size(); j++) {
        for (int i = 0; i < volumeSize; i++) {
            pDataVector.at(j)[i] -= min;
            pDataVector.at(j)[i] /= (max-min);
        }
    }
}

float* DataManager::allocateNewDataBuffer(int bufferSize) {
    pAllocatedBuffer = new float[bufferSize];
    if (pAllocatedBuffer == NULL) {
        cerr << "Allocate memory failed" << endl;
        return pAllocatedBuffer;
    }
    pDataVector.push_back(pAllocatedBuffer);
    return pAllocatedBuffer;
}

void DataManager::calculateLocalMinMax() {
    float min = pAllocatedBuffer[0];
    float max = pAllocatedBuffer[1];

    for (int i = 1; i < volumeSize; i++) {
        min = min < pAllocatedBuffer[i] ? min : pAllocatedBuffer[i];
        max = max > pAllocatedBuffer[i] ? max : pAllocatedBuffer[i];
    }

    pMinMaxVector.push_back(MinMax(min, max));
}
