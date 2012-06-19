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
        std::cout << "Clean data vector" << std::endl;
        for (int i = 0; i < pDataVector.size(); i++) {
            delete [] pDataVector.at(i);
        }
    }
    pMinMaxVector.clear();

    if (pFeatureVectors.size() != 0) {
        for (int i = 0; i < pFeatureVectors.size(); i++) {
            for (int j = 0; j < pFeatureVectors.at(i).size(); j++) {
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

bool DataManager::ReadDataSequence(string filePath, string prefix, string suffix,
                                   int iStart, int iEnd, Vector3i dimXYZ,
                                   Vector3i workerNumProcXYZ, Vector3i workerIDXYZ) {
    volumeDim = dimXYZ / workerNumProcXYZ;
    volumeSize = volumeDim.volume();

    string fileName;
    bool result;
    char numstr[21];
    for (int i = iStart; i <= iEnd; i++) {
        sprintf(numstr, "%d", i);
        fileName = filePath + "/" + prefix + numstr + "." + suffix;
        result = ReadOneDataFile(fileName, volumeDim, workerNumProcXYZ, workerIDXYZ);
    }
    normalizeData();
    return result;
}

// Read one file from disk and save it to the end of the data vector
bool DataManager::ReadOneDataFile(string filePath, Vector3i segLength,
                                  Vector3i workerNumProcessesXYZ,
                                  Vector3i workerIDXYZ) {

    int bufferSize = segLength.volume();
    float *pBuffer = allocateNewDataBuffer(segLength);

    int *gsizes = (segLength * workerNumProcessesXYZ).toArray();
    int *subsizes = segLength.toArray();
    int *starts = (segLength * workerIDXYZ).toArray();

    MPI_Datatype filetype;
    MPI_Type_create_subarray(3, gsizes, subsizes, starts,
                             MPI_ORDER_FORTRAN, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);

    char *filename = new char[filePath.size() + 1];
    copy(filePath.begin(), filePath.end(), filename);
    filename[filePath.size()] = '\0';

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_set_view(file, 0 /* offset */, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(file, pBuffer, bufferSize, MPI_FLOAT, MPI_STATUS_IGNORE);

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

    int iTimelength = pDataVector.size();
    for (int i = 0 ; i < iTimelength; i++) {
        min = min < pMinMaxVector.at(i).min ? min : pMinMaxVector.at(i).min;
        max = max > pMinMaxVector.at(i).max ? max : pMinMaxVector.at(i).max;
    }

    float delta = max - min;
    globalMinMax.x = min;
    globalMinMax.y = max;

    for (int j = 0; j < iTimelength; j++) {
        for (int i = 0; i < volumeSize; i++) {
            pDataVector.at(j)[i] -= min;
            pDataVector.at(j)[i] /= delta;
        }
    }
}

float* DataManager::allocateNewDataBuffer(Vector3i dim) {
    pAllocatedBuffer = new float[dim.volume()];
    if (pAllocatedBuffer == NULL) {
        std::cout << "Allocate memory failed." << std::endl;
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
