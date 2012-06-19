#include "DataManager.h"

DataManager::DataManager() {
    pAllocatedBuffer = NULL;
    pMaskMatrix = NULL;
    pDataVector.clear();
    pLocalMinMaxVector.clear();
    pFeatureVectors.clear();
}

DataManager::~DataManager() {
    if (pDataVector.size() != 0) {
        std::cout << "Clean data vector" << std::endl;
        for (int i = 0; i < pDataVector.size(); i++) {
            delete [] pDataVector.at(i);
        }
    }
    pLocalMinMaxVector.clear();

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
    volumeSize = volumeDim.x * volumeDim.y * volumeDim.z;

    QString fileName;
    bool result;
    char numstr[21];
    for (int i = iStart; i <= iEnd; i++) {
        sprintf(numstr, "%d", i);
        fileName = QString::fromStdString(filePath + "/" + prefix + numstr + "." + suffix);
        qDebug(fileName.toUtf8());
        result = ReadOneDataFile(fileName, volumeDim, workerNumProcXYZ, workerIDXYZ);
    }
    normalizeData();
    return result;
}

// Read one file from disk and save it to the end of the data vector
bool DataManager::ReadOneDataFile(QString strFilePath, Vector3i segLength,
                                  Vector3i workerNumProcessesXYZ,
                                  Vector3i workerIDXYZ) {

    int bufferSize = segLength.x * segLength.y * segLength.z;

    float *pBuffer = allocateNewDataBuffer(segLength);

    // length per dimension
//    int *gsizes = (segLength * workerNumProcessesXYZ).v;

    int gsizes[3] = { segLength.x * workerNumProcessesXYZ.x,
                      segLength.y * workerNumProcessesXYZ.y,
                      segLength.z * workerNumProcessesXYZ.z };

    int subsizes[3] = { segLength.x, segLength.y, segLength.z };

    int starts[3] = { segLength.x * workerIDXYZ.x,
                      segLength.y * workerIDXYZ.y,
                      segLength.z * workerIDXYZ.z };

    char filename[1024];
    sprintf(filename, "%s", strFilePath.toLocal8Bit().constData());

    MPI_Datatype filetype;
    MPI_Type_create_subarray(3, gsizes, subsizes, starts,
                             MPI_ORDER_FORTRAN, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_set_view(file, 0 /* offset */, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(file, pBuffer, bufferSize, MPI_FLOAT, MPI_STATUS_IGNORE);

    MPI_File_close(&file);
    MPI_Type_free(&filetype);

    calculateLocalMinMax();
    return true;
}

void DataManager::normalizeData() {
    // Get the first local min and max
    float min = pLocalMinMaxVector.at(0).x;
    float max = pLocalMinMaxVector.at(0).y;

    int iTimelength = pDataVector.size();
    for (int i = 0 ; i < iTimelength; i++) {
        min = min < pLocalMinMaxVector.at(i).x ? min : pLocalMinMaxVector.at(i).x;
        max = max > pLocalMinMaxVector.at(i).y ? max : pLocalMinMaxVector.at(i).y;
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
    pAllocatedBuffer = new float[dim.x*dim.y*dim.z];
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

    Vector2f minMax; {
        minMax.x = min;
        minMax.y = max;
    }

    pLocalMinMaxVector.push_back(minMax);
}
