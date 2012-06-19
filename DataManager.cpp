#include "DataManager.h"

DataManager::DataManager(QObject *parent) : QObject(parent) {
    pAllocatedBuffer = NULL;
    pMaskMatrix = NULL;
    pDataVector.clear();
    pLocalMinMax.clear();
    pFeatureVectors.clear();
}

DataManager::~DataManager() {
    if (pDataVector.size() != 0) {
        std::cout << "Clean data vector" << std::endl;
        for (int i = 0; i < pDataVector.size(); i++) {
            delete [] pDataVector.at(i);
        }
    }
    pLocalMinMax.clear();

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

float* DataManager::AllocateNewDataBuffer(int x, int y, int z) {
    pAllocatedBuffer = new float[x*y*z];
    if (pAllocatedBuffer == NULL) {
        std::cout << "Allocate memory failed." << std::endl;
        return pAllocatedBuffer;
    }
    pDataVector.push_back(pAllocatedBuffer);
    return pAllocatedBuffer;
}

void DataManager::setVolumeDimension(int x, int y, int z) {
    dimX = x; dimY = y; dimZ = z;
    volumeSize = dimX * dimY * dimZ;
}

void DataManager::CreateNewMaskMatrix(int size) {
    pMaskMatrix = new float[size];
    memset(pMaskMatrix, 0, sizeof(float)*size);
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

    pLocalMinMax.push_back(minMax);
}

bool DataManager::ReadDataSequence(string filePath, string prefix, string suffix,
                                   int iStart, int iEnd, Vector3i dimXYZ,
                                   Vector3i workerNumProcXYZ, Vector3i workerIDXYZ) {

    Vector3i segLength;
    segLength.x = dimXYZ.x / workerNumProcXYZ.x;
    segLength.y = dimXYZ.y / workerNumProcXYZ.y;
    segLength.z = dimXYZ.z / workerNumProcXYZ.z;

    setVolumeDimension(segLength.x, segLength.y, segLength.z);

    QString fileName;
    bool result;
    char numstr[21];
    for (int i = iStart; i <= iEnd; i++) {
        sprintf(numstr, "%d", i);
        fileName = QString::fromStdString(filePath + "/" + prefix + numstr + "." + suffix);
        qDebug(fileName.toUtf8());
        result = ReadOneDataFile(fileName, segLength, workerNumProcXYZ, workerIDXYZ);
    }
    normalizeData();
    return result;
}

// Read one file from disk and save it to the end of the data vector
bool DataManager::ReadOneDataFile(QString strFilePath, Vector3i segLength,
                                  Vector3i workerNumProcessesXYZ,
                                  Vector3i workerIDXYZ) {

    int bufferSize = segLength.x * segLength.y * segLength.z;

    float *pBuffer = AllocateNewDataBuffer(segLength.x, segLength.y, segLength.z);

    // length per dimension
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
    float min, max, delta;

    int iTimelength = getCurrentDataLength();
    int iVolumeSize = getVolumeSize();

    // Get the first local min and max
    min = getMinMaxByIndex(0).x;
    max = getMinMaxByIndex(0).y;

    for (int i = 0 ; i < iTimelength; i++) {
        min = min < getMinMaxByIndex(i).x ? min : getMinMaxByIndex(i).x;
        max = max > getMinMaxByIndex(i).y ? max : getMinMaxByIndex(i).y;
    }

    delta = max - min;
    globalMinMax.x = min;
    globalMinMax.y = max;

    for (int j = 0; j < iTimelength; j++) {
        for (int i = 0; i < iVolumeSize; i++) {
            pDataVector.at(j)[i] -= min;
            pDataVector.at(j)[i] /= delta;
        }
    }
}
