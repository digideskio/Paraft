#include "DataManager.h"

DataManager::DataManager(QObject *parent) : QObject(parent) {
    pDataHolder = new DataHolder();
}

DataManager::~DataManager() {
    pDataHolder->~DataHolder();
}

bool DataManager::ReadDataSequence(string filePath, string prefix, string suffix,
                                   int iStart, int iEnd, Vector3d dimXYZ,
                                   Vector3d workerNumProcXYZ, Vector3d workerIDXYZ) {

    Vector3d segLength;
    segLength.x = dimXYZ.x / workerNumProcXYZ.x;
    segLength.y = dimXYZ.y / workerNumProcXYZ.y;
    segLength.z = dimXYZ.z / workerNumProcXYZ.z;

    pDataHolder->SetVolumeDimension(segLength.x, segLength.y, segLength.z);

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
bool DataManager::ReadOneDataFile(QString strFilePath, Vector3d segLength,
                                  Vector3d workerNumProcessesXYZ,
                                  Vector3d workerIDXYZ) {

        int bufferSize = segLength.x * segLength.y * segLength.z;
        float *pBuffer = pDataHolder->AllocNewDataBuffer(segLength.x,
                                                         segLength.y,
                                                         segLength.z);

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

        pDataHolder->CalculateLocalMinMax();
        return true;
}

void DataManager::normalizeData() {
    float min, max, delta;
    int iTimelength = pDataHolder->GetCurrentDataLength();
    int iVolumeSize = pDataHolder->GetVolumeSize();

    // Get the first local min and max
    min = pDataHolder->getMinMaxByIndex(0)->x();
    max = pDataHolder->getMinMaxByIndex(0)->y();

    for (int i = 0 ; i < iTimelength; i++) {
        min = min < pDataHolder->getMinMaxByIndex(i)->x()?
                    min : pDataHolder->getMinMaxByIndex(i)->x();
        max = max > pDataHolder->getMinMaxByIndex(i)->y()?
                    max : pDataHolder->getMinMaxByIndex(i)->y();
    }

    delta = max - min;
    globalMinMax.setX(min);
    globalMinMax.setY(max);

    for (int j = 0; j < iTimelength; j++) {
        for (int i = 0; i < iVolumeSize; i++) {
            pDataHolder->GetDataPointerByIndex(j)[i] -= min;
            pDataHolder->GetDataPointerByIndex(j)[i] /= delta;
        }
    }
}
