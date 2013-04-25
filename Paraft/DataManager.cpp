#include "DataManager.h"

DataManager::DataManager() {
    tfResolution = -1;
    pMaskVolume = NULL;
    dataSequence.clear();
}

DataManager::~DataManager() {
    if (!dataSequence.empty()) {
        for (DataSequence::iterator it = dataSequence.begin(); it != dataSequence.end(); it++) {
            delete [] it->second;   // pointer to each volume
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

void DataManager::LoadDataSequence(Metadata *meta, int timestep) {
    blockDim = meta->volumeDim;
    volumeSize = blockDim.Product();

    cout << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << endl;

    // delete if data is not within [t-2, t+2] of current timestep t
    for (DataSequence::iterator it = dataSequence.begin(); it != dataSequence.end(); it++) {
        if (it->first < timestep-2 || it->first > timestep+2) {
            delete [] it->second;
            dataSequence.erase(it);
        }
    }

    for (int t = timestep-2; t <= timestep+2; t++) {
        if (t < meta->start || t > meta->end || dataSequence[t] != NULL) {
            continue;
        }

        char timestamp[21];  // up to 64-bit #
        sprintf(timestamp, "%03d", t);
        string fpath = meta->path + "/" + meta->prefix + timestamp + "." + meta->surfix;

        ifstream inf(fpath.c_str(), ios::binary);
        if (!inf) { cout << "cannot read file: " + fpath << endl; exit(1); }

        float *pData = new float[volumeSize];
        inf.read(reinterpret_cast<char*>(pData), volumeSize*sizeof(float));
        inf.close();

        nomalize(pData);
        dataSequence[t] = pData;
    }
}

// load data by given volume data pointers, for in-situ mode
void DataManager::InSituLoadDataSequence(int timestep, float *pData) {
    // delete if data is not within [t-2, t+2] of current timestep t
    for (DataSequence::iterator it = dataSequence.begin(); it != dataSequence.end(); it++) {
        if (it->first < timestep-2 || it->first > timestep+2) {
            delete [] it->second;
            dataSequence.erase(it);
        }
    }
    dataSequence[timestep] = pData;
}

// collectively load volume data from disk, for batch mode
void DataManager::CollectiveLoadDataSequence(Vector3i gridDim, Vector3i blockIdx, Metadata *meta, int timestep) {
    blockDim = meta->volumeDim / gridDim;
    volumeSize = blockDim.Product();

    // delete if data is not within [t-2, t+2] of current timestep t
    for (DataSequence::iterator it = dataSequence.begin(); it != dataSequence.end(); it++) {
        if (it->first < timestep-2 || it->first > timestep+2) {
            delete [] it->second;
            dataSequence.erase(it);
        }
    }

    for (int t = timestep-2; t <= timestep+2; t++) {
        if (t < meta->start || t > meta->end) {
            continue;  // only [t-2, t-1, t, t+1, t+2]
        }

        // 1. allocate new data buffer then add pointer to map
        if (dataSequence[t] == NULL) {
            dataSequence[t] = new float[volumeSize];
        }

        // 2. get file name from time index
        string filePath;
        char timestep[21];  // hold up to 64-bits
        sprintf(timestep, "%03d", t);
        filePath = meta->path+"/"+meta->prefix+timestep+"."+meta->surfix;

        char *filename = new char[filePath.size() + 1];
        std::copy(filePath.begin(), filePath.end(), filename);
        filename[filePath.size()] = '\0';

        // 3. read corresponding file using mpi collective io
        int *gsizes = (blockDim * gridDim).GetPointer();
        int *subsizes = blockDim.GetPointer();
        int *starts = (blockDim * blockIdx).GetPointer();

        MPI_Datatype filetype;
        MPI_Type_create_subarray(3, gsizes, subsizes, starts, MPI_ORDER_FORTRAN, MPI_FLOAT, &filetype);
        MPI_Type_commit(&filetype);

        MPI_File file;
        int not_exist = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        if (not_exist) printf("%s not exist.\n", filename);

        char* native = "native";
        MPI_File_set_view(file, 0, MPI_FLOAT, filetype, native, MPI_INFO_NULL);
        MPI_File_read_all(file, dataSequence[t], volumeSize, MPI_FLOAT, MPI_STATUS_IGNORE);

        cout << "t: " << t << "\t";
        equalize(dataSequence[t]);

        MPI_File_close(&file);
        MPI_Type_free(&filetype);
        delete[] filename;
    }
}

void DataManager::nomalize(float *pData) {
    float min = pData[0], max = pData[0];
    for (int i = 1; i < volumeSize; i++) {
        min = min < pData[i] ? min : pData[i];
        max = max > pData[i] ? max : pData[i];
    }
    for (int i = 0; i < volumeSize; i++) {
        pData[i] = (pData[i] - min) / (max - min);
    }
}

bool compare(const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) {
    return lhs.second > rhs.second;  // descending order
}

void DataManager::equalize(float *pData) {
    std::map<float, int> histMap;
    int granularity = 10;
    int binLength = tfResolution * granularity;

    float min = pData[0], max = pData[0];
    for (int i = 1; i < volumeSize; i++) {
        min = min < pData[i] ? min : pData[i];
        max = max > pData[i] ? max : pData[i];
    }

//    MPI_Allreduce(&min, &min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
//    MPI_Allreduce(&max, &max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    std::cout << "g: min: " << min << " max: " << max << endl;

    float range = max - min;
    int binIndex = 0;

    for (int i = 0; i < volumeSize; i++) {
//        if (IS_BIG_ENDIAN) ReverseEndian(&pData[i]);
        pData[i] = (pData[i] - min) / range;
        binIndex = (int)(pData[i] * binLength);
        if (histMap.find(binIndex) != histMap.end()) {
            histMap[binIndex]++;
        } else {
            histMap[binIndex] = 1;
        }
    }

    vector<pair<float, int> > histVector(histMap.begin(), histMap.end());
    std::sort(histVector.begin(), histVector.end(), &compare);

    histMap.clear();
    for (int i = 0; i < tfResolution; i++) {
        histMap[histVector[i].first] = i;
    }
    for (int i = 0; i < volumeSize; i++) {
        binIndex = (int)(pData[i] * binLength);
        pData[i] = (float)histMap[binIndex] / tfResolution;
    }
}
