#include "DataManager.h"

DataManager::DataManager() {
    tfResolution = -1;
    pMaskVolume = NULL;
    dataSequence.clear();
}

DataManager::~DataManager() {
    if (!dataSequence.empty()) {
        for (DataSequence::iterator it = dataSequence.begin(); it != dataSequence.end(); it++) {
            delete [] it->second;
        }   // unload data
    }

    if (pMaskVolume != NULL) {
        delete [] pMaskVolume;
    }
}

void DataManager::CreateNewMaskVolume() {
    pMaskVolume = new float[volumeSize];
    std::fill(pMaskVolume, pMaskVolume+volumeSize, 0);
}

void DataManager::InitTFSettings(const string &filename) {
    ifstream inf(filename.c_str(), ios::binary);
    if (!inf) { cout << "cannot read tf setting: " + filename << endl; exit(1); }

    float tfResF = 0.0f;
    inf.read(reinterpret_cast<char*>(&tfResF), sizeof(float));
    if (tfResF < 1) { cout << "tfResolution = " << tfResF << endl; exit(2); }

    tfResolution = (int)tfResF;
    pTFOpacityMap = new float[tfResolution];
    inf.read(reinterpret_cast<char*>(pTFOpacityMap), tfResolution*sizeof(float));
    inf.close();
}

void DataManager::SaveMaskVolume(float* pData, const Metadata &meta, const int timestep) {
    char timestamp[21];  // up to 64-bit #
    sprintf(timestamp, (meta.timeFormat()).c_str(), timestep);
    string fpath = meta.path() + "/" + meta.prefix() + timestamp + ".mask";
    ofstream outf(fpath.c_str(), ios::binary);
    if (!outf) { cerr << "cannot output to file: " << fpath << endl; return; }

    outf.write(reinterpret_cast<char*>(pData), volumeSize*sizeof(float));
    outf.close();
}

void DataManager::LoadDataSequence(const Metadata &meta, const int timestep) {
    blockDim = meta.volumeDim();
    volumeSize = blockDim.Product();

    // delete if data is not within [t-2, t+2] of current timestep t
    for (DataSequence::iterator it = dataSequence.begin(); it != dataSequence.end(); it++) {
        if (it->first < timestep-2 || it->first > timestep+2) {
            delete [] it->second;
            dataSequence.erase(it);
        }
    }

    for (int t = timestep-2; t <= timestep+2; t++) {
        if (t < meta.start() || t > meta.end() || dataSequence[t] != NULL) {
            continue;
        }

        char timestamp[21];  // up to 64-bit #
        sprintf(timestamp, meta.timeFormat().c_str(), t);
        string fpath = meta.path() + "/" + meta.prefix() + timestamp + "." + meta.suffix();

        ifstream inf(fpath.c_str(), ios::binary);
        if (!inf) { cout << "cannot read file: " + fpath << endl; exit(1); }

        dataSequence[t] = new float[volumeSize];
        inf.read(reinterpret_cast<char*>(dataSequence[t]), volumeSize*sizeof(float));
        inf.close();

        preprocessData(dataSequence[t], meta.remapping());
    }
}

void DataManager::preprocessData(float *pData, bool remapping) {
    if (!remapping) {
        normalize(pData); return;
    }

    // still not working witht the jet dataset ...

    int granularity = 1;
    int binLength = tfResolution * granularity;  // divide [0,1] into 1024 * 1 bins

    float min = pData[0], max = pData[0];
    for (int i = 1; i < volumeSize; i++) {
        min = min < pData[i] ? min : pData[i];
        max = max > pData[i] ? max : pData[i];
    }

    int binIndex = 0;

    map<float, int> histMap;
    for (int i = 0; i < volumeSize; i++) {
        pData[i] = (pData[i] - min) / (max - min);
        binIndex = (int)(pData[i] * binLength);
        if (histMap.find(binIndex) != histMap.end()) {
            histMap[binIndex]++;
        } else {
            histMap[binIndex] = 1;
        }
    }

    vector<pair<float, int> > samples(histMap.begin(), histMap.end());
    std::sort(samples.begin(), samples.end(), &util::descending);

    float peakValue = samples[0].first / binLength;
    min = peakValue - 0.1;
    max = peakValue + 0.1;

    for (int i = 0; i < volumeSize; i++) {
        if (pData[i] < min) {
            pData[i] = 0;
        } else if (pData[i] > max) {
            pData[i] = 1;
        } else {
            pData[i] = (pData[i] - min) / 0.2;
        }
    }
}

void DataManager::normalize(float *pData) {
    float min = pData[0], max = pData[0];
    for (int i = 1; i < volumeSize; i++) {
        min = min < pData[i] ? min : pData[i];
        max = max > pData[i] ? max : pData[i];
    }
    for (int i = 0; i < volumeSize; i++) {
        pData[i] = (pData[i] - min) / (max - min);
    }
}
