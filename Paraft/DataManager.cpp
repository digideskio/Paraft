#include "DataManager.h"

DataManager::DataManager() {}

DataManager::~DataManager() {
    if (!dataSequence_.empty()) {
        for (auto it = dataSequence_.begin(); it != dataSequence_.end(); it++) {
            delete [] it->second;
        }   // unload data
    }

    if (!tfSequence_.empty()) {
        for (auto it = tfSequence_.begin(); it != tfSequence_.end(); it++) {
            delete [] it->second;
        }   // unload transfer function setting
    }
}

void DataManager::InitTF(const Metadata &meta) {
    if (meta.dynamicTF()) {  // no need to load from file
        tfRes_ = DEFAULT_TF_RES; return;
    }

    ifstream inf(meta.tfPath().c_str(), ios::binary);
    if (!inf) { cout << "cannot load tf setting: " << meta.tfPath() << endl; exit(EXIT_FAILURE); }

    float tfResF = 0.0f;
    inf.read(reinterpret_cast<char*>(&tfResF), sizeof(float));
    if (tfResF < 1) { cout << "tfResolution = " << tfResF << endl; exit(EXIT_FAILURE); }

    tfRes_ = (int)tfResF;
    pStaticTfMap_ = new float[tfRes_];
    inf.read(reinterpret_cast<char*>(pStaticTfMap_), tfRes_*sizeof(float));
    inf.close();
}

void DataManager::SaveMaskVolume(float* pData, const Metadata &meta, const int timestep) {
    char timestamp[21];  // up to 64-bit #
    sprintf(timestamp, (meta.timeFormat()).c_str(), timestep);
    string fpath = meta.path() + "/" + meta.prefix() + timestamp + ".mask";
    ofstream outf(fpath.c_str(), ios::binary);
    if (!outf) { cerr << "cannot output to file: " << fpath.c_str() << endl; return; }

    outf.write(reinterpret_cast<char*>(pData), volumeSize_*sizeof(float));
    outf.close();
}

void DataManager::LoadDataSequence(const Metadata &meta, const int timestep) {
    blockDim_ = meta.volumeDim();
    volumeSize_ = blockDim_.VolumeSize();

    // delete if data is not within [t-2, t+2] of current timestep t
    for (auto it = dataSequence_.begin(); it != dataSequence_.end(); it++) {
        if (it->first < timestep-2 || it->first > timestep+2) {
            delete [] it->second;
            dataSequence_.erase(it);
        }
    }

    for (int t = timestep-2; t <= timestep+2; t++) {
        if (t < meta.start() || t > meta.end() || dataSequence_[t] != NULL) continue;

        char timestamp[21];  // up to 64-bit #
        sprintf(timestamp, meta.timeFormat().c_str(), t);
        string fpath = meta.path() + "/" + meta.prefix() + timestamp + "." + meta.suffix();

        ifstream inf(fpath.c_str(), ios::binary);
        if (!inf) { cout << "cannot read file: " + fpath << endl; exit(1); }

        dataSequence_[t] = new float[volumeSize_];
        inf.read(reinterpret_cast<char*>(dataSequence_[t]), volumeSize_*sizeof(float));
        inf.close();

        // normalize data and returns the position of peak value in [0, tfRes]
        int peakPos = preprocessData(dataSequence_[t], meta.dynamicTF());

        if (meta.dynamicTF()) {
            float *pDynamicTF = new float[tfRes_];
            for (int i = 0; i < tfRes_; i++) {
                pDynamicTF[i] = i < peakPos ? 0.0 : 0.6;
            }
            tfSequence_[t] = pDynamicTF;
        } else {
            tfSequence_[t] = pStaticTfMap_;
        }
    }
}

int DataManager::preprocessData(float *pData, bool dynamicTF) {
    float min = pData[0], max = pData[0];
    for (int i = 1; i < volumeSize_; i++) {
        min = std::min(min, pData[i]);
        max = std::max(max, pData[i]);
    }

    if (dynamicTF) {
        std::map<float, int> dataHistMap;
        for (int i = 0; i < volumeSize_; i++) {
            pData[i] = (pData[i] - min) / (max - min);

            int binIndex = (int)(pData[i] * tfRes_);
            if (dataHistMap.find(binIndex) != dataHistMap.end()) {
                dataHistMap[binIndex]++;
            } else {
                dataHistMap[binIndex] = 1;
            }
        }

        vector<pair<float, int> > tfHistgram(dataHistMap.begin(), dataHistMap.end());
        std::sort(tfHistgram.begin(), tfHistgram.end(), &util::descending);

        return (int)tfHistgram[0].first;
    } else {
        for (int i = 0; i < volumeSize_; i++) {
            pData[i] = (pData[i] - min) / (max - min);
        }
        return -1;
    }
}

void DataManager::normalize(float *pData) {
    float min = pData[0], max = pData[0];
    for (int i = 1; i < volumeSize_; i++) {
        min = min < pData[i] ? min : pData[i];
        max = max > pData[i] ? max : pData[i];
    }
    for (int i = 0; i < volumeSize_; i++) {
        pData[i] = (pData[i] - min) / (max - min);
    }
}
