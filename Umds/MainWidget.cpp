#include "MainWidget.h"

MainWidget::MainWidget(QWidget *parent) : QWidget(parent) {
    setupViews();
}

MainWidget::~MainWidget() {
    if (gridLayout)     delete gridLayout;
    if (projView)       delete projView;
    if (loadButton)     delete loadButton;
    if (sampleButton)   delete sampleButton;
    if (projectButton)  delete projectButton;
    if (lamp)           delete lamp;
}

void MainWidget::setupViews() {
    this->resize(DEFAULT_WIDTH, DEFAULT_HEIGHT);

    projView = new ProjectionView();

    loadButton = new QPushButton("Load");
    sampleButton = new QPushButton("Sample");
    projectButton = new QPushButton("Project");
    resetButton = new QPushButton("Reset");

    connect(loadButton, SIGNAL(clicked()), this, SLOT(onLoadButtonClicked()));
    connect(sampleButton, SIGNAL(clicked()), this, SLOT(onSampleButtonClicked()));
    connect(projectButton, SIGNAL(clicked()), this, SLOT(onProjectButtonClicked()));
    connect(resetButton, SIGNAL(clicked()), this, SLOT(onResetButtonClicked()));

    gridLayout = new QGridLayout();  // 8*8
    gridLayout->addWidget(projView, 1, 0, 8, 8);
    gridLayout->addWidget(loadButton, 0, 0, 1, 1);
    gridLayout->addWidget(sampleButton, 0, 1, 1, 1);
    gridLayout->addWidget(projectButton, 0, 2, 1, 1);

    this->setLayout(gridLayout);
}

void MainWidget::spreadSeed() {
    for (unsigned int i = 0; i < seedLabelVec.size(); i++)
        projView->addNode(static_cast<int>(seedLabelVec[i]), -20*i, 20*i, true);
}

void MainWidget::projectData() {
    std::vector<double> projSeed = projView->getProjSeed();
    std::vector<double> projData;
    lamp->project(seedVec, projSeed, dataVec, projData);
    for (double value : projData)
        qDebug() << value;

    for (unsigned int i = 0; i < projData.size()/2; i++) {
        int x = static_cast<int>(projData[i*2]);
        int y = static_cast<int>(projData[i*2+1]);
        projView->addNode(static_cast<int>(dataLabelVec[i]), x, y, false);
    }
}

void MainWidget::reset() {

}

void MainWidget::loadData() {
    dataMat.clear();
    seedMat.clear();
    dataVec.clear();
    seedVec.clear();
    dataLabelVec.clear();
    seedLabelVec.clear();

    QFile f("/Users/Yang/Develop/Paraft/Umds/Data/winequality-red.csv");
    if (f.open(QIODevice::ReadOnly)) {
        QString dataString = f.readAll();
        QStringList entries = dataString.split('\n');
        for (int row = 0; row < entries.size(); row++) {
            QString entry = entries[row];
            if (entry.isEmpty()) continue;
            QStringList tokens = entries[row].split(';');
            if (tokens.isEmpty()) continue;
            dataVec.clear();
            for (QString token : tokens)
                dataVec.push_back(token.toDouble());
            dataLabelVec.push_back(dataVec.back());
            dataVec.pop_back();  // pop label
            dataMat[row] = dataVec;
        }
        f.close();
    }

    numData = static_cast<int>(dataMat.size());
    int numDataDim = static_cast<int>(dataVec.size());
    qDebug() << "numDataDim: " << numDataDim;

    // -- normalize to gaussian distribution with µ = 0 -- //
    std::vector<double> tempDataVec(numDataDim, 0.0);
    for (auto entry : dataMat)
        for (int i = 0; i < numDataDim; i++)
            tempDataVec[i] += entry.second[i];

    for (unsigned int i = 0; i < tempDataVec.size(); i++) {
        std::cout << tempDataVec[i] << "->";
        tempDataVec[i] /= numData;
        std::cout << tempDataVec[i] << std::endl;
    }

    for (auto entry : dataMat)
        for (int i = 0; i < numDataDim; i++) {
            std::cout << entry.second[i] << "->";
            entry.second[i] = entry.second[i] / tempDataVec[i] - 1;
            std::cout << entry.second[i] << std::endl;
        }

    std::vector<int> indices = generateSeedIndices();
    for (int i = 0; i < numSeed; i++) {
        int seedId = indices[i];
        seedMat[i] = dataMat[seedId];
        seedLabelVec.push_back(dataLabelVec[seedId]);
    }

    dataVec.clear();
    seedVec.clear();

    for (auto entry : dataMat)
        for (double value : entry.second)
            dataVec.push_back(value);

    for (auto entry : seedMat)
        for (double value : entry.second)
            seedVec.push_back(value);

    lamp = new Lamp(numData, numDataDim, numSeed, 2);

    qDebug() << "Total data entry: " << dataLabelVec.size();
}

std::vector<int> MainWidget::generateSeedIndices() {
    numSeed = NUM_SEED;
    numData = numData < NUM_DATA ? numData : NUM_DATA;
    std::vector<int> indices;
    for (int i = 0; i < numSeed; i++) {
        int randomIndex;
        do { randomIndex = qrand() % numData;
        } while (std::find(indices.begin(), indices.end(), randomIndex) != indices.end());
        indices.push_back(randomIndex);
    }
    return indices;
}

// -------- slots -------- //
void MainWidget::onLoadButtonClicked() {
    loadData();
}

void MainWidget::onSampleButtonClicked() {
    spreadSeed();
}

void MainWidget::onProjectButtonClicked() {
    projectData();
}

void MainWidget::onResetButtonClicked() {
    reset();
}
