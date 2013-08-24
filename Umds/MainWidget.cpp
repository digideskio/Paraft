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
}

void MainWidget::setupViews() {
    this->resize(DEFAULT_WIDTH, DEFAULT_HEIGHT);

    projView = new ProjectionView();

    loadButton = new QPushButton("Load");
    sampleButton = new QPushButton("Sample");
    projectButton = new QPushButton("Project");

    connect(loadButton, SIGNAL(clicked()), this, SLOT(onLoadButtonClicked()));
    connect(sampleButton, SIGNAL(clicked()), this, SLOT(onSampleButtonClicked()));
    connect(projectButton, SIGNAL(clicked()), this, SLOT(onProjectButtonClicked()));

    gridLayout = new QGridLayout();  // 8*8
    gridLayout->addWidget(projView, 1, 0, 8, 8);
    gridLayout->addWidget(loadButton, 0, 0, 1, 1);
    gridLayout->addWidget(sampleButton, 0, 1, 1, 1);
    gridLayout->addWidget(projectButton, 0, 2, 1, 1);

    this->setLayout(gridLayout);
}

void MainWidget::spreadSeed() {
    for (unsigned int i = 0; i < seedLabelVec.size(); i++) {
        projView->addNode(i, static_cast<int>(seedLabelVec[i]));
    }
}

void MainWidget::loadData() {
    QFile f("/Users/Yang/Develop/Paraft/Umds/Data/winequality-red.csv");
    if (f.open(QIODevice::ReadOnly)) {
        QString dataString = f.readAll();
        QStringList entries = dataString.split('\n');
        for (int row = 0; row < entries.size(); row++) {
            QString entry = entries[row];
            if (entry.isEmpty()) continue;
            QStringList tokens = entries[row].split(';');
            if (tokens.isEmpty()) continue;
            for (QString token : tokens)
                dataVec.push_back(token.toDouble());
            dataLabelVec.push_back(dataVec.back());
            dataVec.pop_back();  // pop label
            dataMat[row] = dataVec;
        }
        f.close();
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

    qDebug() << "Total data entry: " << dataLabelVec.size();
}

std::vector<int> MainWidget::generateSeedIndices() {
    numSeed = 10;
    numData = dataMat.size();
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
    qDebug() << "onProjectButtonClicked";
}
