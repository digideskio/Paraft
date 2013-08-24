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

void MainWidget::loadData() {
    QFile f("/Users/Yang/Develop/Paraft/Umds/Data/winequality-red.csv");
    if (f.open(QIODevice::ReadOnly)) {
        QString dataString = f.readAll();
        QStringList entries = dataString.split('\n');
        for (QString entry : entries) {
            QStringList tokens = entry.split(';');
            for (QString token : tokens) {
                dataVec.push_back(token.toDouble());
            }
            labelVec.push_back(dataVec.back());
            dataVec.pop_back();  // pop label
            dataMat.push_back(dataVec);
        }
        f.close();
    }

    std::vector<int> indices;
    generateSeedIndices(indices);

    for (int i : indices) {
        seedMat.push_back(dataMat[i]);
    }

    dataVec.clear();
    seedVec.clear();

    for (std::vector<double> entry : dataMat) {
        for (double value : entry) {
            dataVec.push_back(value);
        }
    }

    for (std::vector<double> entry : seedMat) {
        for (double value : entry) {
            seedVec.push_back(value);
        }
    }
}

void MainWidget::generateSeedIndices(std::vector<int> &indices) {
    numSeed = 10;
    numData = dataMat.size();
    for (int i = 0; i < numSeed; i++) {
        int randomIndex;
        do {
            randomIndex = qrand() % numData;
        } while (std::find(indices.begin(), indices.end(), randomIndex) != indices.end());
        indices.push_back(randomIndex);
    }
}

// -------- slots -------- //
void MainWidget::onLoadButtonClicked() {
    loadData();
}

void MainWidget::onSampleButtonClicked() {
    qDebug() << "onSampleButtonClicked";
}

void MainWidget::onProjectButtonClicked() {
    qDebug() << "onProjectButtonClicked";
}
