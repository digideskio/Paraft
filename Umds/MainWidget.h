#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QFile>
#include <QDebug>
#include <QWidget>
#include <QGridLayout>
#include <QPushButton>
#include <unordered_map>
#include "ProjectionView.h"
#include "Lamp.h"

class MainWidget : public QWidget {
    Q_OBJECT
    
public:
    MainWidget(QWidget *parent = 0);
    ~MainWidget();

public slots:
    void onLoadButtonClicked();
    void onSampleButtonClicked();
    void onProjectButtonClicked();

private:
    const int DEFAULT_WIDTH = 1280;
    const int DEFAULT_HEIGHT = 800;

    int numSeed;
    int numData;

    QGridLayout *gridLayout = nullptr;
    ProjectionView *projView = nullptr;

    QPushButton *loadButton = nullptr;
    QPushButton *sampleButton = nullptr;
    QPushButton *projectButton = nullptr;

    Lamp *lamp = nullptr;
    // ---------------------------- //

    void loadData();
    void setupViews();
    void spreadSeed();
    void projectData();
    std::vector<int> generateSeedIndices();

    std::unordered_map<int, std::vector<double> > dataMat;
    std::unordered_map<int, std::vector<double> > seedMat;
    std::vector<double> dataVec;
    std::vector<double> seedVec;
    std::vector<double> dataLabelVec;
    std::vector<double> seedLabelVec;
};

#endif // MAINWIDGET_H
