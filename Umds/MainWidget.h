#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QFile>
#include <QDebug>
#include <QWidget>
#include <QGridLayout>
#include <QPushButton>
#include "ProjectionView.h"

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
    // ---------------------------- //

    void loadData();
    void setupViews();
    void generateSeedIndices(std::vector<int> &indices);

    std::vector<std::vector<double> > dataMat;
    std::vector<std::vector<double> > seedMat;
    std::vector<double> dataVec;
    std::vector<double> seedVec;
    std::vector<double> labelVec;

};

#endif // MAINWIDGET_H
