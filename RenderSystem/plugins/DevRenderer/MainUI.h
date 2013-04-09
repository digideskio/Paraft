#ifndef MAINUI_H
#define MAINUI_H

#include <QtGui>

#include "QParameterSet.h"
#include "ParameterEditor.h"

// VisKit
#include "QTFEditor.h"

class RenderEffectPanel;
class GeneralToolBar;

//
// shared UI provided by MainWindow
//
class MainUI : public QObject
{
    Q_OBJECT
public:
    MainUI(const QString &workingPath);
    ~MainUI();
    void setWorkingPath(const QString &path) { _workingPath = path; }
    const QString &workingPath() const { return _workingPath; }
    void setParameterSet(QParameterSet *ps);
    void unsetParameterSet();
    RenderEffectPanel *getRenderEffectPanel() { return _renderEff; }
    GeneralToolBar *getGeneralToolBar() { return _generalToolBar; }
    Parameter *getParameter(const String &name) { return _psc.getParameter(name); }
    QParameterConnecter *getParameterConnecter(const String &name) { return &_psc[name]; }
    void connectParameter(const String &name, QAction *editor);
    void connectParameter(const String &name, QCheckBox *editor);
    void connectParameter(const String &name, QComboBox *editor);
    void connectParameter(const String &name, ScalarEditor *editor);
    void connectParameter(const String &name, IntScalarEditor *editor);
    void connectParameter(const String &name, ButtonGroup *editor);

    QTFEditor *getTFEditor() { return _tfEditor; }

    void triggerAction(const QString &name) { emit actionTriggered(name); }

protected:
    QString _workingPath;

    QParameterSetConnecter _psc;
    RenderEffectPanel *_renderEff;

    GeneralToolBar *_generalToolBar;

    QTFEditor *_tfEditor;

signals:
    void actionTriggered(const QString &name);
};

class RenderEffectPanel : public QTabWidget
{
    Q_OBJECT
public:
    RenderEffectPanel(MainUI *mainUI, QWidget *parent = 0);
    ~RenderEffectPanel() { qDebug("~RenderEffectPanel()"); }
    //void setManager(ControlPanels *cp) { _manager = cp; }
    //void setSlicers(int slicerIdx, QList<float> slicerPos, QList<bool> slicerMask);
    void setSlicers(int slicerCount, int slicerIdx);

protected:
    void _addScalarEditor(ScalarEditor *editor, QGridLayout *layout, int row, int column);
    void _addIntScalarEditor(IntScalarEditor *editor, QGridLayout *layout, int row, int column);
    void _setSliceEnabled(bool enable);

protected:
    MainUI *_mainUI;

    QWidget *_devTab;
    QComboBox *_component;
    QCheckBox *_preIntEnabled;
    //QPushButton *_genPreIntTable;
    //QPushButton *_genPreIntTable2;
    QCheckBox *_segmentEnabled;

    QWidget *_generalTab;
    ScalarEditor *_sampleStep;
    QCheckBox *_mainAxis;
    QCheckBox *_sideAxis;
    QCheckBox *_boundingBox;

    QWidget *_lightTab;
    QCheckBox *_lightEnabled;
    ScalarEditor *_ambient;
    ScalarEditor *_diffuse;
    ScalarEditor *_specular;
    ScalarEditor *_shininess;

    QWidget *_sliceTab;
    QRadioButton *_mouseCamera;
    QRadioButton *_mouseTrack;
    QRadioButton *_mouseMove;
    ButtonGroup *_mouseMode;
    QComboBox *_slicerList;
    //QList<float> _slicerListPos;
    //QList<bool> _slicerListMask;
    int _newSlicerId;
    QPushButton *_addSlicer;
    QPushButton *_delSlicer;
    QCheckBox *_slicerMask;
    QPushButton *_xp, *_yp, *_zp, *_xn, *_yn, *_zn;
    ScalarEditor *_slicerPos;

    QWidget *_timeTab;
    IntScalarEditor *_timestep;
    QLabel *_totalTimesteps;

public slots:
    //void genPreIntTable();
    //void genPreIntTable2();
    void setVarNames();
    void slicerIdxChanged(int idx);
    void addSlicer();
    void delSlicer();
    void setSlicerXp();
    void setSlicerYp();
    void setSlicerZp();
    void setSlicerXn();
    void setSlicerYn();
    void setSlicerZn();
    //void setSlicerMask(bool enable);
    //void slicerPosChanged(float val);
    void setTotalTimeSteps(int timesteps);

signals:
    void actionTriggered(const QString &name);
};

class GeneralToolBar : public QToolBar
{
    Q_OBJECT
public:
    GeneralToolBar(const QString &title, MainUI *mainUI, QWidget *parent = 0);
    QAction *getOpenAction() { return _open; }

protected:
    MainUI *_mainUI;

    QAction *_open;

    QAction *_resetCamera;
    QAction *_mainAxis;
    QAction *_sideAxis;
    QAction *_boundingBox;

    QAction *_lighting;

public slots:
    void resetCamera() { _mainUI->triggerAction("resetCamera"); }
};

#endif // MAINUI_H
