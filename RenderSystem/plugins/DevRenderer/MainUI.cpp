#include "MainUI.h"

RenderEffectPanel::RenderEffectPanel(MainUI *mainUI, QWidget *parent)
    : QTabWidget(parent), _mainUI(mainUI) {
    // Dev tab
    _devTab = new QWidget(this);
    _component = new QComboBox(this);
    for (int i = 0; i < 5; i++)
        _component->addItem(QString("Component %1").arg(i));
    _preIntEnabled = new QCheckBox(tr("Pre-Integration"), this);
    _segmentEnabled = new QCheckBox(tr("Segmented Ray Casting"), this);

    _mainUI->connectParameter("compIdx", _component);
    _mainUI->getParameterConnecter("compIdx")->setAutoUpdate(false);   // do not auto update to avoid it being set to -1
    _mainUI->connectParameter("preIntEnabled", _preIntEnabled);
    _mainUI->connectParameter("segmentEnabled", _segmentEnabled);
    connect(_mainUI->getParameterConnecter("varNames"), SIGNAL(valueChanged()), this, SLOT(setVarNames()));

    QVBoxLayout *devLayout = new QVBoxLayout();
    devLayout->addWidget(_component);
    devLayout->addWidget(_preIntEnabled);
    devLayout->addWidget(_segmentEnabled);

    _devTab->setLayout(devLayout);
    addTab(_devTab, tr("Dev"));

    // General tab
    _generalTab = new QWidget(this);
    _sampleStep = new ScalarEditor(tr("Sample Step"), this);
    _sampleStep->setMinMax(0.0001f, 0.01f);
    _sampleStep->setSingleStep(0.0001f);
    _sampleStep->spinBox()->setDecimals(4);
    _mainAxis = new QCheckBox(tr("Main Axis"), this);
    _sideAxis = new QCheckBox(tr("Side Axis"), this);
    _boundingBox = new QCheckBox(tr("Bounding Box"), this);

    _mainUI->connectParameter("sampleStep", _sampleStep);
    _mainUI->connectParameter("mainAxis", _mainAxis);
    _mainUI->connectParameter("sideAxis", _sideAxis);
    _mainUI->connectParameter("boundingBox", _boundingBox);

    QVBoxLayout *generalLayout = new QVBoxLayout();
    QGridLayout *generalStepLayout = new QGridLayout();
    _addScalarEditor(_sampleStep, generalStepLayout, 0, 0);
    generalLayout->addLayout(generalStepLayout);
    generalLayout->addWidget(_mainAxis);
    generalLayout->addWidget(_sideAxis);
    generalLayout->addWidget(_boundingBox);
    generalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding));
    _generalTab->setLayout(generalLayout);

    addTab(_generalTab, tr("General"));

    // Light tab
    _lightTab = new QWidget(this);
    _lightEnabled = new QCheckBox(tr("Lighting"), this);
    _ambient = new ScalarEditor(tr("Ambient"), this);
    _diffuse = new ScalarEditor(tr("Diffuse"), this);
    _specular = new ScalarEditor(tr("Specular"), this);
    _shininess = new ScalarEditor(tr("Shininess"), this);
    _shininess->setMinMax(0.0f, 128.0f);

    _ambient->setEnabled(_lightEnabled->isChecked());
    _diffuse->setEnabled(_lightEnabled->isChecked());
    _specular->setEnabled(_lightEnabled->isChecked());
    _shininess->setEnabled(_lightEnabled->isChecked());

    connect(_lightEnabled, SIGNAL(toggled(bool)), _ambient, SLOT(setEnabled(bool)));
    connect(_lightEnabled, SIGNAL(toggled(bool)), _diffuse, SLOT(setEnabled(bool)));
    connect(_lightEnabled, SIGNAL(toggled(bool)), _specular, SLOT(setEnabled(bool)));
    connect(_lightEnabled, SIGNAL(toggled(bool)), _shininess, SLOT(setEnabled(bool)));

    _mainUI->connectParameter("lightEnabled", _lightEnabled);
    _mainUI->connectParameter("ambient", _ambient);
    _mainUI->connectParameter("diffuse", _diffuse);
    _mainUI->connectParameter("specular", _specular);
    _mainUI->connectParameter("shininess", _shininess);

    QGridLayout *lightLayout = new QGridLayout();
    lightLayout->addWidget(_lightEnabled, 0, 0, 1, 3);
    _addScalarEditor(_ambient, lightLayout, 1, 0);
    _addScalarEditor(_diffuse, lightLayout, 2, 0);
    _addScalarEditor(_specular, lightLayout, 3, 0);
    _addScalarEditor(_shininess, lightLayout, 4, 0);
    lightLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding), 5, 0, 1, 3);
    _lightTab->setLayout(lightLayout);

    addTab(_lightTab, tr("Light"));

    // Slice tab
    _sliceTab = new QWidget(this);
    _mouseCamera = new QRadioButton(tr("Camera"), this);
    _mouseTrack = new QRadioButton(tr("Track"), this);
    _mouseMove = new QRadioButton(tr("Move"), this);
    _mouseMode = new ButtonGroup(this);
    _mouseMode->addButton(_mouseCamera, 0);
    _mouseMode->addButton(_mouseTrack, 1);
    _mouseMode->addButton(_mouseMove, 2);
    _mouseMode->toggle(0);
    _slicerList = new QComboBox(this);
    _addSlicer = new QPushButton(tr("Add"), this);
    _delSlicer = new QPushButton(tr("Del"), this);
    _slicerMask = new QCheckBox(tr("Mask"), this);
    _xp = new QPushButton(tr("x+"), this);
    _xp->setFixedWidth(40);
    _yp = new QPushButton(tr("y+"), this);
    _yp->setFixedWidth(40);
    _zp = new QPushButton(tr("z+"), this);
    _zp->setFixedWidth(40);
    _xn = new QPushButton(tr("x-"), this);
    _xn->setFixedWidth(40);
    _yn = new QPushButton(tr("y-"), this);
    _yn->setFixedWidth(40);
    _zn = new QPushButton(tr("z-"), this);
    _zn->setFixedWidth(40);
    _slicerPos = new ScalarEditor(tr("Position"), this);
    _slicerPos->setMinMax(-1.0f, 1.0f);
    _slicerPos->setSingleStep(0.01f);
    _newSlicerId = 0;

    _setSliceEnabled(false);

    _mainUI->connectParameter("mouseMode", _mouseMode);
    _mainUI->connectParameter("slicerIdx", _slicerList);
    _mainUI->connectParameter("slicerMask", _slicerMask);
    _mainUI->connectParameter("slicerPos", _slicerPos);

    connect(_slicerList, SIGNAL(currentIndexChanged(int)), this, SLOT(slicerIdxChanged(int)));
    connect(_addSlicer, SIGNAL(clicked()), this, SLOT(addSlicer()));
    connect(_delSlicer, SIGNAL(clicked()), this, SLOT(delSlicer()));
    connect(_xp, SIGNAL(clicked()), this, SLOT(setSlicerXp()));
    connect(_yp, SIGNAL(clicked()), this, SLOT(setSlicerYp()));
    connect(_zp, SIGNAL(clicked()), this, SLOT(setSlicerZp()));
    connect(_xn, SIGNAL(clicked()), this, SLOT(setSlicerXn()));
    connect(_yn, SIGNAL(clicked()), this, SLOT(setSlicerYn()));
    connect(_zn, SIGNAL(clicked()), this, SLOT(setSlicerZn()));

    QVBoxLayout *sliceLayout = new QVBoxLayout();
    QHBoxLayout *sliceLayout1 = new QHBoxLayout();
    QLabel *mouseLabel = new QLabel(tr("Mouse:"), this);
    sliceLayout1->addWidget(mouseLabel);
    sliceLayout1->addWidget(_mouseCamera);
    sliceLayout1->addWidget(_mouseTrack);
    sliceLayout1->addWidget(_mouseMove);
    sliceLayout1->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));
    sliceLayout->addLayout(sliceLayout1);
    sliceLayout->addWidget(_slicerList);
    QHBoxLayout *sliceLayout2 = new QHBoxLayout();
    sliceLayout2->addWidget(_addSlicer);
    sliceLayout2->addWidget(_delSlicer);
    sliceLayout2->addWidget(_slicerMask);
    sliceLayout->addLayout(sliceLayout2);
    QHBoxLayout *sliceLayout3 = new QHBoxLayout();
    sliceLayout3->addWidget(_xp);
    sliceLayout3->addWidget(_yp);
    sliceLayout3->addWidget(_zp);
    sliceLayout3->addWidget(_xn);
    sliceLayout3->addWidget(_yn);
    sliceLayout3->addWidget(_zn);
    sliceLayout3->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum));
    sliceLayout->addLayout(sliceLayout3);
    QGridLayout *sliceLayout4 = new QGridLayout();
    _addScalarEditor(_slicerPos, sliceLayout4, 0, 0);
    sliceLayout->addLayout(sliceLayout4);
    sliceLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding));
    _sliceTab->setLayout(sliceLayout);

    addTab(_sliceTab, tr("Slice"));

    // Time tab
    _timeTab = new QWidget(this);
    _timestep = new IntScalarEditor(tr("Timestep"), this);
    _timestep->setMinMax(1, 1);
    _totalTimesteps = new QLabel(tr("/1"), this);

    _mainUI->getParameterConnecter("totalSteps")->setType(Parameter::INT_TYPE);
    connect(_mainUI->getParameterConnecter("totalSteps"), SIGNAL(valueChanged(int)), this, SLOT(setTotalTimeSteps(int)));
    _mainUI->connectParameter("timestep", _timestep);

    QGridLayout *timeLayout = new QGridLayout();
    _addIntScalarEditor(_timestep, timeLayout, 0, 0);
    timeLayout->addWidget(_totalTimesteps, 0, 3);
    timeLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding), 1, 0, 1, 4);
    _timeTab->setLayout(timeLayout);

    addTab(_timeTab, tr("Time"));
}

void RenderEffectPanel::setVarNames() {
    qDebug("setVarNames(%s)", _mainUI->getParameter("varNames")->name().c_str());
    _component->blockSignals(true);
    _component->clear();
    Parameter &param = *_mainUI->getParameter("varNames");
    for (int i = 0; i < param.size(); i++)
        _component->addItem(QString::fromStdString(param[i].toString()));
    qDebug("%s: compIdx = %d", __FUNCTION__, _mainUI->getParameter("compIdx")->toInt());
    _component->setCurrentIndex(_mainUI->getParameter("compIdx")->toInt());
    _component->blockSignals(false);
}

void RenderEffectPanel::setSlicers(int slicerCount, int slicerIdx) {
    _slicerList->blockSignals(true);
    _slicerList->clear();
    _newSlicerId = 0;
    for (int i = 0; i < slicerCount; i++) {
        _slicerList->addItem(tr("Slicer%1").arg(_newSlicerId++));
    }
    if (_slicerList->count() == 0) {
        _setSliceEnabled(false);
    } else {
        qDebug("slicerIdx=%d", slicerIdx);
        _slicerList->setCurrentIndex(slicerIdx);
        _setSliceEnabled(true);
    }
    _slicerList->blockSignals(false);   ////
}

void RenderEffectPanel::_addScalarEditor(ScalarEditor *editor, QGridLayout *layout, int row, int column) {
    layout->addWidget(editor->label(), row, column);
    layout->addWidget(editor->slider(), row, column + 1);
    layout->addWidget(editor->spinBox(), row, column + 2);
}

void RenderEffectPanel::_addIntScalarEditor(IntScalarEditor *editor, QGridLayout *layout, int row, int column) {
    layout->addWidget(editor->label(), row, column);
    layout->addWidget(editor->slider(), row, column + 1);
    layout->addWidget(editor->spinBox(), row, column + 2);
}

void RenderEffectPanel::_setSliceEnabled(bool enable) {
    _delSlicer->setEnabled(enable);
    _slicerMask->setEnabled(enable);
    _xp->setEnabled(enable);
    _yp->setEnabled(enable);
    _zp->setEnabled(enable);
    _xn->setEnabled(enable);
    _yn->setEnabled(enable);
    _zn->setEnabled(enable);
    _slicerPos->setEnabled(enable);
}

void RenderEffectPanel::slicerIdxChanged(int idx) {
    //emit actionTriggered("slicerIdxChanged");
}

void RenderEffectPanel::addSlicer() {
    emit actionTriggered("addSlicer");
    _slicerList->addItem(tr("Slicer%1").arg(_newSlicerId++));
    _slicerList->setCurrentIndex(_slicerList->count() - 1);
    _setSliceEnabled(true);
}

void RenderEffectPanel::delSlicer() {
    int idx = _slicerList->currentIndex();
    if (idx >= 0) {
        emit actionTriggered("delSlicer");
        _slicerList->removeItem(idx);
        idx = _slicerList->currentIndex();
        if (idx >= 0) {
            //_slicerPos->setValue(_slicerListPos[idx]);
            //_slicerMask->setChecked(_slicerListMask[idx]);
        } else {
            _setSliceEnabled(false);
        }
    }
}

void RenderEffectPanel::setSlicerXp() {
    emit actionTriggered("setSlicerXp");
}

void RenderEffectPanel::setSlicerYp() {
    emit actionTriggered("setSlicerYp");
}

void RenderEffectPanel::setSlicerZp() {
    emit actionTriggered("setSlicerZp");
}

void RenderEffectPanel::setSlicerXn() {
    emit actionTriggered("setSlicerXn");
}

void RenderEffectPanel::setSlicerYn() {
    emit actionTriggered("setSlicerYn");
}

void RenderEffectPanel::setSlicerZn() {
    emit actionTriggered("setSlicerZn");
}

void RenderEffectPanel::setTotalTimeSteps(int timesteps) {
    if (_timestep != 0) {
        _timestep->setMinMax(1, timesteps);
        _timestep->setEnabled(timesteps > 1);
    }
    if (_totalTimesteps != 0) {
        _totalTimesteps->setText(QString("/%1").arg(timesteps));
        _totalTimesteps->setEnabled(timesteps > 1);
    }
}

GeneralToolBar::GeneralToolBar(const QString &title, MainUI *mainUI, QWidget *parent)
    : QToolBar(title, parent), _mainUI(mainUI) {
    setIconSize(QSize(20, 20));

    _open = new QAction(tr("Open"), this);
    _open->setIcon(QIcon(_mainUI->workingPath() + "/icons/glyphicons_144_folder_open.png"));

    _resetCamera = new QAction(tr("Reset Camera"), this);
    _resetCamera->setIcon(QIcon(_mainUI->workingPath() + "/icons/glyphicons_011_camera.png"));
    _mainAxis = new QAction(tr("Main Axis"), this);
    _mainAxis->setCheckable(true);
    _sideAxis = new QAction(tr("Side Axis"), this);
    _sideAxis->setCheckable(true);
    _boundingBox = new QAction(tr("Bounding Box"), this);
    _boundingBox->setCheckable(true);
    _lighting = new QAction(tr("Lighting"), this);
    _lighting->setIcon(QIcon(_mainUI->workingPath() + "/icons/glyphicons_189_brightness_increase.png"));
    _lighting->setCheckable(true);

    connect(_resetCamera, SIGNAL(triggered()), this, SLOT(resetCamera()));
    _mainUI->connectParameter("mainAxis", _mainAxis);
    _mainUI->connectParameter("sideAxis", _sideAxis);
    _mainUI->connectParameter("boundingBox", _boundingBox);
    _mainUI->connectParameter("lightEnabled", _lighting);

    addAction(_open);
    QAction *sepAct = new QAction(this);
    sepAct->setSeparator(true);
    addAction(sepAct);
    addAction(_resetCamera);
    addAction(_mainAxis);
    addAction(_sideAxis);
    addAction(_boundingBox);
    addAction(_lighting);
}

MainUI::MainUI(const QString &workingPath) : _workingPath(workingPath) {
    _renderEff = new RenderEffectPanel(this);
    connect(_renderEff, SIGNAL(actionTriggered(const QString &)), this, SIGNAL(actionTriggered(const QString &)));
    _generalToolBar = new GeneralToolBar(tr("General"), this);
    _tfEditor = new QTFEditor(1024);
}

MainUI::~MainUI() {
    qDebug("~MainUI()");
}

void MainUI::setParameterSet(QParameterSet *ps) {
    _psc.unbind();
    _psc.bind(ps);
    _psc.update();
}

void MainUI::unsetParameterSet() {
    _psc.unbind();
}

void MainUI::connectParameter(const String &name, QAction *editor) {
    _psc[name].setType(Parameter::BOOL_TYPE);
    connect(&_psc[name], SIGNAL(valueChanged(bool)), editor, SLOT(setChecked(bool)));
    connect(editor, SIGNAL(toggled(bool)), &_psc[name], SLOT(setValue(bool)));
}

void MainUI::connectParameter(const String &name, QCheckBox *editor) {
    _psc[name].setType(Parameter::BOOL_TYPE);
    QObject::connect(&_psc[name], SIGNAL(valueChanged(bool)), editor, SLOT(setChecked(bool)));
    QObject::connect(editor, SIGNAL(toggled(bool)), &_psc[name], SLOT(setValue(bool)));
}

void MainUI::connectParameter(const String &name, QComboBox *editor) {
    _psc[name].setType(Parameter::INT_TYPE);
    QObject::connect(&_psc[name], SIGNAL(valueChanged(int)), editor, SLOT(setCurrentIndex(int)));
    QObject::connect(editor, SIGNAL(currentIndexChanged(int)), &_psc[name], SLOT(setValue(int)));
}

void MainUI::connectParameter(const String &name, ScalarEditor *editor) {
    _psc[name].setType(Parameter::FLOAT_TYPE);
    QObject::connect(&_psc[name], SIGNAL(valueChanged(float)), editor, SLOT(setValue(float)));
    QObject::connect(editor, SIGNAL(valueChanged(float)), &_psc[name], SLOT(setValue(float)));
}

void MainUI::connectParameter(const String &name, IntScalarEditor *editor) {
    _psc[name].setType(Parameter::INT_TYPE);
    QObject::connect(&_psc[name], SIGNAL(valueChanged(int)), editor, SLOT(setValue(int)));
    QObject::connect(editor, SIGNAL(valueChanged(int)), &_psc[name], SLOT(setValue(int)));
}

void MainUI::connectParameter(const String &name, ButtonGroup *editor) {
    _psc[name].setType(Parameter::INT_TYPE);
    QObject::connect(&_psc[name], SIGNAL(valueChanged(int)), editor, SLOT(toggle(int)));
    QObject::connect(editor, SIGNAL(buttonClicked(int)), &_psc[name], SLOT(setValue(int)));
}
