#include "VolumeRenderWindow.h"

#define debug(msg) qDebug("%s(): %s", __FUNCTION__, msg)

VolumeRenderWindow::VolumeRenderWindow(QWidget *parent, const QGLWidget *shareWidget)
    : QRenderWindow(parent, shareWidget) {
}

VolumeRenderWindow::VolumeRenderWindow(VolumeModel *model, QWidget *parent, const QGLWidget *shareWidget)
    : QRenderWindow(parent, shareWidget), _model(model) {
    qDebug("Model name: %s", _model->name().c_str());
    setFocusPolicy(Qt::StrongFocus);    // important when there are more than one sub-window
    setWindowTitle(QString::fromStdString(model->name()));
}

VolumeRenderWindow::~VolumeRenderWindow() {
    qDebug("~VolumeRenderWindow()");

    delete _bufferTex[0];
    delete _bufferTex[1];
    delete _bufferFbo[0];
    delete _bufferFbo[1];

    //delete m_data;
    delete m_histogram;

    if (_renderer != nullptr) {
        delete _renderer;
    }
}

void VolumeRenderWindow::initializeGL() {
    QRenderWindow::initializeGL();

    if (_workingPath.isEmpty()) {
        _workingPath = QDir::currentPath();
    }

    // blocks
    _model->initSubblocks(Vector3i(2, 2, 1), 4);

    Vector3i dim = _model->dim();
    Vector3f scaledDim = _model->scaledDim();

    box.setPitch(Vector3((double)scaledDim.x, (double)scaledDim.y, (double)scaledDim.z));

    sampleSpacing = 0.001f;
    enableLight = false;

    _ps["projection"].setValue(m_cameraProjection == Camera::Perspective ? 1.0f : 0.0f);
    _ps["compIdx"].setValue(0);
    _ps["compCount"].setValue(_model->varCount());
    _ps["preIntEnabled"].setValue(false);

    for (int i = 0; i < _model->varCount(); i++) {
        _ps["varNames"][i].setValue(_model->varName(i));
    }

    _ps["segmentEnabled"].setValue(false);
    _ps["sampleStep"].setValue(0.001f);
    _ps["mainAxis"].setValue(true);
    _ps["sideAxis"].setValue(true);
    _ps["boundingBox"].setValue(true);

    _ps["lightEnabled"].setValue(false);
    _ps["ambient"].setValue(0.4f);
    _ps["diffuse"].setValue(0.7f);
    _ps["specular"].setValue(1.0f);
    _ps["shininess"].setValue(20.0f);

    _ps["mouseMode"].setValue(0);
    _ps["slicerIdx"].setValue(-1);
    _ps["slicerMask"].setValue(true);
    _ps["slicerPos"].setValue(0.0f);

    _ps["abc"]["def"].setValue(0.0f);
    _ps["xyz"][2].setValue("hello world");
    _ps["abc"]["ghi"][1].setValue(Vector2i(3, 4));

    qDebug("[xyz][2] = %s", _ps.getParameter("xyz[2]")->toCString());

    _ps["totalSteps"].setValue(_model->stepCount());
    _ps["timestep"].setValue(1);

    qDebug("init texture");

    // init TF
    m_transferFunction.backgroundColor = QColor(0, 0, 0);

    qDebug("Init tf...");
    _mainUI->getTFEditor()->getColorMap()->disconnect();
    _mainUI->getTFEditor()->getTFPanel()->disconnect();
    _mainUI->getTFEditor()->getTFPanel()->loadSettings(m_transferFunction);

    qDebug("Init histogram...");
    _mainUI->getTFEditor()->getHistogram()->clear();
    float *rawData = _model->data();
    size_t dataSize = dim.x * dim.y * dim.z;
    for (size_t i = 0; i < dataSize; i++) {
        _mainUI->getTFEditor()->incrementHistogram(rawData[i]);
    }
    m_histogram = new Histogram(256);
    *m_histogram = *(_mainUI->getTFEditor()->getHistogram());

    m_camera.setFocus(scaledDim.x * 0.5, scaledDim.y * 0.5, scaledDim.z * 0.5);
    m_camera.saveOptions(_defaultCamera);

    // init renderer
    _renderer = new VolumeRenderer(*this, *_mainUI->getTFEditor(), _ps, *_model, box, _workingPath.toStdString());
    _segmentedRenderer = nullptr;

    // segmented
    _bufferTex[0] = new MSLib::GLTexture2D(GL_RGBA32F, width(), height(), 0, GL_RGBA, GL_FLOAT, 0);
    _bufferFbo[0] = new MSLib::GLFramebufferObject();
    _bufferFbo[0]->bind();
    _bufferFbo[0]->attachColorTexture(*_bufferTex[0]);
    _bufferFbo[0]->attachDepthBuffer(width(), height());
    if (!_bufferFbo[0]->checkStatus()) {
        qDebug("Error: bufferFbo");
    }
    _bufferFbo[0]->release();

    _bufferTex[1] = new MSLib::GLTexture2D(GL_RGBA32F, width(), height(), 0, GL_RGBA, GL_FLOAT, 0);
    _bufferFbo[1] = new MSLib::GLFramebufferObject();
    _bufferFbo[1]->bind();
    _bufferFbo[1]->attachColorTexture(*_bufferTex[1]);
    _bufferFbo[1]->attachDepthBuffer(width(), height());
    if (!_bufferFbo[1]->checkStatus()) {
        qDebug("Error: bufferFbo");
    }
    _bufferFbo[1]->release();

    _copyColorShader = new GLShader();
    _copyColorShader->loadVertexShader((_workingPath + "/shaders/Default.vert").toAscii().constData());
    _copyColorShader->loadFragmentShader((_workingPath + "/shaders/CopyColor.frag").toAscii().constData());
    _copyColorShader->link();
    _copyColorShader->printVertexShaderInfoLog();
    _copyColorShader->printFragmentShaderInfoLog();
    _copyColorShader->addUniformSampler("colorBuffer", 0);
    _copyColorShader->addUniform2f("invImageDim", 1.0f / (float)width(), 1.0f / (float)height());

    _testMode = -1;
    _hand = Vector2f(100.0f, 100.0f);
    _state = 0;

    connect(&_ps, SIGNAL(parameterChanged(const String &)), this, SLOT(parameterChanged(const String &)));
    qDebug("initialize... done");
}

void VolumeRenderWindow::render() {
    glClearColor(m_transferFunction.backgroundColor.redF(),
                 m_transferFunction.backgroundColor.greenF(),
                 m_transferFunction.backgroundColor.blueF(), 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // determine box color
    Vector4 bgcolor;
    glGetDoublev(GL_COLOR_CLEAR_VALUE, (double *)bgcolor);
    Vector3 boxcolor(bgcolor.x(), bgcolor.y(), bgcolor.z());
    if (boxcolor.length() > 0.7) {
        boxcolor = Vector3(0.0, 0.0, 0.0);
    } else {
        boxcolor = Vector3(1.0, 1.0, 1.0);
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    pushMatrices();
    m_camera.updateCamera();    // projection, modelview, scale, translation

    if (m_drawBoundingBox) {
        glCullFace(GL_FRONT);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glColor3d(unpack3(boxcolor));
        if (slicerEnabled()) {
            box.drawMultiSlicedBox(m_slicers);
        } else {
            drawBox();
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glCullFace(GL_BACK);
    }

    // draw the back side masking slicing plane
    if (slicerEnabled()) {
        glCullFace(GL_FRONT);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        box.drawMultiSlicedBox(m_slicers, true);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glCullFace(GL_BACK);
    }

    _renderer->setSlicerEnabled(slicerEnabled());

    if (_ps["segmentEnabled"].toBool()) {
        //_segmentedRayCastingRenderer.preRender();
        //_segmentedRayCastingRenderer.preRender(0);
    } else {
        _renderer->renderBegin();
    }

    if (_ps["segmentEnabled"].toBool()) {
        int target = 0;     // target FBO index; (target ^ 1) is source

        _bufferFbo[target]->bind();
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        _bufferFbo[target]->release();
        _segmentedRenderer->renderBegin();

        for (int i = 0; i < _model->blockCount(); i++) {
            if (_testMode > 0 && i != _testMode - 1) {
                continue;
            }

            target ^= 1;                // switch target and source

            _bufferFbo[target]->bind();
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            _segmentedRenderer->renderBegin(i, *_bufferTex[target ^ 1]);

            if (slicerEnabled()) {
                box.drawMultiSlicedBox(m_slicers);
            } else {
                box.drawBox();
            }

            // if the camera is inside the volume, draw the volume from the clipping plane
            pushMatrices();
            double n = m_camera.nearClip();
            m_camera.setNearclip(n / 2.0);
            m_camera.updateCamera();
            box.drawSlice(m_camera, n);
            m_camera.setNearclip(n);
            popMatrices();

            _bufferFbo[target]->release();
        }

        _segmentedRenderer->renderEnd();
        _bufferTex[target]->bind(0);
        _copyColorShader->setUniform2f("invImageDim", 1.0f / (float)width(), 1.0f / (float)height());
        _copyColorShader->use();

        if (slicerEnabled())
            box.drawMultiSlicedBox(m_slicers);
        else
            box.drawBox();

        // if the camera is inside the volume, draw the volume from the clipping plane
        pushMatrices();
        double n = m_camera.nearClip();
        m_camera.setNearclip(n / 2.0);
        m_camera.updateCamera();
        box.drawSlice(m_camera, n);
        m_camera.setNearclip(n);
        popMatrices();

        _copyColorShader->useFixed();
        _bufferTex[target]->release();
    } else { // regular, not segmented
        if (slicerEnabled()) {
            box.drawMultiSlicedBox(m_slicers);
        } else {
            box.drawBox();
        }

        pushMatrices();
        double n = m_camera.nearClip();
        m_camera.setNearclip(n / 2.0);
        m_camera.updateCamera();
        box.drawSlice(m_camera, n);
        m_camera.setNearclip(n);
        popMatrices();
    }

    if (_ps["segmentEnabled"].toBool()) {
        //_segmentedRayCastingRenderer.postRender();
    } else {
        _renderer->renderEnd();
    }

    // draw the front side masking slicing plane
    if (slicerEnabled()) {
        glCullFace(GL_BACK);
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        box.drawMultiSlicedBox(m_slicers, true);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }

    // draw front side bounding box
    if (m_drawBoundingBox) {
        glDisable(GL_DEPTH_TEST);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glColor3d(unpack3(boxcolor));
        if (slicerEnabled())
            box.drawMultiSlicedBox(m_slicers);
        else
            drawBox();
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);



    ////
    // draw palm
    ////
    if (false)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, (double)width(), 0.0, (double)height(), -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        if (_state == 1)
            glColor3d(1.0, 0.0, 0.0);
        else
            glColor3d(unpack3(boxcolor));
        glBegin(GL_QUADS);
        //glTexCoord2f(0.0f, 0.0f);
        glVertex2f(_hand.x - 1.0f, _hand.y - 1.0f);
        //glTexCoord2f(1.0f, 0.0f);
        glVertex2f(_hand.x + 1.0f, _hand.y - 1.0f);
        //glTexCoord2f(1.0f, 1.0f);
        glVertex2f(_hand.x + 1.0f, _hand.y + 1.0f);
        //glTexCoord2f(0.0f, 1.0f);
        glVertex2f(_hand.x - 1.0f, _hand.y + 1.0f);
        glEnd();
    }

    drawAxis();                 //!! shift
    popMatrices();
    glFlush();
}

void VolumeRenderWindow::updateGL() {
    QRenderWindow::updateGL();
}

void VolumeRenderWindow::resizeBuffer(MSLib::GLFramebufferObject *&fbo, MSLib::GLTexture2D *&bufferTex, int width, int height) {
    fbo->bind();
    fbo->detachColorTexture();
    fbo->detachDepthBuffer();
    delete bufferTex;
    bufferTex = new MSLib::GLTexture2D(GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
    fbo->attachColorTexture(*bufferTex);
    fbo->attachDepthBuffer(width, height);
    fbo->release();
}

void VolumeRenderWindow::resizeGL(int width, int height) {
    qDebug("resizeGL()");
    QRenderWindow::resizeGL(width, height);

    resizeBuffer(_bufferFbo[0], _bufferTex[0], width, height);
    resizeBuffer(_bufferFbo[1], _bufferTex[1], width, height);
}

void VolumeRenderWindow::mousePressEvent(QMouseEvent *e) {
    QRenderWindow::mousePressEvent(e);
}

void VolumeRenderWindow::mouseMoveEvent(QMouseEvent *e) {
    QRenderWindow::mouseMoveEvent(e);
    if (m_mousetarget == MTSliceMove) {
        _ps["slicerPos"].setValue((float)m_slicers[m_slicerIdx].getDist());
    }
}

void VolumeRenderWindow::mouseReleaseEvent(QMouseEvent *e) {
    QRenderWindow::mouseReleaseEvent(e);
}

void VolumeRenderWindow::wheelEvent(QWheelEvent *e) {
    QRenderWindow::wheelEvent(e);
}

void VolumeRenderWindow::keyPressEvent(QKeyEvent *e) {
    if (e->key() == Qt::Key_F5) {
        _renderer->reloadShader();
        updateGL();
    } else if (e->key() == Qt::Key_F8) {
        parentWidget()->resize(768 + parentWidget()->width() - width(),
                               768 + parentWidget()->height() - height());
    } else if (e->key() == Qt::Key_F9) {
        QString dateTime = QDateTime::currentDateTime().toString("yyyy-MM-dd_hh.mm.ss");
        for (int i = 1; i <= _ps["totalSteps"].toInt(); i++) {
            _ps["timestep"].setValue(i);
            QImage img = grabFrameBuffer();
            img.save(QString("screenshot_%1_%2.png").arg(_ps["timestep"].toInt(), 2, 10, QChar('0')).arg(dateTime));
        }
    } else if (e->key() == Qt::Key_F10) {
        QString dateTime = QDateTime::currentDateTime().toString("yyyy-MM-dd_hh.mm.ss");
        QImage img = grabFrameBuffer();
        img.save(QString("screenshot_%1.png").arg(dateTime));
    }
    // for testing
    else if (e->key() == Qt::Key_0) { _testMode = 0; updateGL(); }
    else if (e->key() == Qt::Key_1) { _testMode = 1; updateGL(); }
    else if (e->key() == Qt::Key_2) { _testMode = 2; updateGL(); }
    else if (e->key() == Qt::Key_3) { _testMode = 3; updateGL(); }
    else if (e->key() == Qt::Key_4) { _testMode = 4; updateGL(); }
    else if (e->key() == Qt::Key_5) { _testMode = 5; updateGL(); }
    else if (e->key() == Qt::Key_6) { _testMode = 6; updateGL(); }
    else if (e->key() == Qt::Key_7) { _testMode = 7; updateGL(); }
    else if (e->key() == Qt::Key_8) { _testMode = 8; updateGL(); }
    else if (e->key() == Qt::Key_9) { _testMode = 9; updateGL(); }
    else if (e->key() == Qt::Key_C) {
        qDebug("Save camera settings");
        QFile file(_workingPath + "/camera.bin");
        file.open(QIODevice::WriteOnly);
        m_camera.saveSettings(file);
        file.close();
    }
    else if (e->key() == Qt::Key_V) {
        qDebug("Load camera settings");
        QFile file(_workingPath + "/camera.bin");
        file.open(QIODevice::ReadOnly);
        m_camera.loadSettings(file);
        file.close();
        updateGL();
    }
    QRenderWindow::keyPressEvent(e);
}

void VolumeRenderWindow::keyReleaseEvent(QKeyEvent *e) {
    QRenderWindow::keyReleaseEvent(e);
}

void VolumeRenderWindow::focusInEvent(QFocusEvent *e) {
    qDebug("%s: focusInEvent()", _model->name().c_str());

    // set TFEditor
    qDebug("disconnecting TFEditor...");
    _mainUI->getTFEditor()->getColorMap()->disconnect();
    _mainUI->getTFEditor()->getTFPanel()->disconnect();

    _mainUI->getTFEditor()->getTFPanel()->loadSettings(m_transferFunction);
    _mainUI->getTFEditor()->updateHistogram(m_histogram);
    _mainUI->getTFEditor()->getQHistogram()->updateHistogram();

    qDebug("connecting TFEditor...");
    connect(_mainUI->getTFEditor()->getColorMap(), SIGNAL(bgColorChanged(const QColor &)), this, SLOT(bgColorChanged(const QColor &)));
    connect(_mainUI->getTFEditor()->getTFPanel(), SIGNAL(tfChanged(float *, bool)), this, SLOT(tfChanged(float *, bool)));
    connect(_mainUI->getTFEditor()->getTFPanel(), SIGNAL(tfMappingChanged(float *, float *, bool)), this, SLOT(tfMappingChanged(float *, float *, bool)));

    // set ParameterSet
    qDebug("disconnecting PS...");
    _mainUI->unsetParameterSet();
    _mainUI->disconnect();
    _mainUI->getRenderEffectPanel()->setSlicers(m_slicers.size(), m_slicerIdx);

    qDebug("connecting PS...");
    connect(_mainUI, SIGNAL(actionTriggered(const QString &)), this, SLOT(actionTriggered(const QString &)));
    _mainUI->setParameterSet(&_ps);

    QRenderWindow::focusInEvent(e);

    qDebug("done.");
}

// gnavvy
void VolumeRenderWindow::reloadData() {
    qDebug("reloadData()");

    float *rawData = _model->data(_ps["timestep"].toInt() - 1, _ps["compIdx"].toInt());

    _renderer->updateData();


    _mainUI->getTFEditor()->getHistogram()->clear();
    Vector3i dim = _model->dim();
    size_t dataSize = dim.x * dim.y * dim.z;
    for (size_t i = 0; i < dataSize; i++) {
        _mainUI->getTFEditor()->incrementHistogram(rawData[i]);
    }
    _mainUI->getTFEditor()->getQHistogram()->updateHistogram();
    *m_histogram = *(_mainUI->getTFEditor()->getHistogram());
}

void VolumeRenderWindow::activated() {
    qDebug("activated");
}

void VolumeRenderWindow::bgColorChanged(const QColor &color) {
    if (!isActiveSubWindow()) return;
    m_transferFunction.backgroundColor = color;
    updateGL();
}

void VolumeRenderWindow::tfChanged(float *colorMap, bool immediate) {
    Q_UNUSED(colorMap)

    qDebug("tfChanged()");
    makeCurrent();
    _renderer->updateTF();
    _mainUI->getTFEditor()->getTFPanel()->saveSettings(m_transferFunction);

    if (immediate) {
        updateGL();
    }
}

void VolumeRenderWindow::tfMappingChanged(float *colorMap1, float *colorMap2, bool immediate) {
    Q_UNUSED(colorMap1)
    Q_UNUSED(colorMap2)

    if (!isActiveSubWindow()) return
    qDebug("tfMappingChanged()");
    if (immediate) updateGL();
}

void VolumeRenderWindow::sliceVectorChanged(Vector3 val) {
    if (!isActiveSubWindow()) return;
    if (m_slicerIdx >= 0) {
        m_slicers[m_slicerIdx].setVec(val);
        updateGL();
    }
}

void VolumeRenderWindow::timestepChanged(int val) {
    if (!isActiveSubWindow()) return;
    if (_model != 0 || _dataset != 0) {
        reloadData();
    }
    updateGL();
}

void VolumeRenderWindow::parameterChanged(const String &name) {
    qDebug("parameterChanged(%s)", name.c_str());

    if (name == "compIdx") {
        qDebug("compIdx = %d", _ps["compIdx"].toInt());
        if (_model != 0 || _dataset != 0) {
            if (_ps["compIdx"].toInt() >= 0 && _ps["compIdx"].toInt() < _ps["compCount"].toInt()) {
                reloadData();
            }
        }
    } else if (name == "preIntEnabled") {
        _renderer->updateTF();
    } else if (name == "segmentEnabled") {
        makeCurrent();
        delete _renderer;
        _renderer = nullptr;
        if (_ps["segmentEnabled"].toBool()) {
            _segmentedRenderer = new SegmentedVolumeRenderer(*this, *_mainUI->getTFEditor(), _ps, *_model, box, _workingPath.toStdString());
            _renderer = _segmentedRenderer;
        } else {
            _renderer = new VolumeRenderer(*this, *_mainUI->getTFEditor(), _ps, *_model, box, _workingPath.toStdString());
        }
    } else if (name == "sampleStep") {
        sampleSpacing = _ps["sampleStep"].toFloat();
        _renderer->updateTF();
    } else if (name == "mainAxis") {
        if (_ps["mainAxis"].toBool()) {
            m_axisOptions |= 0x2;
        } else {
            m_axisOptions &= (~0x2);
        }
    } else if (name == "sideAxis") {
        if (_ps["sideAxis"].toBool()) {
            m_axisOptions |= 0x1;
        } else {
            m_axisOptions &= (~0x1);
        }
    } else if (name == "boundingBox") {
        m_drawBoundingBox = _ps["boundingBox"].toBool();
    } else if (name == "lightEnabled") {
        enableLight = _ps["lightEnabled"].toBool();
    } else if (name == "ambient") {
        lightParam.Kamb = _ps["ambient"].toFloat();
    } else if (name == "diffuse") {
        lightParam.Kdif = _ps["diffuse"].toFloat();
    } else if (name == "specular") {
        lightParam.Kspe = _ps["specular"].toFloat();
    } else if (name == "shininess") {
        lightParam.Kshi = _ps["shininess"].toFloat();
    } else if (name == "mouseMode") {
        switch (_ps["mouseMode"].toInt()) {
            case 0: m_mousetarget = MTCamera; break;
            case 1: m_mousetarget = MTSliceTrack; break;
            case 2: m_mousetarget = MTSliceMove; break;
        }
    } else if (name == "slicerIdx") {
        m_slicerIdx = _ps["slicerIdx"].toInt();
        if (m_slicerIdx >= 0 && m_slicerIdx < m_slicers.size()) {
            _ps["slicerMask"].setValue(m_slicers[m_slicerIdx].isMasking());
            _ps["slicerPos"].setValue((float)m_slicers[m_slicerIdx].getDist());
        }
        qDebug("m_slicerIdx=%d", m_slicerIdx);
    } else if (name == "slicerMask") {
        if (m_slicerIdx >= 0) {
            m_slicers[m_slicerIdx].setMasking(_ps["slicerMask"].toBool());
        }
    } else if (name == "slicerPos") {
        if (m_slicerIdx >= 0) {
            m_slicers[m_slicerIdx].setDist((double)_ps["slicerPos"].toFloat());
        }
    } else if (name == "timestep") {
        reloadData();
    }

    updateGL();
}

void VolumeRenderWindow::actionTriggered(const QString &name) {
    if (!isActiveSubWindow()) return;
    qDebug("actionTriggered(%s)", name.toAscii().constData());

    if (name == "resetCamera") {
        m_camera.loadOptions(_defaultCamera);
    }
    else if (name == "addSlicer") {
        Slicer slicer;
        slicer.resize(m_width, m_height);
        m_slicers.append(slicer);
    } else if (name == "delSlicer") {
        if (m_slicerIdx >= 0) {
            m_slicers.removeAt(m_slicerIdx);
            if (m_slicerIdx == m_slicers.size()) {
                m_slicerIdx--;
            }
            if (m_slicerIdx >= 0 && m_slicerIdx < m_slicers.size()) {
                _ps["slicerMask"].setValue(m_slicers[m_slicerIdx].isMasking());
                _ps["slicerPos"].setValue((float)m_slicers[m_slicerIdx].getDist());
            }
        }
    } else if (name == "setSlicerXp") {
        sliceVectorChanged(Vector3(1.0, 0.0, 0.0));
    } else if (name == "setSlicerYp") {
        sliceVectorChanged(Vector3(0.0, 1.0, 0.0));
    } else if (name == "setSlicerZp") {
        sliceVectorChanged(Vector3(0.0, 0.0, 1.0));
    } else if (name == "setSlicerXn") {
        sliceVectorChanged(Vector3(-1.0, 0.0, 0.0));
    } else if (name == "setSlicerYn") {
        sliceVectorChanged(Vector3(0.0, -1.0, 0.0));
    } else if (name == "setSlicerZn") {
        sliceVectorChanged(Vector3(0.0, 0.0, -1.0));
    } else {
        qDebug("Error: VolumeRenderWindow::actionTriggered(): unknown action");
    }

    updateGL();
}

void VolumeRenderWindow::cameraTrack(int x, int y, int state) {
    _hand = Vector2f((float)x / 640.0 * (float)width(), (float)y / 480.0 * (float)height());

    if (state != _state && state == 1) {
        m_camera.start(Vector2((double)_hand.x, (double)_hand.y));
    }
    _state = state;

    if (_state == 1) {
        m_camera.track(Vector2((double)_hand.x, (double)_hand.y));
    }

    updateGL();
}
