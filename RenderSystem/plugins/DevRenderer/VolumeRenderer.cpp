#include "VolumeRenderer.h"

VolumeRenderer::VolumeRenderer(QRenderWindow &renderWindow,
                               QTFEditor &tfEditor,
                               ParameterSet &ps,
                               VolumeModel &model,
                               Box &box,
                               const String &workingPath)
    : _renderWindow(&renderWindow),
      _tfEditor(&tfEditor),
      _ps(&ps),
      _model(&model),
      _box(&box),
      _slicerEnabled(false) {

    qDebug("Init VolumeRenderer...");
    initDataTexture();
    initTFTexture();
    initPreIntegrator();
    initShaders(workingPath);
}

VolumeRenderer::~VolumeRenderer() {
    if (_rayCastingShader      != nullptr) delete _rayCastingShader;
    if (_sliceRayCastingShader != nullptr) delete _sliceRayCastingShader;
    if (_preIntegrationShader  != nullptr) delete _preIntegrationShader;
    if (_preIntegrator         != nullptr) delete _preIntegrator;
    if (_dataTex               != nullptr) delete _dataTex;
    if (_tfTex                 != nullptr) delete _tfTex;
}

VolumeRenderer::VolumeRenderer(QRenderWindow &renderWindow,
                               QTFEditor &tfEditor,
                               ParameterSet &ps,
                               VolumeModel &model)
    : _renderWindow(&renderWindow),
      _tfEditor(&tfEditor),
      _ps(&ps),
      _model(&model),
      _box(nullptr),
      _rayCastingShader(nullptr),
      _sliceRayCastingShader(nullptr),
      _preIntegrationShader(nullptr),
      _dataTex(nullptr),
      _tfTex(nullptr),
      _preIntegrator(nullptr),
      _slicerEnabled(false) {
}

void VolumeRenderer::initDataTexture() {
    _dataTex = new MSLib::GLTexture3D(GL_R32F,
                                      _model->dim().x,
                                      _model->dim().y,
                                      _model->dim().z,
                                      0,
                                      GL_RED,
                                      GL_FLOAT,
                                      _model->data(timeStep(), varIndex()));
}

void VolumeRenderer::initTFTexture() {
    _tfTex = new MSLib::GLTexture1D(GL_RGBA32F,
                                    _tfEditor->getTFColorMapResolution(),
                                    0,
                                    GL_RGBA,
                                    GL_FLOAT,
                                    _tfEditor->getTFColorMap());
}

void VolumeRenderer::initPreIntegrator() {
    float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
    _preIntegrator = new PreIntegratorGL(_tfEditor->getTFColorMapResolution(), sampleInterval, 0.01f);
    _preIntegrator->update(*_tfTex);
}

void VolumeRenderer::initShaders(const String &workingPath) {
    _rayCastingShader = new RayCastingShader();
    _rayCastingShader->init(*_renderWindow, *_tfEditor, *_ps, *_model, *_dataTex, *_tfTex, workingPath);

    _sliceRayCastingShader = new SliceRayCastingShader();
    _sliceRayCastingShader->init(*_renderWindow, *_tfEditor, *_ps, *_model, *_box, *_dataTex, *_tfTex, workingPath);

    _preIntegrationShader = new PreIntegrationRenderer();
    _preIntegrationShader->init(*_renderWindow, *_tfEditor, *_ps, *_model, *_dataTex, *_tfTex, workingPath, _preIntegrator);
}

void VolumeRenderer::updateData() {
    float *data = _model->data(timeStep(), varIndex());
    _dataTex->load(data);
}

void VolumeRenderer::updateTF() {
    _tfTex->load(_tfEditor->getTFColorMap());
    if (preIntegrationEnabled()) {
        float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
        _preIntegrator->setStepSize(sampleInterval);
        _preIntegrator->update(*_tfTex);
    }
}

void VolumeRenderer::renderBegin() {
    if (preIntegrationEnabled()) {
        _preIntegrationShader->preRender();
    } else {
        if (_slicerEnabled) {
            _sliceRayCastingShader->preRender();
        } else {
            _rayCastingShader->preRender();
        }
    }
}

void VolumeRenderer::renderEnd() {
    if (preIntegrationEnabled()) {
        _preIntegrationShader->postRender();
    } else {
        if (_slicerEnabled) {
            _sliceRayCastingShader->postRender();
        } else {
            _rayCastingShader->postRender();
        }
    }
}

void VolumeRenderer::reloadShader() {
    _rayCastingShader->reloadShader();
    _sliceRayCastingShader->reloadShader();
    _preIntegrationShader->reloadShader();
}
