#include "PreIntegrationRenderer.h"

PreIntegrationRenderer::PreIntegrationRenderer()
    : RayCastingShader(),
      _preIntegrator(nullptr),
      _internalPreIntegratorUsed(false)
{
}

PreIntegrationRenderer::~PreIntegrationRenderer()
{
    if (_internalPreIntegratorUsed && _preIntegrator != nullptr) delete _preIntegrator;
}

void PreIntegrationRenderer::init(QRenderWindow &renderWindow,
                                  QTFEditor &tfEditor,
                                  ParameterSet &ps,
                                  VolumeModel &model,
                                  //float scaleDimX, float scaleDimY, float scaleDimZ, bool projection,
                                  MSLib::GLTexture3D &dataTex,
                                  MSLib::GLTexture1D &tfTex,
                                  const String &workingPath,
                                  PreIntegratorGL *preIntegrator)
{
    _renderWindow = &renderWindow;
    _tfEditor = &tfEditor;
    _ps = &ps;
    _model = &model;

    //_scaleDim[0] = scaleDimX;
    //_scaleDim[1] = scaleDimY;
    //_scaleDim[2] = scaleDimZ;
    //_projection = projection;
    _dataTex = &dataTex;
    _tfTex = &tfTex;

    _workingDir = workingPath;
    _vertShaderFileName = _workingDir + "/shaders/raycasting.vert";
    _fragShaderFileName = _workingDir + "/shaders/preInt.frag";

    RayCastingShader::init();

    //_preIntegrator = new PreIntegratorGL(_tfEditor->getTFColorMapResolution(), (*_ps)["sampleStep"].toFloat(), 0.01f);
    //_preIntegrator->update(*_tfTex);
    //_preIntTex = _glPreIntegrator->colorTable();
    //_preIntFrontTex = _glPreIntegrator->frontTable();
    //_preIntBackTex = _glPreIntegrator->backTable();

    if (preIntegrator != nullptr)
    {
        _preIntegrator = preIntegrator;
        _internalPreIntegratorUsed = false;
    }
    else
    {
        //Vector3f scaledDim(_scaleDim);
        float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
        _preIntegrator = new PreIntegratorGL(_tfEditor->getTFColorMapResolution(), sampleInterval, 0.01f);
        _preIntegrator->update(*_tfTex);
        _internalPreIntegratorUsed = true;
    }

    _shader->addUniformSampler("preInt", 3);
    _shader->addUniformSampler("preIntFront", 4);
    _shader->addUniformSampler("preIntBack", 5);
}

void PreIntegrationRenderer::preRender()
{
    _preIntegrator->colorTable()->bind(3);
    _preIntegrator->frontTable()->bind(4);
    _preIntegrator->backTable()->bind(5);

    _shader->setUniformSampler("preInt", *_preIntegrator->colorTable());
    _shader->setUniformSampler("preIntFront", *_preIntegrator->frontTable());
    _shader->setUniformSampler("preIntBack", *_preIntegrator->backTable());

    RayCastingShader::preRender();
}

void PreIntegrationRenderer::postRender()
{
    RayCastingShader::postRender();

    _preIntegrator->colorTable()->release();
    _preIntegrator->frontTable()->release();
    _preIntegrator->backTable()->release();
}

void PreIntegrationRenderer::setStepSize(float stepSize)
{
    _preIntegrator->setStepSize(stepSize);
}

void PreIntegrationRenderer::update(MSLib::GLTexture1D &tfTex)
{
    _preIntegrator->update(tfTex);
}
