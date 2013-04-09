#include "SegmentedRayCastingRenderer.h"

#define nullptr 0

SegmentedRayCastingRenderer::SegmentedRayCastingRenderer()
    : _shader(nullptr),
      _boundBlock(0)
{
}

SegmentedRayCastingRenderer::~SegmentedRayCastingRenderer()
{
    if (_shader != nullptr)
        delete _shader;
}

void SegmentedRayCastingRenderer::init()
{
    _shader = new GLShader();
    _shader->loadVertexShader(_vertShaderFileName.c_str());
    _shader->loadFragmentShader(_fragShaderFileName.c_str());
    _shader->link();
    _shader->printVertexShaderInfoLog();
    _shader->printFragmentShaderInfoLog();

    _shader->addUniform2f("imageScale", 1.0f / (float)_renderWindow->width(), 1.0f / (float)_renderWindow->height());
    _shader->addUniform3f("viewVec", 0.0f, 0.0f, 1.0f);
    //_shader->addUniform3f("scaledDim", _model->scaledDim());        ////
    _shader->addUniform3f("boxLo", 0.0f, 0.0f, 0.0f);
    _shader->addUniform3f("boxHi", 0.0f, 0.0f, 0.0f);
    //_shader->addUniform3f("scaledPaddedLo", 0.0f, 0.0f, 0.0f);
    //_shader->addUniform3f("scaledPaddedDim", 0.0f, 0.0f, 0.0f);
    _shader->addUniform3f("offset", 0.0f, 0.0f, 0.0f);
    _shader->addUniform3f("scale", 0.0f, 0.0f, 0.0f);
    //_shader->addUniform1f("sampleSpacing", &(*_ps)["sampleStep"]);
    _shader->addUniform1f("sampleInterval", (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length());
    _shader->addUniform4f("lightParam", (*_ps)["ambient"].toFloat(), (*_ps)["diffuse"].toFloat(), (*_ps)["specular"].toFloat(), (*_ps)["shininess"].toFloat());
    _shader->addUniform1b("lightEnabled", &(*_ps)["lightEnabled"]);
    _shader->addUniform1f("projection",  &(*_ps)["projection"]);
    ////_shader->addUniform1f("mapping", 0.0f);
    _shader->addUniformSampler("data", 0);
    _shader->addUniformSampler("tf", 1);
    ////_shader->addUniformSampler("tfMapping", 2);
    _shader->addUniformSampler("colorBuffer", 2);
}

void SegmentedRayCastingRenderer::init(QRenderWindow &renderWindow,
                                       QTFEditor &tfEditor,
                                       ParameterSet &ps,
                                       VolumeModel &model,
                                       QList<MSLib::GLTexture3D *> &dataBlockTex,
                                       MSLib::GLTexture1D &tfTex,
                                       const String &workingPath)
{
    _renderWindow = &renderWindow;
    _tfEditor = &tfEditor;
    _ps = &ps;
    _model = &model;
    _dataBlockTex = &dataBlockTex;
    _tfTex = &tfTex;
    _workingPath = workingPath;
    _vertShaderFileName = _workingPath + "/shaders/SegmentedRayCasting.vert";
    _fragShaderFileName = _workingPath + "/shaders/SegmentedRayCasting.frag";

    init();
}

void SegmentedRayCastingRenderer::preRender()
{
    _tfTex->bind(1);

    _shader->setUniform2f("imageScale", 1.0f / (float)_renderWindow->width(), 1.0f / (float)_renderWindow->height());
    _shader->setUniform1f("sampleInterval", (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length());
    _shader->setUniform4f("lightParam", (*_ps)["ambient"].toFloat(), (*_ps)["diffuse"].toFloat(), (*_ps)["specular"].toFloat(), (*_ps)["shininess"].toFloat());
    Vector3 v = _renderWindow->getCamera().getCamPosition();
    Vector3f camPos((float)v.x(), (float)v.y(), (float)v.z());
    _shader->setUniform3f("viewVec", camPos);
    //_shader->setUniformSampler("data", 0);
    //_shader->setUniformSampler("tf", 1);

    _visOrder.clear();
    for (int i = 0; i < _model->blockCount(); i++)
    {
        RegularGridDataBlock &dataBlock = _model->volumeDataBlock(i, (*_ps)["timestep"].toInt() - 1, (*_ps)["compIdx"].toInt());
        float dist = (dataBlock.boxCenter() - camPos).length();
        _visOrder.append(std::pair<float, int>(dist, i));
    }
    std::sort(_visOrder.begin(), _visOrder.end());

    for (int i = 0; i < _visOrder.size(); i++)
        qDebug("%d: (%f, %d)", i, _visOrder[i].first, _visOrder[i].second);

}

void SegmentedRayCastingRenderer::preRender(int blockOrder, MSLib::GLTexture2D &colorBuffer)
{
    int blockIndex = _visOrder[blockOrder].second;
    (*_dataBlockTex)[blockIndex]->bind(0);
    _boundBlock = blockIndex;

    colorBuffer.bind(2);
    _boundColorBuffer = &colorBuffer;

    RegularGridDataBlock &dataBlock = _model->volumeDataBlock(blockIndex, (*_ps)["timestep"].toInt() - 1, (*_ps)["compIdx"].toInt());
    //_shader->setUniform3f("boxLo", dataBlock.scaledLo());
    //_shader->setUniform3f("boxHi", dataBlock.scaledHi());
    _shader->setUniform3f("boxLo", dataBlock.boxLo());
    _shader->setUniform3f("boxHi", dataBlock.boxHi());
    //_shader->setUniform3f("scaledPaddedLo", dataBlock.scaledPaddedLo());
    //_shader->setUniform3f("scaledPaddedDim", dataBlock.scaledPaddedDim());

    // texture -> object space: p_obj = p_tex * scale + offset
    // object -> texture space: p_tex = (p_obj - offset) / scale
    //                                = p_obj * scale' + offset'
    //                       => scale' = 1/scale, offset' = -offset/scale
    //_shader->setUniform3f("scale", Vector3f(1.0f, 1.0f, 1.0f) / dataBlock.scaledPaddedDim());
    //_shader->setUniform3f("offset", -dataBlock.scaledPaddedLo() / dataBlock.scaledPaddedDim());
    _shader->setUniform3f("scale", Vector3f(1.0f, 1.0f, 1.0f) / dataBlock.scaledDim());
    _shader->setUniform3f("offset", -dataBlock.scaledLo() / dataBlock.scaledDim());

    _shader->use();
}

void SegmentedRayCastingRenderer::postRender()
{
    _shader->useFixed();

    (*_dataBlockTex)[_boundBlock]->release();
    _tfTex->release();

    _boundColorBuffer->release();
}

void SegmentedRayCastingRenderer::reloadShader()
{
    _shader->reload();
}

SegmentedPreIntegrationRenderer::SegmentedPreIntegrationRenderer()
    : SegmentedRayCastingRenderer(),
      _preIntegrator(nullptr),
      _internalPreIntegratorUsed(false)
{
}

SegmentedPreIntegrationRenderer::~SegmentedPreIntegrationRenderer()
{
    if (_internalPreIntegratorUsed && _preIntegrator != nullptr) delete _preIntegrator;
}

void SegmentedPreIntegrationRenderer::init(QRenderWindow &renderWindow,
                                           QTFEditor &tfEditor,
                                           ParameterSet &ps,
                                           VolumeModel &model,
                                           QList<MSLib::GLTexture3D *> &dataBlockTex,
                                           MSLib::GLTexture1D &tfTex,
                                           const String &workingPath,
                                           PreIntegratorGL *preIntegrator)
{
    _renderWindow = &renderWindow;
    _tfEditor = &tfEditor;
    _ps = &ps;
    _model = &model;
    _dataBlockTex = &dataBlockTex;
    _tfTex = &tfTex;
    _workingPath = workingPath;
    _vertShaderFileName = _workingPath + "/shaders/SegmentedRayCasting.vert";
    _fragShaderFileName = _workingPath + "/shaders/SegmentedRayCastingPreInt.frag";

    SegmentedRayCastingRenderer::init();

    if (preIntegrator != nullptr)
    {
        _preIntegrator = preIntegrator;
        _internalPreIntegratorUsed = false;
    }
    else
    {
        float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
        _preIntegrator = new PreIntegratorGL(_tfEditor->getTFColorMapResolution(), sampleInterval, 0.01f);
        _preIntegrator->update(*_tfTex);
        _internalPreIntegratorUsed = true;
    }

    _shader->addUniformSampler("preInt", 3);
    _shader->addUniformSampler("preIntFront", 4);
    _shader->addUniformSampler("preIntBack", 5);
}

void SegmentedPreIntegrationRenderer::preRender()
{
    _preIntegrator->colorTable()->bind(3);
    _preIntegrator->frontTable()->bind(4);
    _preIntegrator->backTable()->bind(5);

    //_shader->setUniformSampler("preInt", *_preIntegrator->colorTable());
    //_shader->setUniformSampler("preIntFront", *_preIntegrator->frontTable());
    //_shader->setUniformSampler("preIntBack", *_preIntegrator->backTable());

    SegmentedRayCastingRenderer::preRender();
}

void SegmentedPreIntegrationRenderer::preRender(int blockOrder, MSLib::GLTexture2D &colorBuffer)
{
    SegmentedRayCastingRenderer::preRender(blockOrder, colorBuffer);
}

void SegmentedPreIntegrationRenderer::postRender()
{
    SegmentedRayCastingRenderer::postRender();

    _preIntegrator->colorTable()->release();
    _preIntegrator->frontTable()->release();
    _preIntegrator->backTable()->release();
}
