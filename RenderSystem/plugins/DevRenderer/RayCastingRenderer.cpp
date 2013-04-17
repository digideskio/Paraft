#include "RayCastingRenderer.h"

#define nullptr 0

RayCastingRenderer::RayCastingRenderer(QRenderWindow &renderWindow,
                                       QTFEditor &tfEditor,
                                       ParameterSet &ps,
                                       VolumeModel &model,
                                       const String &workingPath)
    : _renderWindow(&renderWindow),
      _tfEditor(&tfEditor),
      _ps(&ps),
      _model(&model)
{
    _vertShaderFileName = workingPath + "/shaders/raycasting.vert";
    _fragShaderFileName = workingPath + "/shaders/regularRaycasting.frag";

    init();
}

RayCastingRenderer::RayCastingRenderer(QRenderWindow &renderWindow,
                                       QTFEditor &tfEditor,
                                       ParameterSet &ps,
                                       VolumeModel &model)
    : _renderWindow(&renderWindow),
      _tfEditor(&tfEditor),
      _ps(&ps),
      _model(&model)
{
}

RayCastingRenderer::~RayCastingRenderer()
{
    if (_dataTex != nullptr)
        delete _dataTex;
    if (_tfTex != nullptr)
        delete _tfTex;
    if (_shader != nullptr)
        delete _shader;
}

void RayCastingRenderer::init()
{
    qDebug("Init RayCastingRenderer...");

    initShader();
    initTextures();
}

void RayCastingRenderer::initShader()
{
    _shader = new GLShader();
    _shader->loadVertexShader(_vertShaderFileName.c_str());
    _shader->loadFragmentShader(_fragShaderFileName.c_str());
    _shader->link();
    _shader->printVertexShaderInfoLog();
    _shader->printFragmentShaderInfoLog();

    _shader->addUniform3f("scaleDim", _model->scaledDim());
    float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
    _shader->addUniform1f("sampleSpacing", sampleInterval);
    _shader->addUniform4f("lightParam", 1.0f, 1.0f, 1.0f, 1.0f);
    _shader->addUniform1b("enableLight", &(*_ps)["lightEnabled"]);
    _shader->addUniform1f("projection",  &(*_ps)["projection"]);
    _shader->addUniform1f("mapping", 0.0f);
    _shader->addUniform3f("viewVec", 0.0f, 0.0f, 1.0f);
    _shader->addUniformSampler("data", 0);
    _shader->addUniformSampler("tf", 1);
    _shader->addUniformSampler("tfMapping", 2);
}

void RayCastingRenderer::initTextures()
{
    _dataTex = new MSLib::GLTexture3D(//GL_LUMINANCE32F_ARB,
                                      GL_R32F,
                                      _model->dim().x,
                                      _model->dim().y,
                                      _model->dim().z,
                                      0,
                                      //GL_LUMINANCE,
                                      GL_RED,
                                      GL_FLOAT,
                                      _model->data(timeStep(), varIndex()));

    _tfTex = new MSLib::GLTexture1D(GL_RGBA32F,
                                    _tfEditor->getTFColorMapResolution(),
                                    0,
                                    GL_RGBA,
                                    GL_FLOAT,
                                    _tfEditor->getTFColorMap());

    ////_tfMapTex
}

void RayCastingRenderer::renderBegin()
{
    _dataTex->bind(0);
    _tfTex->bind(1);

    //Vector3f scaledDim(_scaleDim);
    float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
    _shader->setUniform1f("sampleSpacing", sampleInterval);

    _shader->setUniform4f("lightParam", (*_ps)["ambient"].toFloat(), (*_ps)["diffuse"].toFloat(), (*_ps)["specular"].toFloat(), (*_ps)["shininess"].toFloat());
    //_shader->setUniform1f("projection", _projection);   ////
    _shader->setUniform1f("mapping", _tfEditor->getTFPanel()->getIsTurnOnMapping() ? 1.0f : 0.0f);
    Vector3 v = _renderWindow->getCamera().getCamPosition();
    _shader->setUniform3f("viewVec", (float)v.x(), (float)v.y(), (float)v.z());
    _shader->setUniformSampler("data", *_dataTex);
    _shader->setUniformSampler("tf", *_tfTex);
    _shader->setUniformSampler("tfMapping", 2);

    _shader->use();
}

void RayCastingRenderer::renderEnd()
{
    _shader->useFixed();

    _dataTex->release();
    _tfTex->release();
}

void RayCastingRenderer::reloadShader()
{
    _shader->reload();
}

void RayCastingRenderer::updateTF()
{
    _tfTex->load(_tfEditor->getTFColorMap());
}

void RayCastingRenderer::updateData()
{
    float *data = _model->data(timeStep(), varIndex());
    _dataTex->load(data);
}

SliceRayCastingRenderer::SliceRayCastingRenderer(QRenderWindow &renderWindow,
                                                 QTFEditor &tfEditor,
                                                 ParameterSet &ps,
                                                 VolumeModel &model,
                                                 Box &box,
                                                 const String &workingPath)
    : RayCastingRenderer(renderWindow, tfEditor, ps, model),
      _box(&box)
{
    _vertShaderFileName = workingPath + "/shaders/raycasting.vert";
    _fragShaderFileName = workingPath + "/shaders/sliceRaycasting.frag";

    initShader();
    RayCastingRenderer::initTextures();
}

SliceRayCastingRenderer::~SliceRayCastingRenderer()
{
}

void SliceRayCastingRenderer::initShader()
{
    RayCastingRenderer::initShader();

    _shader->addUniform1i("sliceNum", 0);
    float v[10 * 3] = {0.0f};
    for (int i = 0; i < 10 * 3; i++) v[i] = 0.0f;   //// fill
    _shader->addUniform3fv("sliceVec", 10, v);
    _shader->addUniform3fv("slicePnt", 10, v);
}

void SliceRayCastingRenderer::renderBegin()
{
    QList<Slicer> &slicers = _renderWindow->getSlicers();
    _shader->setUniform1i("sliceNum", slicers.size());
    float *sliceVec = new float[10 * 3];
    float *slicePnt = new float[10 * 3];
    for (int i = 0; i < 10; i++)
    {
        if (i < slicers.size())
        {
            Vector3 v = slicers[i].getVec();
            sliceVec[i * 3 + 0] = (float)v.x();
            sliceVec[i * 3 + 1] = (float)v.y();
            sliceVec[i * 3 + 2] = (float)v.z();
            v = _box->getCenterPoint() + v * slicers[i].getDist();
            slicePnt[i * 3 + 0] = (float)v.x();
            slicePnt[i * 3 + 1] = (float)v.y();
            slicePnt[i * 3 + 2] = (float)v.z();
        }
        else
        {
            sliceVec[i * 3 + 0] = sliceVec[i * 3 + 1] = sliceVec[i * 3 + 2] = 0.0f;
            slicePnt[i * 3 + 0] = slicePnt[i * 3 + 1] = slicePnt[i * 3 + 2] = 0.0f;
        }
    }
    _shader->setUniform3fv("sliceVec", sliceVec);
    _shader->setUniform3fv("slicePnt", slicePnt);

    RayCastingRenderer::renderBegin();
}

void SliceRayCastingRenderer::renderEnd()
{
    RayCastingRenderer::renderEnd();
}

////////////////////////////////////////////////////////////////////////////////

RayCastingShader::RayCastingShader()
    : _shader(nullptr)
{
}

RayCastingShader::~RayCastingShader()
{
    if (_shader != nullptr) delete _shader;
}

void RayCastingShader::init()
{
    _shader = new GLShader();
    _shader->loadVertexShader(_vertShaderFileName.c_str());
    _shader->loadFragmentShader(_fragShaderFileName.c_str());
    _shader->link();
    _shader->printVertexShaderInfoLog();
    _shader->printFragmentShaderInfoLog();

    //_shader->addUniform3f("scaleDim", _scaleDim[0], _scaleDim[1], _scaleDim[2]);
    _shader->addUniform3f("scaleDim", _model->scaledDim());
    //_shader->addUniform1f("sampleSpacing", &(*_ps)["sampleStep"]);
    //Vector3f scaledDim(_scaleDim);
    float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
    _shader->addUniform1f("sampleSpacing", sampleInterval);
    _shader->addUniform4f("lightParam", 1.0f, 1.0f, 1.0f, 1.0f);
    _shader->addUniform1b("enableLight", &(*_ps)["lightEnabled"]);
    _shader->addUniform1f("projection",  &(*_ps)["projection"]);
    _shader->addUniform1f("mapping", 0.0f);
    _shader->addUniform3f("viewVec", 0.0f, 0.0f, 1.0f);
    _shader->addUniformSampler("data", 0);
    _shader->addUniformSampler("tf", 1);
    _shader->addUniformSampler("tfMapping", 2);
}

void RayCastingShader::init(QRenderWindow &renderWindow,
                            QTFEditor &tfEditor,
                            ParameterSet &ps,
                            VolumeModel &model,
                            MSLib::GLTexture3D &dataTex,
                            MSLib::GLTexture1D &tfTex,
                            const String &workingPath)
{
    _renderWindow = &renderWindow;
    _tfEditor = &tfEditor;
    _ps = &ps;
    _model = &model;
    _workingDir = workingPath;
    _vertShaderFileName = _workingDir + "/shaders/raycasting.vert";
    _fragShaderFileName = _workingDir + "/shaders/regularRaycasting.frag";

    //_scaleDim[0] = _model->scaledDim().x;
    //_scaleDim[1] = _model->scaledDim().y;
    //_scaleDim[2] = _model->scaledDim().z;
    //_projection = (*_ps)["projection"].toFloat();
    _dataTex = &dataTex;
    _tfTex = &tfTex;
    //_tfMapTex = &tfMapTex;

    init();
}

void RayCastingShader::preRender()
{
    _dataTex->bind(0);
    _tfTex->bind(1);

    //Vector3f scaledDim(_scaleDim);
    float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
    _shader->setUniform1f("sampleSpacing", sampleInterval);

    _shader->setUniform4f("lightParam", (*_ps)["ambient"].toFloat(), (*_ps)["diffuse"].toFloat(), (*_ps)["specular"].toFloat(), (*_ps)["shininess"].toFloat());
    //_shader->setUniform1f("projection", _projection);   ////
    _shader->setUniform1f("mapping", _tfEditor->getTFPanel()->getIsTurnOnMapping() ? 1.0f : 0.0f);
    Vector3 v = _renderWindow->getCamera().getCamPosition();
    _shader->setUniform3f("viewVec", (float)v.x(), (float)v.y(), (float)v.z());
    _shader->setUniformSampler("data", *_dataTex);
    _shader->setUniformSampler("tf", *_tfTex);
    _shader->setUniformSampler("tfMapping", 2);

    _shader->use();
}

void RayCastingShader::postRender()
{
    _shader->useFixed();

    _dataTex->release();
    _tfTex->release();
}

void RayCastingShader::reloadShader()
{
    _shader->reload();
}

SliceRayCastingShader::SliceRayCastingShader()
    : RayCastingShader()
{
}

void SliceRayCastingShader::init(QRenderWindow &renderWindow,
                                 QTFEditor &tfEditor,
                                 ParameterSet &ps,
                                 VolumeModel &model,
                                 Box &box,
                                 MSLib::GLTexture3D &dataTex,
                                 MSLib::GLTexture1D &tfTex,
                                 const String &workingPath)
{
    _renderWindow = &renderWindow;
    _tfEditor = &tfEditor;
    _ps = &ps;
    _model = &model;
    _box = &box;
    _workingDir = workingPath;
    _vertShaderFileName = _workingDir + "/shaders/raycasting.vert";
    _fragShaderFileName = _workingDir + "/shaders/sliceRaycasting.frag";

    //_camera = &_renderWindow->getCamera();

    //_scaleDim[0] = scaleDimX;
    //_scaleDim[1] = scaleDimY;
    //_scaleDim[2] = scaleDimZ;
    //_projection = projection;
    _dataTex = &dataTex;
    _tfTex = &tfTex;
    //_tfMapTex = &tfMapTex;

    RayCastingShader::init();

    _shader->addUniform1i("sliceNum", 0);
    float v[10 * 3] = {0.0f};
    for (int i = 0; i < 10 * 3; i++) v[i] = 0.0f;   //// fill
    _shader->addUniform3fv("sliceVec", 10, v);
    _shader->addUniform3fv("slicePnt", 10, v);
}

void SliceRayCastingShader::preRender()
{
    QList<Slicer> &slicers = _renderWindow->getSlicers();
    _shader->setUniform1i("sliceNum", slicers.size());
    float *sliceVec = new float[10 * 3];
    float *slicePnt = new float[10 * 3];
    for (int i = 0; i < 10; i++)
    {
        if (i < slicers.size())
        {
            Vector3 v = slicers[i].getVec();
            sliceVec[i * 3 + 0] = (float)v.x();
            sliceVec[i * 3 + 1] = (float)v.y();
            sliceVec[i * 3 + 2] = (float)v.z();
            v = _box->getCenterPoint() + v * slicers[i].getDist();
            slicePnt[i * 3 + 0] = (float)v.x();
            slicePnt[i * 3 + 1] = (float)v.y();
            slicePnt[i * 3 + 2] = (float)v.z();
        }
        else
        {
            sliceVec[i * 3 + 0] = sliceVec[i * 3 + 1] = sliceVec[i * 3 + 2] = 0.0f;
            slicePnt[i * 3 + 0] = slicePnt[i * 3 + 1] = slicePnt[i * 3 + 2] = 0.0f;
        }
    }
    _shader->setUniform3fv("sliceVec", sliceVec);
    _shader->setUniform3fv("slicePnt", slicePnt);

    RayCastingShader::preRender();
}

void SliceRayCastingShader::postRender()
{
    RayCastingShader::postRender();
}
