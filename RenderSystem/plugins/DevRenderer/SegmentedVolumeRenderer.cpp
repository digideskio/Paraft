#include "SegmentedVolumeRenderer.h"

SegmentedVolumeRenderer::SegmentedVolumeRenderer(QRenderWindow &renderWindow,
                                                 QTFEditor &tfEditor,
                                                 ParameterSet &ps,
                                                 VolumeModel &model,
                                                 Box &box,
                                                 const String &workingPath)
    : VolumeRenderer(renderWindow, tfEditor, ps, model)
      //_renderWindow(&renderWindow),
      //_tfEditor(&tfEditor),
      //_ps(&ps),
      //_model(&model),
      //_box(&box),
      //_slicerEnabled(false)
{
    qDebug("Init SegmentedVolumeRenderer...");

    VolumeRenderer::_box = &box;

    /*for (int i = 0; i < _model->blockCount(); i++)
    {
        RegularGridDataBlock &dataBlock = _model->volumeDataBlock(i, timeStep(), varIndex());
        _dataBlockTex.append(new MSLib::GLTexture3D(GL_R32F,
                                                    dataBlock.dim().x,
                                                    dataBlock.dim().y,
                                                    dataBlock.dim().z,
                                                    0,
                                                    GL_RED,
                                                    GL_FLOAT,
                                                    dataBlock.data()));
        Vector3i lo = dataBlock.lo();
        Vector3i hi = dataBlock.hi();
        qDebug("Block[%d]: lo(%d, %d, %d), hi(%d, %d, %d)", i, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
        Vector3f slo = dataBlock.boxLo();
        Vector3f shi = dataBlock.boxHi();
        qDebug("Block[%d]: slo(%f, %f, %f), shi(%f, %f, %f)", i, slo.x, slo.y, slo.z, shi.x, shi.y, shi.z);
    }*/
    initDataTexture();

    /*_tfTex = new MSLib::GLTexture1D(GL_RGBA32F,
                                    _tfEditor->getTFColorMapResolution(),
                                    0,
                                    GL_RGBA,
                                    GL_FLOAT,
                                    _tfEditor->getTFColorMap());*/
    VolumeRenderer::initTFTexture();

    /*float sampleInterval = (*_ps)["sampleStep"].toFloat() * _model->scaledDim().length();
    _preIntegrator = new PreIntegratorGL(_tfEditor->getTFColorMapResolution(), sampleInterval, 0.01f);
    _preIntegrator->update(*_tfTex);*/
    VolumeRenderer::initPreIntegrator();

    /*_segmentedRayCastingShader = new SegmentedRayCastingRenderer();
    _segmentedRayCastingShader->init(*_renderWindow, *_tfEditor, *_ps, *_model, _dataBlockTex, *_tfTex, workingPath);

    _segmentedPreIntegrationShader = new SegmentedPreIntegrationRenderer();
    _segmentedPreIntegrationShader->init(*_renderWindow, *_tfEditor, *_ps, *_model, _dataBlockTex, *_tfTex, workingPath, _preIntegrator);*/
    initShaders(workingPath);
}

SegmentedVolumeRenderer::~SegmentedVolumeRenderer()
{
    if (_segmentedRayCastingShader     != nullptr) delete _segmentedRayCastingShader;
    if (_segmentedPreIntegrationShader != nullptr) delete _segmentedPreIntegrationShader;

    //delete _preIntegrator;

    for (int i = 0; i < _dataBlockTex.size(); i++)
        delete _dataBlockTex[i];

    //delete _tfTex;
}

void SegmentedVolumeRenderer::initDataTexture()
{
    for (int i = 0; i < _model->blockCount(); i++)
    {
        _model->loadData(timeStep(), varIndex());
        RegularGridDataBlock &dataBlock = _model->volumeDataBlock(i, timeStep(), varIndex());
        _dataBlockTex.append(new MSLib::GLTexture3D(GL_R32F,
                                                    dataBlock.dim().x,
                                                    dataBlock.dim().y,
                                                    dataBlock.dim().z,
                                                    0,
                                                    GL_RED,
                                                    GL_FLOAT,
                                                    dataBlock.data()));

        Vector3i lo = dataBlock.lo();
        Vector3i hi = dataBlock.hi();
        qDebug("Block[%d]: lo(%d, %d, %d), hi(%d, %d, %d)", i, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
        Vector3f slo = dataBlock.boxLo();
        Vector3f shi = dataBlock.boxHi();
        qDebug("Block[%d]: slo(%f, %f, %f), shi(%f, %f, %f)", i, slo.x, slo.y, slo.z, shi.x, shi.y, shi.z);
    }
}

void SegmentedVolumeRenderer::initShaders(const String &workingPath)
{
    _segmentedRayCastingShader = new SegmentedRayCastingRenderer();
    _segmentedRayCastingShader->init(*_renderWindow, *_tfEditor, *_ps, *_model, _dataBlockTex, *_tfTex, workingPath);

    _segmentedPreIntegrationShader = new SegmentedPreIntegrationRenderer();
    _segmentedPreIntegrationShader->init(*_renderWindow, *_tfEditor, *_ps, *_model, _dataBlockTex, *_tfTex, workingPath, _preIntegrator);
}

void SegmentedVolumeRenderer::updateData()
{
    for (int i = 0; i < _model->blockCount(timeStep(), varIndex()); i++)
    {
        qDebug("load subblock %d", i);
        _model->loadData(timeStep(), varIndex());
        float *dataBlock = _model->volumeDataBlock(i, timeStep(), varIndex()).data();
        _dataBlockTex[i]->load(dataBlock);
    }
}

void SegmentedVolumeRenderer::renderBegin()
{
    if (preIntegrationEnabled())
    {
        _segmentedPreIntegrationShader->preRender();
    }
    else
    {
        if (_slicerEnabled)
        {}//_sliceRayCastingShader->preRender();
        else
            _segmentedRayCastingShader->preRender();
    }
}

void SegmentedVolumeRenderer::renderBegin(int blockOrder, MSLib::GLTexture2D &colorBuffer)
{
    if (preIntegrationEnabled())
        _segmentedPreIntegrationShader->preRender(blockOrder, colorBuffer);
    else
        _segmentedRayCastingShader->preRender(blockOrder, colorBuffer);
}

void SegmentedVolumeRenderer::renderEnd()
{
    if (preIntegrationEnabled())
    {
        _segmentedPreIntegrationShader->postRender();
    }
    else
    {
        if (_slicerEnabled)
        {}//_sliceRayCastingShader->postRender();
        else
            _segmentedRayCastingShader->postRender();
    }
}

/*void SegmentedVolumeRenderer::setPreIntegrationSampleInterval(float sampleInterval)
{
    _preIntegrator->setStepSize(sampleInterval);
}*/

void SegmentedVolumeRenderer::reloadShader()
{
    _segmentedRayCastingShader->reloadShader();
    _segmentedPreIntegrationShader->reloadShader();
}
