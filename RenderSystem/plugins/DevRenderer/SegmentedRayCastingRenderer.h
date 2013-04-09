#ifndef SEGMENTEDRAYCASTINGRENDERER_H
#define SEGMENTEDRAYCASTINGRENDERER_H

#include <QtCore>

#include "Containers.h"
#include "GLShader.h"
#include "ParameterSet.h"
#include "VolumeModel.h"
#include "PreIntegratorGL.h"

// VisKit
#include "QRenderWindow.h"
#include "QTFEditor.h"

class SegmentedRayCastingRenderer
{
public:
    SegmentedRayCastingRenderer();
    virtual ~SegmentedRayCastingRenderer();
    virtual void init(QRenderWindow &renderWindow,
                      QTFEditor &tfEditor,
                      ParameterSet &ps,
                      VolumeModel &model,
                      QList<MSLib::GLTexture3D *> &dataBlockTex,
                      MSLib::GLTexture1D &tfTex,
                      const String &workingPath);

    virtual void preRender();
    virtual void preRender(int blockOrder, MSLib::GLTexture2D &colorBuffer);
    virtual void postRender();
    void reloadShader();

protected:
    void init();

protected:
    String _workingPath;
    String _vertShaderFileName;
    String _fragShaderFileName;
    GLShader *_shader;

    ParameterSet *_ps;
    VolumeModel *_model;
    QList<MSLib::GLTexture3D *> *_dataBlockTex;
    MSLib::GLTexture1D *_tfTex;

    QRenderWindow *_renderWindow;
    QTFEditor *_tfEditor;

    Vector<std::pair<float, int> > _visOrder;
    int _boundBlock;
    MSLib::GLTexture2D *_boundColorBuffer;
};

class SegmentedPreIntegrationRenderer : public SegmentedRayCastingRenderer
{
public:
    SegmentedPreIntegrationRenderer();
    virtual ~SegmentedPreIntegrationRenderer();
    virtual void init(QRenderWindow &renderWindow,
                      QTFEditor &tfEditor,
                      ParameterSet &ps,
                      VolumeModel &model,
                      QList<MSLib::GLTexture3D *> &dataBlockTex,
                      MSLib::GLTexture1D &tfTex,
                      const String &workingPath,
                      PreIntegratorGL *preIntegrator = nullptr);
    virtual void preRender();
    virtual void preRender(int blockOrder, MSLib::GLTexture2D &colorBuffer);
    virtual void postRender();

protected:
    PreIntegratorGL *_preIntegrator;
    bool _internalPreIntegratorUsed;
};

#endif // SEGMENTEDRAYCASTINGRENDERER_H
