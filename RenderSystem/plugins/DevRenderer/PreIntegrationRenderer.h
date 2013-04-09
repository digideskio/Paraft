#ifndef PREINTEGRATIONRENDERER_H
#define PREINTEGRATIONRENDERER_H

#include "RayCastingRenderer.h"
#include "PreIntegratorGL.h"

class PreIntegrationRenderer : public RayCastingShader
{
public:
    PreIntegrationRenderer();
    virtual ~PreIntegrationRenderer();
    virtual void init(QRenderWindow &renderWindow,
                      QTFEditor &tfEditor,
                      ParameterSet &ps,
                      VolumeModel &model,
                      //float scaleDimX, float scaleDimY, float scaleDimZ, bool projection,
                      MSLib::GLTexture3D &dataTex,
                      MSLib::GLTexture1D &tfTex,
                      const String &workingPath,
                      PreIntegratorGL *preIntegrator = nullptr);
    virtual void preRender();
    virtual void postRender();
    PreIntegratorGL *getPreIntegrator() { return _preIntegrator; }
    void setStepSize(float stepSize);
    void update(MSLib::GLTexture1D &tfTex);

protected:
    PreIntegratorGL *_preIntegrator;
    bool _internalPreIntegratorUsed;
};

#endif // PREINTEGRATIONRENDERER_H
