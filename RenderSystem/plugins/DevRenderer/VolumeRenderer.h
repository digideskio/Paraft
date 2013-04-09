#ifndef VOLUMERENDERER_H
#define VOLUMERENDERER_H

#include "PreIntegratorGL.h"
#include "RayCastingRenderer.h"
#include "PreIntegrationRenderer.h"

class VolumeRenderer
{
public:
    VolumeRenderer(QRenderWindow &renderWindow,
                   QTFEditor &tfEditor,
                   ParameterSet &ps,
                   VolumeModel &model,
                   Box &box,
                   const String &workingPath);
    virtual ~VolumeRenderer();
    virtual void updateData();
    virtual void updateTF();
    virtual void renderBegin();
    virtual void renderEnd();
    //void setPreIntegrationSampleInterval(float sampleInterval);
    void setSlicerEnabled(bool enable) { _slicerEnabled = enable; }
    virtual void reloadShader();

    PreIntegratorGL *getPreIntegrator() { return _preIntegrator; }

protected:
    VolumeRenderer(QRenderWindow &renderWindow,
                   QTFEditor &tfEditor,
                   ParameterSet &ps,
                   VolumeModel &model);             // for inheritors
    void initDataTexture();
    void initTFTexture();
    void initPreIntegrator();
    void initShaders(const String &workingPath);

    int timeStep() { return ((*_ps)["timestep"].toInt() - 1); }
    int varIndex() { return (*_ps)["compIdx"].toInt(); }
    bool preIntegrationEnabled() { return (*_ps)["preIntEnabled"].toBool(); }

protected:
    QRenderWindow *_renderWindow;
    QTFEditor *_tfEditor;
    ParameterSet *_ps;
    VolumeModel *_model;
    Box *_box;              ////
    //String _workingPath;

    RayCastingShader *_rayCastingShader;
    SliceRayCastingShader *_sliceRayCastingShader;
    PreIntegrationRenderer *_preIntegrationShader;

    MSLib::GLTexture3D *_dataTex;
    MSLib::GLTexture1D *_tfTex;

    PreIntegratorGL *_preIntegrator;

    bool _slicerEnabled;
};

#endif // VOLUMERENDERER_H
