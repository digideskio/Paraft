#ifndef SEGMENTEDVOLUMERENDERER_H
#define SEGMENTEDVOLUMERENDERER_H

#include <QtCore>

//#include "PreIntegratorGL.h"
#include "VolumeRenderer.h"
#include "SegmentedRayCastingRenderer.h"

class SegmentedVolumeRenderer : public VolumeRenderer
{
public:
    SegmentedVolumeRenderer(QRenderWindow &renderWindow,
                            QTFEditor &tfEditor,
                            ParameterSet &ps,
                            VolumeModel &model,
                            Box &box,
                            const String &workingPath);
    virtual ~SegmentedVolumeRenderer();
    virtual void updateData();
    //virtual void updateTF();
    virtual void renderBegin();
    virtual void renderBegin(int blockOrder, MSLib::GLTexture2D &colorBuffer);
    virtual void renderEnd();
    //void setPreIntegrationSampleInterval(float sampleInterval);
    //void setSlicerEnabled(bool enable) { _slicerEnabled = enable; }
    virtual void reloadShader();

    PreIntegratorGL *getPreIntegrator() { return _preIntegrator; }

protected:
    void initDataTexture();
    void initShaders(const String &workingPath);

    int timeStep() { return ((*_ps)["timestep"].toInt() - 1); }
    int varIndex() { return (*_ps)["compIdx"].toInt(); }
    bool preIntegrationEnabled() { return (*_ps)["preIntEnabled"].toBool(); }

protected:
    //QRenderWindow *_renderWindow;
    //QTFEditor *_tfEditor;
    //ParameterSet *_ps;
    //VolumeModel *_model;
    //Box *_box;              ////

    SegmentedRayCastingRenderer *_segmentedRayCastingShader;
    SegmentedPreIntegrationRenderer *_segmentedPreIntegrationShader;

    QList<MSLib::GLTexture3D *> _dataBlockTex;
    //MSLib::GLTexture1D *_tfTex;

    //PreIntegratorGL *_preIntegrator;

    //bool _slicerEnabled;
};

#endif // SEGMENTEDVOLUMERENDERER_H
