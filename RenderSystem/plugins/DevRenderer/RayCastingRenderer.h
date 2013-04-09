#ifndef RAYCASTINGRENDERER_H
#define RAYCASTINGRENDERER_H

#include "GLShader.h"
#include "QParameterSet.h"

// VisKit
#include "QRenderWindow.h"
#include "Camera.h"
#include "QTFEditor.h"

////
#include "VolumeModel.h"

class RayCastingRenderer
{
public:
    //RayCastingRenderer();
    RayCastingRenderer(QRenderWindow &renderWindow,
                       QTFEditor &tfEditor,
                       ParameterSet &ps,
                       VolumeModel &model,
                       const String &workingPath);
    virtual ~RayCastingRenderer();
    /*virtual void init(QRenderWindow &renderWindow,
                      QTFEditor &tfEditor,
                      ParameterSet &ps,
                      VolumeModel &model,
                      const String &workingPath);*/
    virtual void renderBegin();
    virtual void renderEnd();
    virtual void reloadShader();
    virtual void updateTF();
    virtual void updateData();

protected:
    // for inheritors
    RayCastingRenderer(QRenderWindow &renderWindow,
                       QTFEditor &tfEditor,
                       ParameterSet &ps,
                       VolumeModel &model);
    void init();
    void initShader();
    void initTextures();
    int timeStep() { return ((*_ps)["timestep"].toInt() - 1); }
    int varIndex() { return (*_ps)["compIdx"].toInt(); }

protected:
    //String _workingPath;
    String _fragShaderFileName;
    String _vertShaderFileName;

    GLShader *_shader;
    MSLib::GLTexture3D *_dataTex;
    MSLib::GLTexture1D *_tfTex;
    MSLib::GLTexture1D *_tfMapTex;

    // external references
    QRenderWindow *_renderWindow;
    QTFEditor *_tfEditor;
    ParameterSet *_ps;
    VolumeModel *_model;
};

class SliceRayCastingRenderer : public RayCastingRenderer
{
public:
    SliceRayCastingRenderer(QRenderWindow &renderWindow,
                            QTFEditor &tfEditor,
                            ParameterSet &ps,
                            VolumeModel &model,
                            Box &box,
                            const String &workingPath);
    virtual ~SliceRayCastingRenderer();
    virtual void renderBegin();
    virtual void renderEnd();

protected:
    void initShader();

protected:
    Box *_box;      ////
};

////////////////////////////////////////////////////////////////////////////////

class RayCastingShader
{
public:
    RayCastingShader();
    virtual ~RayCastingShader();
    /*virtual void init(QRenderWindow &renderWindow,
                      QTFEditor &tfEditor,
                      ParameterSet &ps,
                      float scaleDimX,
                      float scaleDimY,
                      float scaleDimZ,
                      bool projection,
                      MSLib::GLTexture3D &dataTex,
                      MSLib::GLTexture1D &tfTex,
                      const String &workingDir);*/
    virtual void init(QRenderWindow &renderWindow,
                      QTFEditor &tfEditor,
                      ParameterSet &ps,
                      VolumeModel &model,
                      MSLib::GLTexture3D &dataTex,
                      MSLib::GLTexture1D &tfTex,
                      const String &workingPath);
    virtual void preRender();
    virtual void postRender();
    void reloadShader();

protected:
    void init();

protected:
    String _workingDir;
    String _fragShaderFileName;
    String _vertShaderFileName;

    GLShader *_shader;
    MSLib::GLTexture3D *_dataTex;
    MSLib::GLTexture1D *_tfTex;
    MSLib::GLTexture1D *_tfMapTex;

    ////
    QRenderWindow *_renderWindow;
    QTFEditor *_tfEditor;
    ParameterSet *_ps;
    VolumeModel *_model;
    //float _scaleDim[3];
    //bool _projection;
};

class SliceRayCastingShader : public RayCastingShader
{
public:
    SliceRayCastingShader();
    virtual void init(QRenderWindow &renderWindow,
                      QTFEditor &tfEditor,
                      ParameterSet &ps,
                      VolumeModel &model,
                      Box &box,
                      //float scaleDimX, float scaleDimY, float scaleDimZ, bool projection,
                      MSLib::GLTexture3D &dataTex,
                      MSLib::GLTexture1D &tfTex,
                      const String &workingPath);
    virtual void preRender();
    virtual void postRender();

protected:
    ////
    Box *_box;
};

#endif // RAYCASTINGRENDERER_H
