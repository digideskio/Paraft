#ifndef PREINTEGRATORGL_H
#define PREINTEGRATORGL_H

#include "GLShader.h"
#include "MSGLFramebufferObject.h"
#include "MSGLTexture.h"

class PreIntegratorGL
{
public:
    PreIntegratorGL(int resolution = 1024, float stepSize = 0.001f, float baseStepSize = 0.01f);
    ~PreIntegratorGL();
    MSLib::GLTexture2D *colorTable() { return _colorTable; }
    MSLib::GLTexture2D *frontTable() { return _frontTable; }
    MSLib::GLTexture2D *backTable() { return _backTable; }
    void setStepSize(float stepSize) { _stepSize = stepSize; }

    void update(MSLib::GLTexture1D &tfTex);
    void update(float *colorTable, float *frontTable, float *backTable, const float *tf, int tfSize);

protected:
    GLShader *_shader;
    MSLib::GLFramebufferObject *_fbo;
    MSLib::GLTexture2D *_colorTable;
    MSLib::GLTexture2D *_frontTable;
    MSLib::GLTexture2D *_backTable;

    int _resolution;
    float _stepSize;        // sample step size
    float _baseStepSize;    // base sample step size
};

#endif // PREINTEGRATORGL_H
