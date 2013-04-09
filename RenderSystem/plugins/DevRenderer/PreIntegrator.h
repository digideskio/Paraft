#ifndef PREINTEGRATOR_H
#define PREINTEGRATOR_H

#include <GL/glew.h>

#include <cmath>

#include <QGLWidget>

#include "MSVectors.h"
#include "MSGLTexture.h"
#include "MSGLFramebufferObject.h"
#include "GLShader.h"

class PreIntegrator
{
public:
    PreIntegrator();
    PreIntegrator(QGLWidget *glWidget);
    ~PreIntegrator();
    void generateTable(float *table, float *tf, int resolution);
    void generateTable1(float *table, float *tf, int resolution);
    void generateTable2(float *table, float *tf, int resolution);
    void generateTable3(float *table, float *tf, int resolution);
    void genColorTable(float *colorTable, float *tf, int resolution);
    void generateTables1(float *colorTable, float *frontTable, float *backTable, float *tf, int resolution);
    void generateTables(float *colorTable, float *frontTable, float *backTable, float *tf, int resolution);
    void setStepSize(float stepSize) { _sampleStep = stepSize; }
    //void generateTablesGL(float *colorTable, float *frontTable, float *backTable, float *tf, int resolution);   ////
    void requestUpdate() { _updateRequested = true; }
    void update(float *colorTable, float *frontTable, float *backTable, MSLib::GLTexture1D *tfTex, int resolution);

protected:
    float _adjAlpha(float alpha, float stepRatio);
    void _integrate(Vector3f &color, float &alpha, int front, int back, float *tf, float stepRatio);
    void _integrateWeighted(Vector3f &color, float &alpha, Vector3f &colorw, float &alphaw, int front, int back, float *tf, float stepRatio);
    float _clamp(float x, float minVal, float maxVal) { return (x < minVal) ? minVal : (x > maxVal ? maxVal : x); }
    void _clamp(Vector4f &v, float minVal, float maxVal);   ////

protected:
    QGLWidget *_glWidget;
    GLShader *_shader;
    MSLib::GLFramebufferObject *_fbo;
    //MyLib::GLTexture1D *_tfTex;
    MSLib::GLTexture2D *_colorTable;
    MSLib::GLTexture2D *_frontTable;
    MSLib::GLTexture2D *_backTable;
    GLuint _rboId;
    bool _updateRequested;

    float _baseSample;
    float _sampleStep;

    int _resolution;
};

inline float PreIntegrator::_adjAlpha(float alpha, float stepRatio)
{
    return (1.0f - pow(1.0f - alpha, stepRatio));
}

inline void PreIntegrator::_integrate(Vector3f &color, float &alpha, int front, int back, float *tf, float stepRatio)
{
    // front-to-back integration
    // front..back-1
    color = Vector3f();
    alpha = 0.0f;
    int sign = (front <= back) ? 1 : -1;
    for (int i = front; i != back; i += sign)
    {
        Vector3f c = Vector3f(&tf[i * 4]);
        float a = _adjAlpha(tf[i * 4 + 3], stepRatio) * (1.0f - alpha);
        color += c * a;
        alpha += a;
    }
}

inline void PreIntegrator::_integrateWeighted(Vector3f &color, float &alpha, Vector3f &colorw, float &alphaw, int front, int back, float *tf, float stepRatio)
{
    // front-to-back integration
    // front..back-1
    color = Vector3f();
    alpha = 0.0f;
    colorw = Vector3f();
    alphaw = 0.0f;
    int sign = (front <= back) ? 1 : -1;
    for (int i = front; i != back; i += sign)
    {
        Vector3f c = Vector3f(&tf[i * 4]);
        float a = _adjAlpha(tf[i * 4 + 3], stepRatio) * (1.0f - alpha);
        color += c * a;
        alpha += a;
        colorw += c * a * (float)i;
        alphaw += a * (float)i;
    }
}

////
inline void PreIntegrator::_clamp(Vector4f &v, float minVal, float maxVal)
{
    v.x = _clamp(v.x, minVal, maxVal);
    v.y = _clamp(v.y, minVal, maxVal);
    v.z = _clamp(v.z, minVal, maxVal);
    v.w = _clamp(v.w, minVal, maxVal);
}

#endif // PREINTEGRATOR_H
