#include <cmath>
#include "MSVectors.h"
#include "PreIntegrator.h"

#include <QtCore>   ////
#include <QtGui>    ////

PreIntegrator::PreIntegrator()
{
    _baseSample = 0.01f;
    _sampleStep = 0.001f;

    _glWidget = 0;
    _updateRequested = false;
}

PreIntegrator::PreIntegrator(QGLWidget *glWidget)
    : _glWidget(glWidget)
{
    _baseSample = 0.01f;
    _sampleStep = 0.001f;

    _resolution = 1024;

    // setup shader

    _glWidget->makeCurrent();

    _shader = new GLShader();
    _shader->loadVertexShader(QString("./plugins/DevRenderer/shaders/genPreInt.vert").toAscii().constData());
    _shader->loadFragmentShader(QString("./plugins/DevRenderer/shaders/genPreInt.frag").toAscii().constData());
    _shader->link();
    _shader->printVertexShaderInfoLog();
    _shader->printFragmentShaderInfoLog();

    _shader->addUniform1f("invRes", 1.0f / (float)_resolution);
    _shader->addUniform1f("adjFactor", _sampleStep / _baseSample);
    _shader->addUniformSampler("tf", 0);

    //float *temp = new float[1024];  ////
    //_tfTex = new MyLib::GLTexture1D(GL_RGBA32F_ARB, 1024, 0, GL_RGBA, GL_FLOAT, temp);
    //delete [] temp;

    _colorTable = new MSLib::GLTexture2D(GL_RGBA32F, _resolution, _resolution, 0, GL_RGBA, GL_FLOAT, 0);
    _frontTable = new MSLib::GLTexture2D(GL_RGBA32F, _resolution, _resolution, 0, GL_RGBA, GL_FLOAT, 0);
    _backTable = new MSLib::GLTexture2D(GL_RGBA32F, _resolution, _resolution, 0, GL_RGBA, GL_FLOAT, 0);

    glGenRenderbuffers(1, &_rboId);
    glBindRenderbuffer(GL_RENDERBUFFER, _rboId);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 1024);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    _fbo = new MSLib::GLFramebufferObject();
    _fbo->bind();
    _fbo->attachColorTexture(*_colorTable);
    _fbo->attachTexture(*_frontTable, GL_COLOR_ATTACHMENT1);
    _fbo->attachTexture(*_backTable, GL_COLOR_ATTACHMENT2);
    _fbo->attachRenderbuffer(_rboId, GL_DEPTH_ATTACHMENT);
    if (!_fbo->checkStatus())
        qDebug("Error: fbo.checkStatus() == false");
    _fbo->release();


    _updateRequested = false;
}

PreIntegrator::~PreIntegrator()
{
    if (_glWidget != 0)
    {
        delete _colorTable;
        delete _frontTable;
        delete _backTable;
        // rbo
        delete _fbo;
        delete _shader;
    }
}

void PreIntegrator::generateTable(float *table, float *tf, int resolution)
{
    generateTable3(table, tf, resolution);
}

void PreIntegrator::generateTable1(float *table, float *tf, int resolution)
{
    qDebug("%s()", __FUNCTION__);
    for (int j = 0; j < resolution; j++)
    {
        qDebug("j = %d", j);
        for (int i = 0; i < resolution; i++)
        {
            Vector4f color;
            if (i == j)
            {
                color = Vector4f(tf[i * 4], tf[i * 4 + 1], tf[i * 4 + 2], tf[i * 4 + 3]);
                color.w = 1.0f - pow(1.0f - color.w, _sampleStep / _baseSample);
                color.x *= color.w;
                color.y *= color.w;
                color.z *= color.w;
            }
            else // if (i < j)
            {
                //float factor = _sampleStep / ((float)(j - i) * _baseSample);
                float factor = _sampleStep / (abs((float)(i - j)) * _baseSample);
                int d = (i < j ? 1 : -1);
                for (int m = i, n = i + d; m != j; m += d, n += d)
                {
                    Vector4f c((tf[m * 4] + tf[n * 4]) * 0.5f,
                               (tf[m * 4 + 1] + tf[n * 4 + 1]) * 0.5f,
                               (tf[m * 4 + 2] + tf[n * 4 + 2]) * 0.5f,
                               (tf[m * 4 + 3] + tf[n * 4 + 3]) * 0.5f);
                    float alpha = 1.0f - pow(1.0f - c.w, factor);
                    float accAlpha = (1.0f - color.w) * alpha;
                    color.x += accAlpha * c.x;
                    color.y += accAlpha * c.y;
                    color.z += accAlpha * c.z;
                    color.w += accAlpha;
                }
            }

            int index = (j * resolution + i) * 4;
            table[index] = color.x;
            table[index + 1] = color.y;
            table[index + 2] = color.z;
            table[index + 3] = color.w;
        }
    }
}

void PreIntegrator::generateTable2(float *table, float *tf, int resolution)
{
    for (int j = 0; j < resolution; j++)
    {
        qDebug("j = %d", j);
        for (int i = 0; i < resolution; i++)
        {
            Vector4f color;
            if (i == j)
            {
                color = Vector4f(tf[i * 4], tf[i * 4 + 1], tf[i * 4 + 2], tf[i * 4 + 3]);
                color.w = 1.0f - pow(1.0f - color.w, _sampleStep / _baseSample);
                color.x *= color.w;
                color.y *= color.w;
                color.z *= color.w;
            }
            else // if (i < j)
            {
                //float factor = _sampleStep / ((float)(j - i) * _baseSample);
                float factor = _sampleStep / (abs((float)(i - j)) * _baseSample);
                int d = (i < j ? 1 : -1);
                for (int m = i, n = i + d; m != j; m += d, n += d)
                {
                    Vector4f c(tf[m * 4],
                               tf[m * 4 + 1],
                               tf[m * 4 + 2],
                               tf[m * 4 + 3]);
                    float alpha = 1.0f - pow(1.0f - c.w, factor);
                    float accAlpha = (1.0f - color.w) * alpha;
                    color.x += accAlpha * c.x;
                    color.y += accAlpha * c.y;
                    color.z += accAlpha * c.z;
                    color.w += accAlpha;
                }
            }

            int index = (j * resolution + i) * 4;
            table[index] = color.x;
            table[index + 1] = color.y;
            table[index + 2] = color.z;
            table[index + 3] = color.w;
        }
    }
}


void PreIntegrator::generateTable3(float *table, float *tf, int resolution)
{
    float *table2 = new float[resolution * resolution * 4];
    generateTable2(table2, tf, resolution);

    for (int i = 0; i < resolution * resolution; i++)
        table[i] = -1.0e+4f;

    int count = 0;
    float max0 = -1.0e+30f;
    float max1 = -1.0e+30f;
    float max2 = -1.0e+30f;
    float max3 = -1.0e+30f;
    float max4 = -1.0e+30f;

    float baseFactor = _sampleStep / _baseSample;

    // dist = 0; i = j
    for (int i = 0; i < resolution; i++)
    {
        Vector3f color = Vector3f(&tf[i * 4]);
        float alpha = _adjAlpha(tf[i * 4 + 3], baseFactor);
        color *= alpha;
        int index = (i * resolution + i) * 4;
        color.copyTo(&table[index]);
        table[index + 3] = alpha;

        ////
        count++;
        float dif = abs(table[index] - table2[index]);
        if (max0 < dif) max0 = dif;
    }

    for (int dist = 1; dist < resolution; dist++)
    {
        float factor = baseFactor / (float)dist;
        Vector3f color = Vector3f();
        float alpha = 0.0f;
        for (int sign = 1; sign >= -1; sign -= 2) {

            int index;
            if (sign == 1)
            {
                _integrate(color, alpha, 0, dist, tf, factor);
                index = (dist * resolution + 0) * 4;
            }
            else    // sign == -1
            {
                _integrate(color, alpha, resolution - 1, resolution - 1 - dist, tf, factor);
                index = ((resolution - 1 - dist) * resolution + (resolution - 1)) * 4;
            }
            //int index = (sign == 1) ? (dist * resolution + 0) * 4 :
            //                         ((resolution - 1 - dist) * resolution + (resolution - 1)) * 4;
            color.copyTo(&table[index]);
            table[index + 3] = alpha;

            ////
            count++;
            float dif = abs(table[index] - table2[index]);
            if (max1 < dif) max1 = dif;

            //for (int i = (sign == 1 ? 0 : resolution - 1);
            //     (sign == 1 && i < resolution - dist - 1) || (sign == -1 && i > dist);
            //     i += sign)
            for (int i = (sign == 1 ? 0 : resolution - 1), k = 0;
                     k < resolution - dist - 1;
                     i += sign, k++)
            {
                int j = i + sign * dist;
                Vector3f ci = Vector3f(&tf[i * 4]);
                float ai = _adjAlpha(tf[i * 4 + 3], factor);
                Vector3f cj = Vector3f(&tf[j * 4]);
                float aj = _adjAlpha(tf[j * 4 + 3], factor);

                if (ai < 1.0f)
                {
                    // pop
                    color = (color - ci * ai) / (1.0f - ai);
                    alpha = (alpha - ai) / (1.0f - ai);
                    // push
                    float a = (1.0f - alpha) * aj;
                    color += cj * a;
                    alpha += a;
                }
                else
                {
                    //color = Vector3f();
                    //alpha = 0.0f;
                    //for (int m = i + sign; m != j + sign; m += sign)
                    //{
                    //    Vector3f c = Vector3f(&tf[m * 4]);
                    //    float a = _adjAlpha(tf[m * 4 + 3], factor) * (1.0f - alpha);
                    //    color += c * a;
                    //    alpha += a;
                    //}
                    _integrate(color, alpha, i + sign, j + sign, tf, factor);
                }

                int index = ((j + sign) * resolution + i + sign) * 4;
                color.copyTo(&table[index]);
                table[index + 3] = alpha;

                ////
                count++;
                float dif = abs(table[index] - table2[index]);
                if (max2 < dif) max2 = dif;
            }
        }
    }

    qDebug("done.");

    float min = 1.0e+30f;
    float max = -1.0e+30f;
    float difmin = 1.0e+30f;
    float difmax = -1.0e+30f;
    float difsum = 0.0f;

    for (int i = 0; i < resolution * resolution; i++)
    {
        if (min > table[i])
            min = table[i];
        if (max < table[i])
            max = table[i];
        float dif = abs(table[i] - table2[i]);
        if (difmin > dif)
            difmin = dif;
        if (difmax < dif)
            difmax = dif;
        difsum += dif;
        if (table[i] < 0.0f)
            table[i] = 0.0f;
        if (table[i] > 1.0f)
            table[i] = 1.0f;
    }
    qDebug("count = %d, res^2 = %d", count, resolution * resolution);
    qDebug("min = %f", min);
    qDebug("max = %f", max);
    qDebug("difmin = %f", difmin);
    qDebug("difmax = %f", difmax);
    qDebug("difsum = %f", difsum);
    qDebug("difavg = %f", difsum / (float)(resolution * resolution));
    qDebug("max0 = %f", max0);
    qDebug("max1 = %f", max1);
    qDebug("max2 = %f", max2);
    qDebug("max3 = %f", max3);
    qDebug("max4 = %f", max4);

    //for (int i = 0; i < resolution * resolution; i++)
    //    table[i] = table2[i];

    delete [] table2;
}

void PreIntegrator::genColorTable(float *colorTable, float *tf, int resolution)
{
    float *table2 = new float[resolution * resolution * 4];
    generateTable2(table2, tf, resolution);

    for (int i = 0; i < resolution * resolution; i++)
        colorTable[i] = -1.0e+4f;

    int count = 0;
    float max0 = -1.0e+30f;
    float max1 = -1.0e+30f;
    float max2 = -1.0e+30f;
    float max3 = -1.0e+30f;
    float max4 = -1.0e+30f;

    float baseFactor = _sampleStep / _baseSample;

    // dist = 0; i = j
    for (int i = 0; i < resolution; i++)
    {
        Vector3f color = Vector3f(&tf[i * 4]);
        float alpha = _adjAlpha(tf[i * 4 + 3], baseFactor);
        color *= alpha;
        int index = (i * resolution + i) * 4;
        color.copyTo(&colorTable[index]);
        colorTable[index + 3] = alpha;

        ////
        count++;
        float dif = abs(colorTable[index] - table2[index]);
        if (max0 < dif) max0 = dif;
    }

    for (int dist = 1; dist < resolution; dist++)
    {
        float factor = baseFactor / (float)dist;
        Vector3f color = Vector3f();
        float alpha = 0.0f;

        for (int sign = 1; sign >= -1; sign -= 2)   // sign = 1, -1
        {
            int index;
            // integrate the first dist entries
            if (sign == 1)
            {
                _integrate(color, alpha, 0, dist, tf, factor);
                index = (dist * resolution + 0) * 4;
            }
            else    // sign == -1
            {
                _integrate(color, alpha, resolution - 1, resolution - 1 - dist, tf, factor);
                index = ((resolution - 1 - dist) * resolution + (resolution - 1)) * 4;
            }
            color.copyTo(&colorTable[index]);
            colorTable[index + 3] = alpha;

            ////
            count++;
            float dif = abs(colorTable[index] - table2[index]);
            if (max1 < dif) max1 = dif;

            for (int i = (sign == 1 ? 0 : resolution - 1), k = 0;
                     k < resolution - dist - 1;
                     i += sign, k++)
            {
                int j = i + sign * dist;
                Vector3f ci = Vector3f(&tf[i * 4]);
                float ai = _adjAlpha(tf[i * 4 + 3], factor);
                Vector3f cj = Vector3f(&tf[j * 4]);
                float aj = _adjAlpha(tf[j * 4 + 3], factor);

                if (ai < 1.0f)
                {
                    // pop
                    color = (color - ci * ai) / (1.0f - ai);
                    alpha = (alpha - ai) / (1.0f - ai);
                    // push
                    float a = (1.0f - alpha) * aj;
                    color += cj * a;
                    alpha += a;
                }
                else
                {
                    _integrate(color, alpha, i + sign, j + sign, tf, factor);
                }

                int index = ((j + sign) * resolution + i + sign) * 4;
                color.copyTo(&colorTable[index]);
                colorTable[index + 3] = alpha;

                ////
                count++;
                float dif = abs(colorTable[index] - table2[index]);
                if (max2 < dif) max2 = dif;
            }
        }
    }

    qDebug("done.");

    float min = 1.0e+30f;
    float max = -1.0e+30f;
    float difmin = 1.0e+30f;
    float difmax = -1.0e+30f;
    float difsum = 0.0f;

    for (int i = 0; i < resolution * resolution; i++)
    {
        if (min > colorTable[i])
            min = colorTable[i];
        if (max < colorTable[i])
            max = colorTable[i];
        float dif = abs(colorTable[i] - table2[i]);
        if (difmin > dif)
            difmin = dif;
        if (difmax < dif)
            difmax = dif;
        difsum += dif;
        if (colorTable[i] < 0.0f)
            colorTable[i] = 0.0f;
        //if (colorTable[i] > 1.0f)
        //    colorTable[i] = 1.0f;
    }
    qDebug("count = %d, res^2 = %d", count, resolution * resolution);
    qDebug("min = %f", min);
    qDebug("max = %f", max);
    qDebug("difmin = %f", difmin);
    qDebug("difmax = %f", difmax);
    qDebug("difsum = %f", difsum);
    qDebug("difavg = %f", difsum / (float)(resolution * resolution));
    qDebug("max0 = %f", max0);
    qDebug("max1 = %f", max1);
    qDebug("max2 = %f", max2);
    qDebug("max3 = %f", max3);
    qDebug("max4 = %f", max4);

    //for (int i = 0; i < resolution * resolution; i++)
    //    table[i] = table2[i];

    delete [] table2;
}

void PreIntegrator::generateTables1(float *colorTable, float *frontTable, float *backTable, float *tf, int resolution)
{
    genColorTable(colorTable, tf, resolution);

    float *rampedTF = new float[resolution * 4];    // back-ramped TF
    for (int i = 0; i < resolution; i++)
    {
        rampedTF[i * 4] = tf[i * 4] * (float)i;
        rampedTF[i * 4 + 1] = tf[i * 4 + 1] * (float)i;
        rampedTF[i * 4 + 2] = tf[i * 4 + 2] * (float)i;
        rampedTF[i * 4 + 3] = tf[i * 4 + 3];
    }

    float *gbwTable = new float[resolution * resolution * 4];   // globally-ramped back-weighted table
    genColorTable(gbwTable, rampedTF, resolution);

    ////
    int badCount = 0;

    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            int index = (j * resolution + i) * 4;
            if (i == j)
            {
                backTable[index] = 0.0f;
                backTable[index + 1] = 0.0f;
                backTable[index + 2] = 0.0f;
                backTable[index + 3] = 0.0f;
            }
            else //if (i < j)
            {
                float factor = 1.0f / (float)(j - i);
                backTable[index] = (gbwTable[index] - (float)i * colorTable[index]) * factor;
                backTable[index + 1] = (gbwTable[index + 1] - (float)i * colorTable[index + 1]) * factor;
                backTable[index + 2] = (gbwTable[index + 2] - (float)i * colorTable[index + 2]) * factor;
                backTable[index + 3] = (gbwTable[index + 3] - (float)i * colorTable[index + 3]) * factor;   // wrong

                if (backTable[index] > colorTable[index] + 1.0e-4f) badCount++;
                else if (backTable[index + 1] > colorTable[index + 1] + 1.0e-4f) badCount++;
                else if (backTable[index + 2] > colorTable[index + 2] + 1.0e-4f) badCount++;
                //else if (backTable[index + 3] > colorTable[index + 3]) badCount++;

                backTable[index] = _clamp(backTable[index], 0.0f, colorTable[index]);
                backTable[index + 1] = _clamp(backTable[index + 1], 0.0f, colorTable[index + 1]);
                backTable[index + 2] = _clamp(backTable[index + 2], 0.0f, colorTable[index + 2]);
                backTable[index + 3] = _clamp(backTable[index + 3], 0.0f, colorTable[index + 3]);
            }
            frontTable[index] = colorTable[index] - backTable[index];
            frontTable[index + 1] = colorTable[index + 1] - backTable[index + 1];
            frontTable[index + 2] = colorTable[index + 2] - backTable[index + 2];
            frontTable[index + 3] = colorTable[index + 3] - backTable[index + 3];
        }
    }

    qDebug("badCount = %d", badCount);
}

void PreIntegrator::generateTables(float *colorTable, float *frontTable, float *backTable, float *tf, int resolution)
{
    float baseFactor = _sampleStep / _baseSample;

    float *gbwTable = new float[resolution * resolution * 4];   // globally-ramped back-weighted table

    // dist = 0; i = j
    for (int i = 0; i < resolution; i++)
    {
        Vector3f color = Vector3f(&tf[i * 4]);
        float alpha = _adjAlpha(tf[i * 4 + 3], baseFactor);
        color *= alpha;
        int index = (i * resolution + i) * 4;
        color.copyTo(&colorTable[index]);
        colorTable[index + 3] = alpha;

        Vector3f colorw = color * (float)i;
        colorw.copyTo(&gbwTable[index]);
        float alphaw = alpha * (float)i;
        gbwTable[index + 3] = alphaw;
    }

    for (int dist = 1; dist < resolution; dist++)
    {
        float factor = baseFactor / (float)dist;
        Vector3f color = Vector3f();
        float alpha = 0.0f;
        Vector3f colorw = Vector3f();
        float alphaw = 0.0f;

        for (int sign = 1; sign >= -1; sign -= 2)   // sign = 1, -1
        {
            int index;
            // integrate the first dist entries
            if (sign == 1)
            {
                _integrateWeighted(color, alpha, colorw, alphaw, 0, dist, tf, factor);
                index = (dist * resolution + 0) * 4;
            }
            else    // sign == -1
            {
                _integrateWeighted(color, alpha, colorw, alphaw, resolution - 1, resolution - 1 - dist, tf, factor);
                index = ((resolution - 1 - dist) * resolution + (resolution - 1)) * 4;
            }
            color.copyTo(&colorTable[index]);
            colorTable[index + 3] = alpha;

            colorw.copyTo(&gbwTable[index]);
            gbwTable[index + 3] = alphaw;

            for (int i = (sign == 1 ? 0 : resolution - 1), k = 0;
                     k < resolution - dist - 1;
                     i += sign, k++)
            {
                int j = i + sign * dist;
                Vector3f ci = Vector3f(&tf[i * 4]);
                float ai = _adjAlpha(tf[i * 4 + 3], factor);
                Vector3f cj = Vector3f(&tf[j * 4]);
                float aj = _adjAlpha(tf[j * 4 + 3], factor);

                if (ai < 1.0f)
                {
                    // pop
                    color = (color - ci * ai) / (1.0f - ai);
                    alpha = (alpha - ai) / (1.0f - ai);
                    colorw = (colorw - ci * ai * (float)i) / (1.0f - ai);
                    alphaw = (alphaw - ai * (float)i) / (1.0f - ai);
                    // push
                    float a = (1.0f - alpha) * aj;
                    color += cj * a;
                    alpha += a;
                    colorw += cj * a * (float)j;
                    alphaw += a * (float)j;
                }
                else
                {
                    _integrateWeighted(color, alpha, colorw, alphaw, i + sign, j + sign, tf, factor);
                }

                ////
                color.x = _clamp(color.x, 0.0f, 1.0f);
                color.y = _clamp(color.y, 0.0f, 1.0f);
                color.z = _clamp(color.z, 0.0f, 1.0f);
                alpha = _clamp(alpha, 0.0f, 1.0f);

                int index = ((j + sign) * resolution + i + sign) * 4;
                color.copyTo(&colorTable[index]);
                colorTable[index + 3] = alpha;

                colorw.copyTo(&gbwTable[index]);
                gbwTable[index + 3] = alphaw;
            }
        }
    }

    qDebug("done.");

    ////
    //float *table2 = new float[resolution * resolution * 4];
    //generateTable2(table2, tf, resolution);

    //generateTable2(colorTable, tf, resolution);

    float min = 1.0e+30f;
    float max = -1.0e+30f;
    //float difmin = 1.0e+30f;
    ///float difmax = -1.0e+30f;
    //float difsum = 0.0f;

    for (int i = 0; i < resolution * resolution; i++)
    {
        if (min > colorTable[i])
            min = colorTable[i];
        if (max < colorTable[i])
            max = colorTable[i];
        //float dif = abs(colorTable[i] - table2[i]);
        //if (difmin > dif) difmin = dif;
        //if (difmax < dif) difmax = dif;
        //difsum += dif;
        if (colorTable[i] < 0.0f)
            colorTable[i] = 0.0f;
        //if (colorTable[i] > 1.0f)
        //    colorTable[i] = 1.0f;
    }
    //qDebug("count = %d, res^2 = %d", count, resolution * resolution);
    qDebug("min = %f", min);
    qDebug("max = %f", max);
    //qDebug("difmin = %f", difmin);
    //qDebug("difmax = %f", difmax);
    //qDebug("difsum = %f", difsum);
    //qDebug("difavg = %f", difsum / (float)(resolution * resolution));


    /*float *rampedTF = new float[resolution * 4];    // back-ramped TF
    for (int i = 0; i < resolution; i++)
    {
        rampedTF[i * 4] = tf[i * 4] * (float)i;
        rampedTF[i * 4 + 1] = tf[i * 4 + 1] * (float)i;
        rampedTF[i * 4 + 2] = tf[i * 4 + 2] * (float)i;
        rampedTF[i * 4 + 3] = tf[i * 4 + 3];
    }

    float *gbwTable = new float[resolution * resolution * 4];   // globally-ramped back-weighted table
    genColorTable(gbwTable, rampedTF, resolution);*/

    ////
    int badCount = 0;

    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            int sign = (i < j) ? 1 : -1;
            int index = (j * resolution + i) * 4;
            if (i == j || i + sign == j)
            {
                Vector4f bwcolor(0.0f, 0.0f, 0.0f, 0.0f);
                bwcolor.copyTo(&backTable[index]);
                //backTable[index] = 0.0f;
                //backTable[index + 1] = 0.0f;
                //backTable[index + 2] = 0.0f;
                //backTable[index + 3] = 0.0f;
                Vector4f fwcolor(&colorTable[index]);
                fwcolor.copyTo(&frontTable[index]);
            }
            else //if (i < j)
            {
                float factor = 1.0f / (float)(j - i + sign);
                Vector4f color(&colorTable[index]);
                Vector4f gbwcolor(&gbwTable[index]);
                Vector4f bwcolor = (gbwcolor - color * (float)i) * factor;

                //backTable[index] = (gbwTable[index] - (float)i * colorTable[index]) * factor;
                //backTable[index + 1] = (gbwTable[index + 1] - (float)i * colorTable[index + 1]) * factor;
                //backTable[index + 2] = (gbwTable[index + 2] - (float)i * colorTable[index + 2]) * factor;
                //backTable[index + 3] = (gbwTable[index + 3] - (float)i * colorTable[index + 3]) * factor;

                //if (backTable[index] > colorTable[index] + 1.0e-4f) badCount++;
                //else if (backTable[index + 1] > colorTable[index + 1] + 1.0e-4f) badCount++;
                //else if (backTable[index + 2] > colorTable[index + 2] + 1.0e-4f) badCount++;
                //else if (backTable[index + 3] > colorTable[index + 3]) badCount++;
                if (bwcolor.x > color.x + 1.0e-4f) badCount++;
                else if (bwcolor.y > color.y + 1.0e-4f) badCount++;
                else if (bwcolor.z > color.z + 1.0e-4f) badCount++;
                else if (bwcolor.w > color.w + 1.0e-4f) badCount++;

                //backTable[index] = _clamp(backTable[index], 0.0f, colorTable[index]);
                //backTable[index + 1] = _clamp(backTable[index + 1], 0.0f, colorTable[index + 1]);
                //backTable[index + 2] = _clamp(backTable[index + 2], 0.0f, colorTable[index + 2]);
                //backTable[index + 3] = _clamp(backTable[index + 3], 0.0f, colorTable[index + 3]);
                bwcolor.x = _clamp(bwcolor.x, 0.0f, color.x);
                bwcolor.y = _clamp(bwcolor.y, 0.0f, color.y);
                bwcolor.z = _clamp(bwcolor.z, 0.0f, color.z);
                bwcolor.w = _clamp(bwcolor.w, 0.0f, color.w);
                bwcolor.copyTo(&backTable[index]);

                Vector4f fwcolor = color - bwcolor;
                fwcolor.copyTo(&frontTable[index]);
            }
            //frontTable[index] = colorTable[index] - backTable[index];
            //frontTable[index + 1] = colorTable[index + 1] - backTable[index + 1];
            //frontTable[index + 2] = colorTable[index + 2] - backTable[index + 2];
            //frontTable[index + 3] = colorTable[index + 3] - backTable[index + 3];

        }
    }

    qDebug("badCount = %d", badCount);

    //genColorTable(colorTable, tf, resolution);
}

void PreIntegrator::update(float *colorTable, float *frontTable, float *backTable, MSLib::GLTexture1D *tfTex, int resolution)
{
    if (!_updateRequested) return;

    qDebug("%s()", __FUNCTION__);

    _glWidget->makeCurrent();

    //qDebug("check 1");

    _fbo->bind();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, 1024, 1024);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -10.0, 10.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    //_colorTable->bind();

    GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, buffers);

    tfTex->bind(0);

    _shader->use();

   //qDebug("check 2");

    glBegin(GL_QUADS);

    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(+1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(+1.0f, +1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, +1.0f);

    glEnd();

    _shader->useFixed();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();

    _fbo->release();

    //qDebug("check 3");

    //*
    /*unsigned char *buf = new unsigned char[1024 * 1024 * 4];
    //_colorTable->getImage(GL_BGRA, GL_UNSIGNED_BYTE, buf);
    _frontTable->getImage(GL_BGRA, GL_UNSIGNED_BYTE, buf);
    QImage img(buf, 1024, 1024, QImage::Format_ARGB32);
    QLabel *label = new QLabel();
    label->setPixmap(QPixmap::fromImage(img));
    label->show();
    delete [] buf;*/
    //*/

    _colorTable->getImage(GL_RGBA, GL_FLOAT, colorTable);
    _frontTable->getImage(GL_RGBA, GL_FLOAT, frontTable);
    _backTable->getImage(GL_RGBA, GL_FLOAT, backTable);

    _updateRequested = false;
}
