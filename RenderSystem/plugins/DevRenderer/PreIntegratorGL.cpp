#include <cstring>

#include "PreIntegratorGL.h"

static const char *vertexShader =
"\
void main()\n\
{\n\
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n\
}\n\
";

static const char *fragmentShader =
"\
uniform sampler1D   tf;\n\
\n\
uniform float       invRes;     // inversed TF resolution (ex: 1.0 / 1024)\n\
uniform float       adjFactor;  // = stepSize / baseStepSize (ex: 0.001 / 0.01)\n\
\n\
void main()\n\
{\n\
    float beginPos = gl_FragCoord.x * invRes;\n\
    float delta = 1.0 * invRes;\n\
    float dist = abs(gl_FragCoord.y - gl_FragCoord.x) * invRes;\n\
    float dir = sign(gl_FragCoord.y - gl_FragCoord.x);\n\
    float adjExp = adjFactor / abs(gl_FragCoord.y - gl_FragCoord.x);\n\
\n\
    vec4 color = vec4(0.0);\n\
    vec4 front = vec4(0.0);\n\
    vec4 back = vec4(0.0);\n\
\n\
    if (dist < 0.5 * delta)     // dist == 0\n\
    {\n\
        color = texture1D(tf, beginPos);\n\
        color.a = 1.0 - pow(1.0 - color.a, adjFactor);\n\
        color.rgb *= color.a;\n\
        front = color;\n\
        back = vec4(0.0);\n\
    }\n\
    else\n\
    {\n\
        for (float t = 0.0; t < dist - 0.5 * delta; t += delta)     // 0..dist-1\n\
        {\n\
            vec4 c = texture1D(tf, beginPos + t * dir);\n\
            c.a = 1.0 - pow(1.0 - c.a, adjExp);\n\
            c.a = (1.0 - color.a) * c.a;\n\
            c.rgb *= c.a;\n\
            color += c;\n\
            float b = t / dist;\n\
            front += (1.0 - b) * c;\n\
            back += b * c;\n\
        }\n\
    }\n\
\n\
    gl_FragData[0] = color;\n\
    gl_FragData[1] = front;\n\
    gl_FragData[2] = back;\n\
}\n\
";

PreIntegratorGL::PreIntegratorGL(int resolution, float stepSize, float baseStepSize)
    : _resolution(resolution),
      _stepSize(stepSize),
      _baseStepSize(baseStepSize)
{
    _shader = new GLShader();
    //_shader->loadVertexShader(QString("./plugins/DevRenderer/shaders/genPreInt.vert").toAscii().constData());
    //_shader->loadFragmentShader(QString("./plugins/DevRenderer/shaders/genPreInt.frag").toAscii().constData());
    _shader->setVertexShader(vertexShader, (GLint)strlen(vertexShader));
    _shader->setFragmentShader(fragmentShader, (GLint)strlen(fragmentShader));
    _shader->link();
    _shader->printVertexShaderInfoLog();
    _shader->printFragmentShaderInfoLog();

    _shader->addUniform1f("invRes", 1.0f / (float)_resolution);
    _shader->addUniform1f("adjFactor", _stepSize / _baseStepSize);
    _shader->addUniformSampler("tf", 0);

    _colorTable = new MSLib::GLTexture2D(GL_RGBA32F, _resolution, _resolution, 0, GL_RGBA, GL_FLOAT, 0);
    _frontTable = new MSLib::GLTexture2D(GL_RGBA32F, _resolution, _resolution, 0, GL_RGBA, GL_FLOAT, 0);
    _backTable = new MSLib::GLTexture2D(GL_RGBA32F, _resolution, _resolution, 0, GL_RGBA, GL_FLOAT, 0);

    _fbo = new MSLib::GLFramebufferObject();
    _fbo->bind();
    _fbo->attachTexture(*_colorTable, GL_COLOR_ATTACHMENT0);
    _fbo->attachTexture(*_frontTable, GL_COLOR_ATTACHMENT1);
    _fbo->attachTexture(*_backTable, GL_COLOR_ATTACHMENT2);
    if (!_fbo->checkStatus())
        qDebug("Error: fbo.checkStatus() == false");
    _fbo->release();
}

PreIntegratorGL::~PreIntegratorGL()
{
    delete _colorTable;
    delete _frontTable;
    delete _backTable;
    delete _fbo;
    delete _shader;
}

void PreIntegratorGL::update(MSLib::GLTexture1D &tfTex)
{
    _fbo->bind();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_TRANSFORM_BIT | GL_VIEWPORT_BIT);

    glDisable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glViewport(0, 0, _resolution, _resolution);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -10.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, buffers);

    tfTex.bind(0);

    _shader->setUniform1f("adjFactor", _stepSize / _baseStepSize);
    _shader->use();

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

    tfTex.release();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();

    _fbo->release();
}

void PreIntegratorGL::update(float *colorTable, float *frontTable, float *backTable, const float *tf, int tfSize)
{
    MSLib::GLTexture1D tfTex(GL_RGBA32F_ARB, tfSize, 0, GL_RGBA, GL_FLOAT, tf);
    update(tfTex);
    _colorTable->getImage(GL_RGBA, GL_FLOAT, colorTable);
    _frontTable->getImage(GL_RGBA, GL_FLOAT, frontTable);
    _backTable->getImage(GL_RGBA, GL_FLOAT, backTable);
}
