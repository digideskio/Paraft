#include <cstdio>
#include <iostream>
#include <fstream>
#include "GLShader.h"

namespace MSLib
{

GLShader::GLShader()
{
    _vertShaderId = glCreateShader(GL_VERTEX_SHADER);
    _fragShaderId = glCreateShader(GL_FRAGMENT_SHADER);
    _programId = glCreateProgram();
    _vertShaderAttached = false;    ////
    _fragShaderAttached = false;    ////
}

GLShader::~GLShader()
{
    if (_vertShaderAttached)        ////
        glDetachShader(_programId, _vertShaderId);
    if (_fragShaderAttached)        ////
        glDetachShader(_programId, _fragShaderId);
    glDeleteShader(_vertShaderId);
    glDeleteShader(_fragShaderId);
    glDeleteProgram(_programId);

    for (Hash<String, GLAbstractUniform *>::iterator it = _uniforms.begin(); it != _uniforms.end(); it++)
    {
        delete *it;
    }
}

void GLShader::setVertexShader(const GLchar *string, GLint length)
{
    glShaderSource(_vertShaderId, 1 /* numOfStrings */, &string, &length);
    glCompileShader(_vertShaderId);
    glAttachShader(_programId, _vertShaderId);
}

void GLShader::setFragmentShader(const GLchar *string, GLint length)
{
    glShaderSource(_fragShaderId, 1 /* numOfStrings */, &string, &length);
    glCompileShader(_fragShaderId);
    glAttachShader(_programId, _fragShaderId);
}

void GLShader::link()
{
    glLinkProgram(_programId);
}

void GLShader::use()
{
    glUseProgram(_programId);
    setUniforms();              ////
}

void GLShader::useFixed()
{
    glUseProgram(0);
}

GLint GLShader::getAttribLocation(const GLchar *name)
{
    return glGetAttribLocation(_programId, name);
}

GLint GLShader::getUniformLocation(const GLchar *name)
{
    return glGetUniformLocation(_programId, name);
}

void GLShader::setUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2)
{
    glUniform3f(location, v0, v1, v2);
}

void GLShader::setUniform1i(GLint location, GLint v0)
{
    glUniform1i(location, v0);
}

////
void GLShader::printShaderInfoLog(GLuint obj)
{
    int infoLogLength = 0;
    int charsWritten = 0;
    char *infoLog;

    glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0)
    {
        infoLog = new char[infoLogLength];
        glGetShaderInfoLog(obj, infoLogLength, &charsWritten, infoLog);
        printf("%s\n", infoLog);
        delete [] infoLog;
    }
}

void GLShader::printVertexShaderInfoLog()
{
    printShaderInfoLog(_vertShaderId);
}

void GLShader::printFragmentShaderInfoLog()
{
    printShaderInfoLog(_fragShaderId);
}

////
void GLShader::printProgramInfoLog()
{
    GLuint obj = _programId;
    int infoLogLength = 0;
    int charsWritten = 0;
    char *infoLog;

    glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0)
    {
        infoLog = new char[infoLogLength];
        glGetProgramInfoLog(obj, infoLogLength, &charsWritten, infoLog);
        printf("%s\n", infoLog);
        delete [] infoLog;
    }
}

bool GLShader::loadVertexShader(const char *fileName)
{
    _vertShaderFileName = String(fileName);
    return _loadVertexShader();
}

bool GLShader::loadFragmentShader(const char *fileName)
{
    _fragShaderFileName = String(fileName);
    return _loadFragmentShader();
}

bool GLShader::reload()
{
    if (!_loadVertexShader()) return false;
    printVertexShaderInfoLog();
    if (!_loadFragmentShader()) return false;
    printFragmentShaderInfoLog();
    link();
    for (Hash<String, GLAbstractUniform *>::iterator it = _uniforms.begin(); it != _uniforms.end(); ++it)
    {
        (*it)->setLocation(getUniformLocation((*it)->name().c_str()));
    }
    printProgramInfoLog();
    return true;
}

void GLShader::setUniforms()
{
    for (Hash<String, GLAbstractUniform *>::iterator it = _uniforms.begin(); it != _uniforms.end(); ++it)
    {
        (*it)->set();
    }
}

bool GLShader::_loadVertexShader()
{
    std::ifstream ifs;
    ifs.open(_vertShaderFileName.c_str(), std::ios::in | std::ios::binary);
    if (ifs.fail())
    {
        std::cout << "Error: in GLShader::loadVertexShader(): Cannot open the file \"" << _vertShaderFileName << "\"" << std::endl;
        return false;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    char *buffer = new char[length];
    ifs.read(buffer, length);
    ifs.close();
    setVertexShader(buffer, length);
    delete [] buffer;
    return true;
}

bool GLShader::_loadFragmentShader()
{
    std::ifstream ifs;
    ifs.open(_fragShaderFileName.c_str(), std::ios::in | std::ios::binary);
    if (ifs.fail())
    {
        std::cout << "Error: in GLShader::loadVertexShader(): Cannot open the file \"" << _fragShaderFileName << "\"" << std::endl;
        return false;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    char *buffer = new char[length];
    ifs.read(buffer, length);
    ifs.close();
    setFragmentShader(buffer, length);
    delete [] buffer;
    return true;
}

} // namespace MSLib
