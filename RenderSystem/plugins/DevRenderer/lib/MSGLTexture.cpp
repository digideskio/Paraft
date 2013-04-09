//#include "GLTexture.h"
#include "MSGLTexture.h"

namespace MSLib // avoid conflict with VisKit
{

GLTexture::GLTexture(GLenum target, GLint internalFormat, GLint border, GLenum format, GLenum type)
    : _target(target),
      _internalFormat(internalFormat),
      _border(border),
      _format(format),
      _type(type),
      _boundUnit(-1)
{
    glGenTextures(1, &_handle);
}

GLTexture::~GLTexture()
{
    glDeleteTextures(1, &_handle);
}

void GLTexture::bind()
{
    glBindTexture(_target, _handle);
    _boundUnit = getActiveTexture() - GL_TEXTURE0;
}

void GLTexture::bind(int textureUnit) {
    GLint currentActive = getActiveTexture();
    setActiveTexture(GL_TEXTURE0 + textureUnit);
    bind();
    setActiveTexture(currentActive);    // resume previous active unit
}

void GLTexture::bindDefault() {
    glBindTexture(_target, 0);
    _boundUnit = -1;
}

void GLTexture::release() {
    if (_boundUnit < 0) return;
    GLint currentActive = getActiveTexture();
    setActiveTexture(GL_TEXTURE0 + _boundUnit);
    //glBindTexture(_target, 0);
    bindDefault();
    setActiveTexture(currentActive);
}

void GLTexture::setParameteri(GLenum pname, GLint param) {
    bool wasBound = isBound();
    if (!wasBound)
        bind();
    glTexParameteri(_target, pname, param);
    if (!wasBound)
        release();
}

void GLTexture::getImage(GLenum format, GLenum type, GLvoid *img) {
    bind(); ////
    glGetTexImage(_target, 0 /* mipmap level */, format, type, img);
    release();   ////
}

GLint GLTexture::getActiveTexture() {
    GLint ret;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &ret);
    return ret;
}

void GLTexture::setActiveTexture(GLenum texture) {
    glActiveTexture(texture);
}

GLTexture1D::GLTexture1D(GLint internalFormat, GLsizei width, GLint border, GLenum format, GLenum type, const GLvoid *data)
    : GLTexture(GL_TEXTURE_1D, internalFormat, border, format, type),
      _width(width) {
    glBindTexture(_target, _handle);
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(_target, 0 /* mipmap level */, _internalFormat, _width, _border, _format, _type, data);
    glBindTexture(_target, 0);
}

void GLTexture1D::load(const GLvoid *data)
{
    bind();
    glTexSubImage1D(_target, 0 /* mipmap level */, 0 /* x offset */, _width, _format, _type, data);
    release();
}

GLTexture2D::GLTexture2D(GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *data)
    : GLTexture(GL_TEXTURE_2D, internalFormat, border, format, type),
      _width(width),
      _height(height)
{
    glBindTexture(_target, _handle);
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(_target, 0 /* mipmap level */, _internalFormat, _width, _height, _border, _format, _type, data);
    glBindTexture(_target, 0);
}

void GLTexture2D::load(const GLvoid *data)
{
    bind();
    glTexSubImage2D(_target, 0 /* mipmap level */, 0 /* x offset */, 0 /* y offset */, _width, _height, _format, _type, data);
    release();
}

GLTexture3D::GLTexture3D(GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *data, GLenum magFilter, GLenum minFilter)
    : GLTexture(GL_TEXTURE_3D, internalFormat, border, format, type),
      _width(width),
      _height(height),
      _depth(depth) {
    glBindTexture(_target, _handle);
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(_target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexImage3D(_target, 0 /* mipmap level */, _internalFormat, _width, _height, _depth, _border, _format, _type, data);
    glBindTexture(_target, 0);
}

void GLTexture3D::load(const GLvoid *data)
{
    bind();
    glTexSubImage3D(_target, 0 /* mipmap level */, 0 /* x offset */, 0 /* y offset */, 0 /* z offset */, _width, _height, _depth, _format, _type, data);
    release();
}

} // namespace MyLib
