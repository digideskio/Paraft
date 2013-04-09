#include "MSGLFramebufferObject.h"

namespace MSLib
{

GLRenderbufferObject::GLRenderbufferObject(GLenum internalFormat, GLsizei width, GLsizei height)
{
    glGenRenderbuffers(1, &_handle);
    glBindRenderbuffer(GL_RENDERBUFFER, _handle);
    glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, width, height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

GLRenderbufferObject::~GLRenderbufferObject()
{
    glDeleteRenderbuffers(1, &_handle);
}

GLFramebufferObject::GLFramebufferObject()
    : _colorBuffer(nullptr),
      _depthBuffer(nullptr)
{
    glGenFramebuffers(1, &_handle);
}

GLFramebufferObject::~GLFramebufferObject()
{
    if (_colorBuffer != nullptr) delete _colorBuffer;
    if (_depthBuffer != nullptr) delete _depthBuffer;
    glDeleteFramebuffers(1, &_handle);
}

void GLFramebufferObject::bind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, _handle);
}

void GLFramebufferObject::release() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLFramebufferObject::attachTexture(GLuint textureId, GLenum attachment) const
{
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, textureId, 0 /* mipmap level */);
}

void GLFramebufferObject::attachRenderbuffer(GLuint renderbufferId, GLenum attachment) const
{
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, renderbufferId);
}

bool GLFramebufferObject::checkStatus() const
{
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    return (status == GL_FRAMEBUFFER_COMPLETE);
}

void GLFramebufferObject::attachTexture(const GLTexture2D &texture, GLenum attachment) const
{
    attachTexture(texture.handle(), attachment);
}

void GLFramebufferObject::attachColorTexture(const GLTexture2D &texture) const
{
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture->id(), 0 /* mipmap level */);
    attachTexture(texture, GL_COLOR_ATTACHMENT0);
}

void GLFramebufferObject::attachDepthTexture(const GLTexture2D &texture) const
{
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture->id(), 0 /* mipmap level */);
    attachTexture(texture, GL_DEPTH_ATTACHMENT);
}

void GLFramebufferObject::attachRenderbuffer(const GLRenderbufferObject &rbo, GLenum attachment) const
{
    attachRenderbuffer(rbo.handle(), attachment);
}

void GLFramebufferObject::attachColorBuffer(int width, int height)
{
    if (_colorBuffer != nullptr) delete _colorBuffer;
    _colorBuffer = new GLRenderbufferObject(GL_RGBA, width, height);
    attachRenderbuffer(*_colorBuffer, GL_COLOR_ATTACHMENT0);
}

void GLFramebufferObject::attachDepthBuffer(int width, int height)
{
    if (_depthBuffer != nullptr) delete _depthBuffer;
    _depthBuffer = new GLRenderbufferObject(GL_DEPTH_COMPONENT, width, height);
    attachRenderbuffer(*_depthBuffer, GL_DEPTH_ATTACHMENT);
}

void GLFramebufferObject::detachTexture(GLenum attachment) const
{
    attachTexture(0, attachment);
}

void GLFramebufferObject::detachColorTexture() const
{
    detachTexture(GL_COLOR_ATTACHMENT0);
}

void GLFramebufferObject::detachDepthTexture() const
{
    detachTexture(GL_DEPTH_ATTACHMENT);
}

void GLFramebufferObject::detachRenderbuffer(GLenum attachment) const
{
    attachRenderbuffer(0, attachment);
}

void GLFramebufferObject::detachColorBuffer()
{
    detachRenderbuffer(GL_COLOR_ATTACHMENT0);
    if (_colorBuffer != nullptr) delete _colorBuffer;
    _colorBuffer = nullptr;
}

void GLFramebufferObject::detachDepthBuffer()
{
    detachRenderbuffer(GL_DEPTH_ATTACHMENT);
    if (_depthBuffer != nullptr) delete _depthBuffer;
    _depthBuffer = nullptr;
}

GLuint GLFramebufferObject::currentBinding()
{
    GLint ret;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &ret);
    return (GLuint)ret;
}

} // namespace MSLib
