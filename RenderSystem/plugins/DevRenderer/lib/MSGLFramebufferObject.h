#ifndef GLFRAMEBUFFEROBJECT_H
#define GLFRAMEBUFFEROBJECT_H

#include <GL/glew.h>

#include "MSGLTexture.h"

#define nullptr 0

namespace MSLib     // avoid conflict with VisKit
{

class GLRenderbufferObject
{
public:
    GLRenderbufferObject(GLenum internalFormat, GLsizei width, GLsizei height);
    ~GLRenderbufferObject();
    GLuint handle() const { return _handle; }

protected:
    GLuint _handle;
};

class GLFramebufferObject
{
public:
    GLFramebufferObject();
    ~GLFramebufferObject();

    // low-level
    GLuint handle() const { return _handle; }
    void bind() const;
    void release() const;
    void attachTexture(GLuint textureId, GLenum attachment) const;
    void attachRenderbuffer(GLuint renderbufferId, GLenum attachment) const;
    bool checkStatus() const;
    // bool isBound() const;

    // high-level
    void attachTexture(const GLTexture2D &texture, GLenum attachment) const;
    void attachColorTexture(const GLTexture2D &texture) const;
    void attachDepthTexture(const GLTexture2D &texture) const;
    void attachRenderbuffer(const GLRenderbufferObject &rbo, GLenum attachment) const;
    void attachColorBuffer(int width, int height);
    void attachDepthBuffer(int width, int height);
    void detachTexture(GLenum attachment) const;
    void detachColorTexture() const;
    void detachDepthTexture() const;
    void detachRenderbuffer(GLenum attachment) const;
    void detachColorBuffer();
    void detachDepthBuffer();

    // test
    static GLuint currentBinding();

protected:
    GLuint _handle;
    GLRenderbufferObject *_colorBuffer;
    GLRenderbufferObject *_depthBuffer;
};

} // namespace MSLib

#endif // GLFRAMEBUFFEROBJECT_H
