#ifndef GLBUFFER_H
#define GLBUFFER_H

#include <GL/glew.h>

namespace MSLib
{

class GLBuffer
{
public:
    enum Usage
    {
        STATIC_DRAW  = GL_STATIC_DRAW,
        STATIC_READ  = GL_STATIC_READ,
        STATIC_COPY  = GL_STATIC_COPY,
        DYNAMIC_DRAW = GL_DYNAMIC_DRAW,
        DYNAMIC_READ = GL_DYNAMIC_READ,
        DYNAMIC_COPY = GL_DYNAMIC_COPY,
        STREAM_DRAW  = GL_STREAM_DRAW,
        STREAM_READ  = GL_STREAM_READ,
        STREAM_COPY  = GL_STREAM_COPY
    };

    GLBuffer(GLenum target, GLenum usage);
    virtual ~GLBuffer();

    void      allocate(const void *data, GLsizei size);
    void      allocate(GLsizei size);
    void      bind();
    GLuint    handle() const { return _handle; }
    void    * map(GLenum access = GL_READ_WRITE);
    void      read(GLint offset, void *data, GLsizei size);
    void      release();
    GLboolean unmap();
    void      write(GLint offset, const void *data, GLsizei size);

protected:
    GLuint _handle;
    GLenum _target;
    GLenum _usage;
};

class GLVertexBuffer : public GLBuffer
{
public:
    GLVertexBuffer(GLenum usage = GL_STATIC_DRAW);
};

class GLIndexBuffer : public GLBuffer
{
public:
    GLIndexBuffer(GLenum usage = GL_STATIC_DRAW);
};

class GLPixelBuffer : public GLBuffer
{
public:
    GLPixelBuffer(GLenum usage = GL_DYNAMIC_DRAW);
    void bindPack();
    void bindUnpack();
};

} // namespace MSLib

#endif // GLBUFFER_H
