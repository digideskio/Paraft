#include <cstdlib>

#include "GLBuffer.h"

namespace MSLib
{

GLBuffer::GLBuffer(GLenum target, GLenum usage)
    : _target(target),
      _usage(usage)
{
    glGenBuffers(1, &_handle);
}

GLBuffer::~GLBuffer()
{
    glDeleteBuffers(1, &_handle);
}

void GLBuffer::allocate(const void *data, GLsizei size)
{
    glBufferData(_target, size, data, _usage);
}

void GLBuffer::allocate(GLsizei size)
{
    glBufferData(_target, size, 0, _usage);
}

void GLBuffer::bind()
{
    glBindBuffer(_target, _handle);
}

void *GLBuffer::map(GLenum access)
{
    return glMapBufferARB(_target, access);
}

void GLBuffer::read(GLint offset, void *data, GLsizei size)
{
    glGetBufferSubData(_target, offset, size, data);
}

void GLBuffer::release()
{
    glBindBuffer(_target, 0);
}

GLboolean GLBuffer::unmap()
{
    return glUnmapBufferARB(_target);
}

void GLBuffer::write(GLint offset, const void *data, GLsizei size)
{
    glBufferSubData(_target, offset, size, data);
}

GLVertexBuffer::GLVertexBuffer(GLenum usage)
    : GLBuffer(GL_ARRAY_BUFFER, usage)
{

}

GLIndexBuffer::GLIndexBuffer(GLenum usage)
    : GLBuffer(GL_ELEMENT_ARRAY_BUFFER, usage)
{

}

GLPixelBuffer::GLPixelBuffer(GLenum usage)
    : GLBuffer(GL_PIXEL_PACK_BUFFER, usage)
{

}

void GLPixelBuffer::bindPack()
{
    _target = GL_PIXEL_PACK_BUFFER;
    bind();
}

void GLPixelBuffer::bindUnpack()
{
    _target = GL_PIXEL_UNPACK_BUFFER;
    bind();
}

} // namespace MSLib
