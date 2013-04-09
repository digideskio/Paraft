#ifndef GLTEXTURE_H
#define GLTEXTURE_H

#include <GL/glew.h>

namespace MSLib     // avoid conflict with VisKit
{

class GLTexture
{
public:
    GLTexture(GLenum target, GLint internalFormat, GLint border, GLenum format, GLenum type);
    virtual ~GLTexture();

    GLuint handle() const { return _handle; }
    void   bind();
    void   bind(int textureUnit);     // 0 for GL_TEXTURE0; will not affect current active unit
    void   bindDefault();
    void   release();
    bool   isBound() const { return (_boundUnit >= 0); }
    int    boundUnit() const { return _boundUnit; }
    void   setParameteri(GLenum pname, GLint param);
    void   getImage(GLenum format, GLenum type, GLvoid *img);

    static GLint getActiveTexture();                // GL_TEXTURE0, ...
    static void setActiveTexture(GLenum texture);   // GL_TEXTURE0, ...

    virtual void load(const GLvoid *data) = 0;

protected:
    GLuint _handle;
    GLenum _target;
    GLint  _internalFormat;
    GLint  _border;
    GLenum _format;         // format of the pixel data (GL_RGBA, ...)
    GLenum _type;           // data type (GL_FLOAT, ...)
    GLint  _boundUnit;      // 0 for GL_TEXTURE0; may not work if the texture was bound without using bind(), unbind()
};

class GLTexture1D : public GLTexture
{
public:
    GLTexture1D(GLint internalFormat, GLsizei width, GLint border, GLenum format, GLenum type, const GLvoid *data);
    //~GLTexture1D();
    void load(const GLvoid *data);

    int width() const { return _width; }

protected:
    GLsizei _width;
};

class GLTexture2D : public GLTexture
{
public:
    GLTexture2D(GLint internalFormat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const GLvoid *data);
    //~GLTexture2D();
    //GLuint id() const { return _id; }
    //void bind();
    //void unbind();
    void load(const GLvoid *data);

    int width() const { return _width; }
    int height() const { return _height; }

protected:
    //GLuint _id;
    //GLuint _boundUnitId;
    GLsizei _width;
    GLsizei _height;
};

class GLTexture3D : public GLTexture
{
public:
    GLTexture3D(GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *data, GLenum magFilter = GL_LINEAR, GLenum minFilter = GL_LINEAR);
    //~GLTexture3D();
    void load(const GLvoid *data);

    int width() const { return _width; }
    int height() const { return _height; }
    int depth() const { return _depth; }

protected:
    GLsizei _width;
    GLsizei _height;
    GLsizei _depth;
};

} // namespace MSLib

#endif // GLTEXTURE_H
