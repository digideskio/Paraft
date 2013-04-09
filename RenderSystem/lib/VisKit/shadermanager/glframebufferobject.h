#ifndef _GLFRAMEBUFFEROBJECT_H_
#define _GLFRAMEBUFFEROBJECT_H_

#include <GL/glew.h>
#include <QVector>
#include <QImage>

class GLTexture;
class GLVertexbufferf;
class GLAbstractFBO {
protected:
	int width;
	int height;
	GLenum depthtype;
	GLuint handle;
	GLuint depthhandle;
	static GLAbstractFBO* currentbound;
public:
	virtual void resize(int w, int h)=0;
	virtual void bind(bool check=true, int drawbuffer=-1)=0;
	virtual void release();
	virtual bool checkStatus() const;
	virtual bool isBound() const;
	void blit(GLAbstractFBO* other, GLenum buffers = (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

	GLAbstractFBO(int width, int height, GLenum depthtype);
	virtual ~GLAbstractFBO();
	static GLAbstractFBO* current();

	GLuint getHandle() const { return handle; }

};
class GLVertexbufferf;
class Shader;
class GLFramebufferObject : public GLAbstractFBO {
	QVector<GLTexture*> textures;
	QVector<GLenum> buffers;
	QImage* img;
	QImage mirrored;
	unsigned char* data;
	bool premultiplied;
	GLVertexbufferf* vbo;
	Shader* shader;
	static bool resourcesinit;
public:
	GLFramebufferObject(int width, int height, GLenum depthtype);
	void resize(int w, int h);
	void bind(bool check=true, int drawbuffer=-1);
	void detach(GLenum attach, bool deleteCurrent=true);
	void attachTexture(GLTexture* tex, GLenum attach=GL_COLOR_ATTACHMENT0_EXT, bool deleteCurrent=true);
	void attachTextureLayer(GLTexture* tex, int layer, GLenum attach=GL_COLOR_ATTACHMENT0_EXT, bool deleteCurrent=true);
	void setPremultiplied(bool b);
	QImage& toImage();
	~GLFramebufferObject();
	GLTexture* operator[](int index);
	void draw(int attachment=0, bool usebuiltinshader=true);
};


class GLMultiSampleFramebufferObject : public GLAbstractFBO {
	GLuint colorbufferhandle;
	int samples;
public:
	GLMultiSampleFramebufferObject(int width, int height, GLenum depthtype, int samples);
	void resize(int w, int h);
	void bind(bool check=true, int drawbuffer=-1);
};

#endif
