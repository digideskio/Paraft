#ifndef _GLTEXTURE_H_
#define _GLTEXTURE_H_

#include <GL/glew.h>
#include <QHash>
#include <QObject>

class GLTexture : public QObject {
	Q_OBJECT
protected:
	GLuint handle;

	GLenum target;
	GLint internalFormat;
	int width;
	int border;
	GLenum format;
	bool isbound;
	GLenum datatype;
	int texslot;

	QHash<GLenum, GLenum> params;

	friend class TextureManager;

	virtual void managerBind();
	static bool autobind;
	size_t getInternalFormatSize() const;
	bool uploaded;

public:
	GLTexture(GLenum target, GLint internalFormat, int width,
			  int border, GLenum format, GLenum datatype,
			  GLenum minfilter, GLenum magfilter, GLenum wrap_s);
	virtual void bind(int active=-1);
	virtual ~GLTexture();
	virtual void reload(void* data, int active=-1, GLenum newdatatype=GL_NONE)=0;
	int getWidth() const { return width; }
	int getBorder() const { return border; }
	GLenum getFormat() const { return format; }
	GLenum getInternalFormat() const { return internalFormat; }
	GLenum getMinFilter() const { return params[GL_TEXTURE_MIN_FILTER]; }
	GLenum getMagFilter() const { return params[GL_TEXTURE_MAG_FILTER]; }
	GLenum getWrapS() const { return params[GL_TEXTURE_WRAP_S]; }
	GLuint getHandle() const { return handle; }
	GLenum getTarget() const { return target; }
	void setBound(bool bound) { isbound = bound; }
	void setSlot(int slot) { texslot = slot; }
	bool isBound() const { return isbound; }
	int getSlot() const {
		if(!isbound)
			return -1;
		return texslot;
	}
	static void setAutoBind(bool ab) { autobind = ab; }
	static bool isAutoBind() { return autobind; }
	virtual size_t getSize() const=0;
signals:
	void deleted();
};

class GLVertexbufferf;
class GLTextureBuffer : public GLTexture {
	Q_OBJECT
protected:
	GLVertexbufferf* vbo;
public:
	GLTextureBuffer(GLVertexbufferf* vbo, GLenum internalFormat);
	void setBuffer(GLVertexbufferf* vbo, GLenum internalFormat);
	virtual ~GLTextureBuffer() {}
	virtual size_t getSize() const { return 0; }
	virtual void reload(void*, int=-1, GLenum=GL_NONE) {}
};

class GLTexture1D : public GLTexture {
	Q_OBJECT
	public:
		GLTexture1D(GLint internalFormat, int width,
					int border, GLenum format, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum datatype=GL_FLOAT);
		~GLTexture1D();
		void upload(void* data);
		void reload(void* data, int active=-1, GLenum newdatatype=GL_NONE);
		void loadNew(GLint internalFormat, int width,
					int border, GLenum format, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum datatype=GL_FLOAT);
		void loadNew(GLint internalFormat, int width,
					int border, GLenum format, GLenum datatype, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP);
		void resize(int w, void* data=0);
		size_t getSize() const;
};

class GLTexture2D : public GLTexture {
	Q_OBJECT
	int height;
	public:
		GLTexture2D(GLint internalFormat, int width, int height,
					int border, GLenum format, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum wrap_t=GL_CLAMP, GLenum datatype=GL_FLOAT);
		GLTexture2D(GLenum target, GLint internalFormat, int width, int height,
					int border, GLenum format, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum wrap_t=GL_CLAMP, GLenum datatype=GL_FLOAT);
		~GLTexture2D();
		void upload(void* data);
		void reload(void* data, int active=-1, GLenum newdatatype=GL_NONE);
		void loadNew(GLint internalFormat, int width, int height,
					int border, GLenum format, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum wrap_t=GL_CLAMP, GLenum datatype=GL_FLOAT);
		void loadNew(GLint internalFormat, int width, int height,
					int border, GLenum format, GLenum datatype, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum wrap_t=GL_CLAMP);
		int getHeight() const { return height; }
		GLenum getWrapT() const { return params[GL_TEXTURE_WRAP_T]; }
		void resize(int w, int h, void* data=0);
		size_t getSize() const;
};

class GLTexture3D : public GLTexture {
	Q_OBJECT
	int height, depth;
	public:
		GLTexture3D(GLint internalFormat, int width, int height, int depth,
					int border, GLenum format, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum wrap_t=GL_CLAMP, GLenum wrap_r=GL_CLAMP, GLenum datatype=GL_FLOAT);
		GLTexture3D(GLint internalFormat, int width, int height, int depth,
					int border, GLenum format, GLenum datatype, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t, GLenum wrap_r);
		~GLTexture3D();
		void upload(void* data);
		void reload(void* data, int active=-1, GLenum newdatatype=GL_NONE);
		void loadNew(GLint internalFormat, int width, int height, int depth,
					int border, GLenum format, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum wrap_t=GL_CLAMP, GLenum wrap_r=GL_CLAMP, GLenum datatype=GL_FLOAT);
		void loadNew(GLint internalFormat, int width, int height, int depth,
					int border, GLenum format, GLenum datatype, void* data,
					GLenum minfilter=GL_LINEAR, GLenum magfilter=GL_LINEAR,
					GLenum wrap_s=GL_CLAMP, GLenum wrap_t=GL_CLAMP, GLenum wrap_r=GL_CLAMP);
		int getHeight() const { return height; }
		int getDepth() const { return depth; }
		GLenum getWrapT() const { return params[GL_TEXTURE_WRAP_T]; }
		GLenum getWrapR() const { return params[GL_TEXTURE_WRAP_R]; }
		void resize(int w, int h, int d, void* data=0);
		size_t getSize() const;
};



#endif

