#ifndef _GLBUFFER_H_
#define _GLBUFFER_H_

#include <GL/glew.h>
#include "../camera/vectors.h"
#include <QVector>
#include <QIODevice>
#include <QHash>
#include <QDataStream>

struct GLAttribute {
	GLAttribute(const QString& name=QString(), int size=3, GLsizei stride=0, size_t offset=0, bool normalized=false);

	QString name;
	GLsizei stride;
	size_t offset;
	int size;
	bool normalized;

	void bind();
	void release();
	void save(QIODevice& ios);
	void load(QIODevice& ios);
};


class GLBuffer {
protected:
	GLuint id;
	GLenum target;
	size_t count;
	GLenum usage;
	bool mapped;
public:
	GLBuffer(GLenum target, GLenum usage);
	virtual ~GLBuffer();
	virtual void bind();
	virtual void release();
	virtual void update()=0;
	virtual bool isBound()=0;
	virtual void reset();
	GLuint handle() const { return id; }

	virtual void save(QIODevice& ios, size_t forceCount=0)=0;
	virtual void load(QIODevice& ios)=0;
};


QDataStream& operator<<(QDataStream& lhs, const GLAttribute& rhs);
QDataStream& operator<<(QDataStream& lhs, GLBuffer& rhs);

QDataStream& operator>>(QDataStream& lhs, GLAttribute& rhs);
QDataStream& operator>>(QDataStream& lhs, GLBuffer& rhs);

class GLIndexbuffer : public GLBuffer {
	QVector<unsigned int> queue;
	static GLIndexbuffer* currentbound;
	GLenum type;
	size_t datasize;
	unsigned int* qdata;
	unsigned int restart;
	bool useRestart;
public:
	GLIndexbuffer(GLenum type, GLenum usage=GL_STATIC_DRAW);
	GLIndexbuffer& operator<<(const unsigned int& rhs);
	void bind();
	void release();
	void update();
	void draw();
	void drawInstanced(int instances);
	unsigned int* map(GLenum access=GL_READ_ONLY);
	bool unmap();

	virtual void save(QIODevice& file, size_t forceCount=0);
	virtual void load(QIODevice& file);

	void setData(unsigned int* data, size_t size);
	bool isBound();
	bool usesRestart() const { return useRestart; }
	void setRestartIndex(int index) { restart = index; }
	void setUseRestart(bool v) { useRestart = v; }
};

#ifndef _USE_OPENGL_3
struct VertexStride {

};
#endif

class GLVertexbuffer : public GLBuffer {
protected:
	GLenum drawType;
	static GLVertexbuffer* currentbound;
	bool hNormals;
	bool hTexCoords;
	bool hColors;
	bool hVertices;
	unsigned int normalStride;
	unsigned int texCoordStride;
	unsigned int colorStride;
	unsigned int vertexStride;
	unsigned int normalOffset;
	unsigned int texCoordOffset;
	unsigned int colorOffset;
	unsigned int vertexOffset;

	int texCoordSize;
	int vertexSize;
	int colorSize;
	QHash<QString, GLAttribute*> attribs;

	virtual void addVector3(const Vector3&)=0;
	virtual void addVector4(const Vector4&)=0;
	virtual void addFloat(float)=0;
public:
	virtual void setPointers()=0;
	virtual void reset();
	GLVertexbuffer(GLenum type, GLenum usage);
	GLVertexbuffer& operator<<(const Vector3& rhs);
	GLVertexbuffer& operator<<(const Vector4& rhs);
	GLVertexbuffer& operator<<(float rhs);
	virtual void save(QIODevice& file, size_t forceCount=0)=0;
	virtual void load(QIODevice& file)=0;
	bool unmap();
	void bind(bool setpointers=true);
	void release(bool cleanpointers=true);
	void cleanPointers();
	void draw(int first=0, int count=0);
	void drawInstanced(int instances, int first=0, int count=0);
	virtual void update()=0;
	void addAttribute(const QString& name, int size=3, GLsizei stride=0, size_t offset=0, bool normalized=false);
	GLAttribute* operator[](const QString& index);
#ifndef _USE_OPENGL_3
	bool hasNormals() const { return hNormals; }
	bool hasTexCoords() const { return hTexCoords; }
	bool hasColors() const { return hColors; }
	bool hasVertices() const { return hVertices; }

	void setHasColors(bool b) { hColors = b; }
	void setHasTexCoords(bool b) { hTexCoords = b; }
	void setHasVertices(bool b) { hVertices = b; }
	void setHasNormals(bool b) { hNormals = b; }

	unsigned int getNormalStride() const { return normalStride; }
	unsigned int getTexCoordStride() const { return texCoordStride; }
	unsigned int getColorStride() const { return colorStride; }
	unsigned int getVertexStride() const { return vertexStride; }

	unsigned int getNormalOffset() const { return normalOffset; }
	unsigned int getTexCoordOffset() const { return texCoordOffset; }
	unsigned int getColorOffset() const { return colorOffset; }
	unsigned int getVertexOffset() const { return vertexOffset; }

	void setNormalStride(unsigned int i) { normalStride = i; }
	void setColorStride(unsigned int i) { colorStride = i; }
	void setTexCoordStride(unsigned int i) { texCoordStride = i; }
	void setVertexStride(unsigned int i) { vertexStride = i; }

	void setNormalOffset(unsigned int i) { normalOffset = i; }
	void setColorOffset(unsigned int i) { colorOffset = i; }
	void setTexCoordOffset(unsigned int i) { texCoordOffset = i; }
	void setVertexOffset(unsigned int i) { vertexOffset = i; }

	int getTexCoordSize() const { return texCoordSize; }
	int getVertexSize() const { return vertexSize; }
	int getColorSize() const { return colorSize; }

	void setColorSize(int i) { colorSize = i; }
	void setTexCoordSize(int i) { texCoordSize = i; }
	void setVertexSize(int i) { vertexSize = i; }
#endif
	bool isBound();
};

class GLVertexbufferf : public GLVertexbuffer {
	QVector<float> queue;
	float* qdata;
	size_t datasize;
protected:
	void addVector3(const Vector3&);
	void addVector4(const Vector4&);
	void addFloat(float v);
public:
	void setPointers();
	void update();
	GLVertexbufferf(GLenum type, GLenum usage=GL_STATIC_DRAW);
	float* map(GLenum access=GL_READ_ONLY);

	void setData(float* data, size_t size);
	virtual void save(QIODevice& file, size_t forceCount=0);
	virtual void load(QIODevice& file);
};

class GLVertexbufferd : public GLVertexbuffer {
	QVector<double> queue;
	double* qdata;
	size_t datasize;
protected:
	void addVector3(const Vector3&);
	void addVector4(const Vector4&);
	void addFloat(float) {}
public:
	void setPointers();
	void update();
	GLVertexbufferd(GLenum type, GLenum usage=GL_STATIC_DRAW);
	double* map(GLenum access=GL_READ_ONLY);
	void setData(double* data, size_t size);
	virtual void save(QIODevice& file, size_t forceCount=0);
	virtual void load(QIODevice& file);
};

class GLPixelbuffer : public GLBuffer {
public:
	GLPixelbuffer(int size, GLenum usage=GL_DYNAMIC_DRAW);
	bool isBound() { return false; }
	void bind(GLenum target);
	void resize(int size);
	void update() {}
	void* map(GLenum usage=GL_READ_WRITE);
	bool unmap();
	virtual void save(QIODevice& file, size_t forceCount=0) {}
	virtual void load(QIODevice& file) {}
};

#endif
