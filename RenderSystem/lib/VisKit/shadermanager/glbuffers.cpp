#include <GL/glew.h>
#include <cstring>
#include "glbuffers.h"
#include "shader.h"
#include "glerror.h"

GLIndexbuffer* GLIndexbuffer::currentbound = 0;
GLVertexbuffer* GLVertexbuffer::currentbound = 0;

GLAttribute::GLAttribute(const QString& name, int size, GLsizei stride, size_t offset, bool normalized)
:name(name), stride(stride), offset(offset), size(size), normalized(normalized) {
}


GLBuffer::GLBuffer(GLenum target, GLenum usage):target(target), count(0), usage(usage), mapped(false) {
	glGenBuffers(1, &id);
}

void GLBuffer::reset() {
	count = 0;
}

GLBuffer::~GLBuffer() {
	glDeleteBuffers(1, &id);
}

void GLBuffer::bind() {
	glBindBuffer(target, id);
}

void GLBuffer::release() {
	glBindBuffer(target, 0);
}

GLIndexbuffer::GLIndexbuffer(GLenum type, GLenum usage)
	:GLBuffer(GL_ELEMENT_ARRAY_BUFFER, usage), type(type), datasize(0), qdata(0), restart(0), useRestart(false) {}

GLIndexbuffer& GLIndexbuffer::operator<<(const unsigned int& rhs) {
	queue.push_back(rhs);
	return *this;
}

void GLIndexbuffer::bind() {
	if(currentbound == this)
		return;
	currentbound = this;
	GLBuffer::bind();
	update();
	if(useRestart) {
		glEnable(GL_PRIMITIVE_RESTART);
		glPrimitiveRestartIndex(restart);
	}
}

bool GLIndexbuffer::isBound() {
	return currentbound == this;
}

void GLIndexbuffer::update() {
	if(queue.isEmpty() && !qdata)
		return;
	GLIndexbuffer* lastbound = currentbound;
	bind();
	unsigned int* t = new unsigned int[count + queue.size() + datasize];
	if(count) {
		unsigned int* m = map();
		memcpy(t, m, count*4);
		unmap();
	}
	memcpy(t + count, queue.data(), queue.size()*4);
	if(qdata) {
		memcpy(t + count + queue.size(), qdata, datasize*4);
	}
	glBufferData(target, count*4 + queue.size()*4 + datasize*4, t, usage);
	if(lastbound)
		lastbound->bind();
	else
		release();
	delete [] t;
	count += static_cast<int>(queue.size() + datasize);
	if(qdata) {
		delete [] qdata;
		qdata = 0;
		datasize = 0;
	}
	queue.clear();
}

void GLIndexbuffer::release() {
	if(!isBound())
		return;
	currentbound = 0;
	GLBuffer::release();
	if(useRestart) {
		glDisable(GL_PRIMITIVE_RESTART);
	}
}

unsigned int* GLIndexbuffer::map(GLenum access) {
	if(!isBound())
		bind();
	mapped = true;
	return reinterpret_cast<unsigned int*>(glMapBuffer(target, access));
}

bool GLIndexbuffer::unmap() {
	if(!isBound() || !mapped)
		return false;
	mapped = false;
	return glUnmapBuffer(target);
}

GLVertexbuffer::GLVertexbuffer(GLenum type, GLenum usage)
:GLBuffer(GL_ARRAY_BUFFER, usage), drawType(type),
hNormals(false), hTexCoords(false), hColors(false), hVertices(true),
normalStride(0), texCoordStride(0), colorStride(0), vertexStride(0),
normalOffset(0), texCoordOffset(0), colorOffset(0), vertexOffset(0),
texCoordSize(3), vertexSize(3), colorSize(3) {}

bool GLVertexbuffer::isBound() {
	return currentbound == this;
}

bool GLVertexbuffer::unmap() {
	if(!isBound() || !mapped)
		return false;
	mapped = false;
	return glUnmapBuffer(target);
}

void GLVertexbuffer::bind(bool setpointers) {
	if(isBound())
		return;
	currentbound = this;
	GLBuffer::bind();
	update();

	if(setpointers)
		setPointers();
}

void GLVertexbuffer::release(bool releasePointers) {
	if(!isBound())
		return;
	currentbound = 0;

	GLBuffer::release();

	if(releasePointers)
		cleanPointers();

}

void GLVertexbuffer::cleanPointers() {
	for(QHash<QString,GLAttribute*>::iterator it = attribs.begin(); it != attribs.end(); ++it) {
		it.value()->release();
	}
	if(hNormals)
		glDisableClientState(GL_NORMAL_ARRAY);
	if(hColors)
		glDisableClientState(GL_COLOR_ARRAY);
	if(hTexCoords)
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	if(hVertices)
		glDisableClientState(GL_VERTEX_ARRAY);
}

GLVertexbuffer& GLVertexbuffer::operator <<(const Vector3& rhs) {
	addVector3(rhs);
	return * this;
}

GLVertexbuffer& GLVertexbuffer::operator <<(const Vector4& rhs) {
	addVector4(rhs);
	return * this;
}

GLVertexbuffer& GLVertexbuffer::operator <<(float v) {
	addFloat(v);
	return * this;
}

void GLVertexbufferf::setPointers() {
	for(QHash<QString,GLAttribute*>::iterator it = attribs.begin(); it != attribs.end(); ++it) {
		it.value()->bind();
	}
	if(hNormals) {
		glNormalPointer(GL_FLOAT,
			static_cast<GLsizei>(normalStride),
			reinterpret_cast<const GLvoid*>(normalOffset));
		glEnableClientState(GL_NORMAL_ARRAY);
	}
	if(hTexCoords) {
		glTexCoordPointer(texCoordSize,
			GL_FLOAT,
			static_cast<GLsizei>(texCoordStride),
			reinterpret_cast<const GLvoid*>(texCoordOffset));
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	}
	if(hColors) {
		glColorPointer(colorSize,
			GL_FLOAT,
			static_cast<GLsizei>(colorStride),
			reinterpret_cast<const GLvoid*>(colorOffset));
		glEnableClientState(GL_COLOR_ARRAY);
	}
	if(hVertices) {
		glVertexPointer(vertexSize,
			GL_FLOAT,
			static_cast<GLsizei>(vertexStride),
			reinterpret_cast<const GLvoid*>(vertexOffset));
		glEnableClientState(GL_VERTEX_ARRAY);
	}
}

void GLVertexbufferd::setPointers() {
	if(hNormals) {
		glNormalPointer(GL_DOUBLE,
			static_cast<GLsizei>(normalStride),
			reinterpret_cast<const GLvoid*>(normalOffset));
		glEnableClientState(GL_NORMAL_ARRAY);
	}
	if(hTexCoords) {
		glTexCoordPointer(texCoordSize,
			GL_DOUBLE,
			static_cast<GLsizei>(texCoordStride),
			reinterpret_cast<const GLvoid*>(texCoordOffset));
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	}
	if(hColors) {
		glColorPointer(colorSize,
			GL_DOUBLE,
			static_cast<GLsizei>(colorStride),
			reinterpret_cast<const GLvoid*>(colorOffset));
		glEnableClientState(GL_COLOR_ARRAY);
	}
	if(hVertices) {
		glVertexPointer(vertexSize,
			GL_DOUBLE,
			static_cast<GLsizei>(vertexStride),
			reinterpret_cast<const GLvoid*>(vertexOffset));
		glEnableClientState(GL_VERTEX_ARRAY);
	}
}

double* GLVertexbufferd::map(GLenum access) {
	if(!isBound())
		bind();
	mapped = true;
	return reinterpret_cast<double*>(glMapBuffer(target, access));
}

float* GLVertexbufferf::map(GLenum access) {
	if(!isBound())
		bind();
	mapped = true;
	return reinterpret_cast<float*>(glMapBuffer(target, access));
}


void GLVertexbufferd::update() {
	if(queue.isEmpty())
		return;
	GLVertexbuffer* lastbound = currentbound;
	bind();
	double* t = new double[count + queue.size()];
	if(count) {
		double* m = map();
		memcpy(t, m, count*8);
		unmap();
	}
	memcpy(t + count, queue.data(), queue.size()*8);
	glBufferData(target, (count + queue.size())*8, t, usage);
	if(lastbound)
		lastbound->bind();
	else
		release();
	count += queue.size();
	if(qdata) {
		delete [] qdata;
		qdata = 0;
		datasize = 0;
	}
	delete [] t;
	queue.clear();
}

void GLVertexbufferf::update() {
	if(queue.isEmpty() && !qdata)
		return;
	GLVertexbuffer* lastbound = currentbound;
	bind();
	float* t = new float[count + queue.size()];
	if(count) {
		float* m = map();
		memcpy(t, m, count*4);
		unmap();
	}
	memcpy(t + count, queue.data(), queue.size()*4);
	glBufferData(target, (count + queue.size())*4, t, usage);
	if(lastbound)
		lastbound->bind();
	else
		release();
	count += queue.size();
	delete [] t;
	queue.clear();
}

void GLVertexbuffer::draw(int first, int num) {
	GLVertexbuffer* lastbound = currentbound;
	bind();
	int i = (hVertices ? vertexSize : 0) + (hTexCoords ? texCoordSize : 0) + (hColors ? colorSize : 0) + (hNormals ? 3 : 0);
	for(QHash<QString, GLAttribute*>::iterator it = attribs.begin(); it != attribs.end(); ++it) {
			i += it.value()->size;
	}
	glDrawArrays(drawType, first, static_cast<GLsizei>(num ? num : count/i - first));

	if(lastbound)
		lastbound->bind();
	else
		release();
}

void GLVertexbuffer::drawInstanced(int instances, int first, int num) {
	GLVertexbuffer* lastbound = currentbound;
	bind();
	int i = (hVertices ? vertexSize : 0) + (hTexCoords ? texCoordSize : 0) + (hColors ? colorSize : 0) + (hNormals ? 3 : 0);
	for(QHash<QString, GLAttribute*>::iterator it = attribs.begin(); it != attribs.end(); ++it) {
			i += it.value()->size;
	}
	glDrawArraysInstanced(drawType, first, static_cast<GLsizei>(num ? num : count/i - first), instances);

	if(lastbound)
		lastbound->bind();
	else
		release();
}

void GLIndexbuffer::draw() {
	GLIndexbuffer* lastbound = currentbound;
	bind();
	glDrawElements(type, count, GL_UNSIGNED_INT, 0);
	if(lastbound)
		lastbound->bind();
	else
		release();
}


void GLIndexbuffer::drawInstanced(int instances) {
	GLIndexbuffer* lastbound = currentbound;
	bind();
	glDrawElementsInstanced(type, count, GL_UNSIGNED_INT, 0, instances);
	if(lastbound)
		lastbound->bind();
	else
		release();
}

GLVertexbufferd::GLVertexbufferd(GLenum type, GLenum usage):GLVertexbuffer(type, usage), qdata(0), datasize(0) {}
GLVertexbufferf::GLVertexbufferf(GLenum type, GLenum usage):GLVertexbuffer(type, usage), qdata(0), datasize(0) {}

void GLVertexbuffer::reset() {
//	queue.clear();
	count = 0;
	for(QHash<QString,GLAttribute*>::iterator it = attribs.begin(); it != attribs.end(); ++it) {
		delete *it;
	}
	attribs.clear();

	hVertices = true;
	hTexCoords = false;
	hColors = false;
	hNormals = false;

	normalStride = 0;
	texCoordStride = 0;
	colorStride = 0;
	vertexStride = 0;

	normalOffset = 0;
	texCoordOffset = 0;
	colorOffset = 0;
	vertexOffset = 0;
}

void GLVertexbufferd::setData(double* data, size_t size) {
	reset();
	queue.clear();
	for(size_t i = 0; i < size; ++i) {
		queue.push_back(data[i]);
	}
}

void GLVertexbufferf::setData(float* data, size_t size) {
	reset();
	queue.clear();
	for(size_t i = 0; i < size; ++i) {
		queue.push_back(data[i]);
	}
}

void GLIndexbuffer::setData(unsigned int* data, size_t size) {
	reset();
	queue.clear();
	for(size_t i = 0; i < size; ++i) {
		queue.push_back(data[i]);
	}
}

void GLVertexbufferf::save(QIODevice& file, size_t forceCount) {
	if(!file.isWritable())
		return;
	GLVertexbuffer* lastbound = currentbound;

	
	if(forceCount <= 0)
		forceCount = count;

	float* t = map();
	int zero = 0;
	file.write((char*)&zero, 4);
	file.write((char*)&forceCount, 4);
	file.write((char*)t, forceCount*4);
	unmap();

	QDataStream ds(&file);
	ds << hNormals << hTexCoords << hColors << hVertices;
	ds << normalStride << normalOffset
		<< colorStride << colorOffset
		<< texCoordStride << texCoordOffset
		<< vertexStride << vertexOffset;
	ds << texCoordSize << vertexSize << colorSize;

	ds << attribs.count();
	for(QHash<QString, GLAttribute*>::iterator it = attribs.begin(); it != attribs.end(); ++it) {
		ds << *it.value();
	}

	if(lastbound)
		lastbound->bind();
	else
		release();
}

void GLVertexbufferf::load(QIODevice& file) {
	if(!file.isReadable())
		return;
	int s;
	file.read((char*)&s, 4);
	if(s) {
		float* t = new float[s];
		file.read((char*)t, s*4);
		setData(t, s);
		delete [] t;
	} else {
		file.read((char*)&s, 4);
		float* t = new float[s];
		file.read((char*)t, s*4);
		setData(t, s);
		delete [] t;
		QDataStream ds(&file);
		ds >> hNormals >> hTexCoords >> hColors >> hVertices;
		ds >> normalStride >> normalOffset
			>> colorStride >> colorOffset
			>> texCoordStride >> texCoordOffset
			>> vertexStride >> vertexOffset;
		ds >> texCoordSize >> vertexSize >> colorSize;

		int attribcount;
		ds >> attribcount;
		for(int i = 0; i < attribcount; ++i) {
			GLAttribute* at = new GLAttribute();
			ds >> *at;
			attribs[at->name] = at;
		}
	}
}

void GLVertexbufferd::save(QIODevice& file, size_t forceCount) {
	if(!file.isWritable())
		return;
	GLVertexbuffer* lastbound = currentbound;
	
	if(forceCount <= 0)
		forceCount = count;

	double* t = map();
	file.write((char*)&forceCount, 4);
	file.write((char*)t, forceCount*8);
	unmap();

	if(lastbound)
		lastbound->bind();
	else
		release();
}

void GLVertexbufferd::load(QIODevice& file) {
	if(!file.isReadable())
		return;
	int s;
	file.read((char*)&s, 4);
	double* t = new double[s];
	file.read((char*)t, s*8);
	setData(t, s);
	delete [] t;
}

void GLIndexbuffer::save(QIODevice& file, size_t forceCount) {
	if(!file.isWritable())
		return;

	if(forceCount <= 0)
		forceCount = count;

	GLIndexbuffer* lastbound = currentbound;

	unsigned int* t = map();
	file.write((char*)&forceCount, 4);
	file.write((char*)t, forceCount*4);
	unmap();

	if(lastbound)
		lastbound->bind();
	else
		release();
}

void GLIndexbuffer::load(QIODevice& file) {
	if(!file.isReadable())
		return;
	int s;
	file.read((char*)&s, 4);
	unsigned int* t = new unsigned int[s];
	file.read((char*)t, s*4);
	setData(t, s);
	delete [] t;
}
void GLVertexbufferf::addFloat(float v) {
	queue.push_back(v);
}

void GLVertexbufferf::addVector3(const Vector3& p) {
	for(int i = 0; i < 3; ++i) {
		queue.push_back(p.elements[i]);
	}
}
void GLVertexbufferf::addVector4(const Vector4& p) {
	for(int i = 0; i < 4; ++i) {
		queue.push_back(p.elements[i]);
	}
}
void GLVertexbufferd::addVector3(const Vector3& p) {
	for(int i = 0; i < 3; ++i) {
		queue.push_back(p.elements[i]);
	}
}
void GLVertexbufferd::addVector4(const Vector4& p) {
	for(int i = 0; i < 4; ++i) {
		queue.push_back(p.elements[i]);
	}
}

void GLPixelbuffer::bind(GLenum t) {
	target = t;
	GLBuffer::bind();
}

GLPixelbuffer::GLPixelbuffer(int size, GLenum usage):GLBuffer(GL_ARRAY_BUFFER, usage) {
	GLBuffer::bind();
	glBufferData(GL_ARRAY_BUFFER, size, 0, usage);
	release();
}

void GLPixelbuffer::resize(int size) {
	GLenum t = target;
	bind(GL_PIXEL_PACK_BUFFER);
	glBufferData(GL_PIXEL_PACK_BUFFER, size, 0, usage);
	release();
	target = t;
}


void GLAttribute::bind() {
	int index;
	if(!Shader::current() ||
	   ((index = Shader::current()->getAttribLocation(name.toAscii().data())) == -1)) {
		return;
	}

	glEnableVertexAttribArray(index);
	glVertexAttribPointer(index, size, GL_FLOAT, normalized, stride, (GLvoid*)offset);

}

void GLAttribute::release() {
	int index;
	if(!Shader::current() ||
	   ((index = Shader::current()->getAttribLocation(name.toAscii().data())) == -1)) {
		return;
	}
	glDisableVertexAttribArray(index);
}

void GLVertexbuffer::addAttribute(const QString& name, int size, GLsizei stride, size_t offset, bool normalized) {
	if(attribs.contains(name))
		return;
	attribs[name] = new GLAttribute(name, size, stride, offset, normalized);
}

GLAttribute* GLVertexbuffer::operator[](const QString& index) {
	if(attribs.contains(index))
		return attribs[index];
	return 0;
}

bool GLPixelbuffer::unmap() {
	return glUnmapBuffer(target);
}

void* GLPixelbuffer::map(GLenum usage) {
	GLERROR(void* t = glMapBuffer(target,usage));
	return t;
}

QDataStream& operator<<(QDataStream& lhs, const GLAttribute& rhs) {
	return lhs << rhs.name
			<< (quint32)rhs.stride
			<< (quint64)rhs.offset
			<< rhs.size
			<< rhs.normalized;
}

QDataStream& operator>>(QDataStream& lhs, GLAttribute& rhs) {
	quint64 t64;
	quint32 t32;
	lhs >> rhs.name >> t32 >> t64 >> rhs.size >> rhs.normalized;
	rhs.stride = t32;
	rhs.offset = t64;
	return lhs;
}

QDataStream& operator<<(QDataStream& lhs, GLBuffer& rhs) {
	rhs.save(*lhs.device());
	return lhs;
}

QDataStream& operator>>(QDataStream& lhs, GLBuffer& rhs) {
	rhs.load(*lhs.device());
	return lhs;
}
