#include "gltexture.h"
#include "texturemanager.h"
#define DEBUG
#include "glerror.h"
#include "glbuffers.h"

bool GLTexture::autobind = false;

GLTexture::GLTexture(GLenum target, GLint internalFormat, int width,
					 int border, GLenum format, GLenum datatype,
					 GLenum minfilter, GLenum magfilter, GLenum wrap_s)
:target(target), internalFormat(internalFormat), width(width), border(border), format(format), isbound(false), datatype(datatype), texslot(-1), uploaded(false) {
	GLERROR(glGenTextures(1, &handle));
	params[GL_TEXTURE_MIN_FILTER] = minfilter;
	params[GL_TEXTURE_MAG_FILTER] = magfilter;
	params[GL_TEXTURE_WRAP_S] = wrap_s;
}


GLTexture::~GLTexture() {
	if(TextureManager::getInstance()) {
		TextureManager::getInstance()->remove(this);
	}
	emit deleted();
	GLERROR(glDeleteTextures(1, &handle));
}

void GLTexture::managerBind() {
	GLERROR(glBindTexture(target, handle));
	isbound = true;
}

GLTexture1D::~GLTexture1D() {
	if(TextureManager::getInstance()) {
		TextureManager::getInstance()->removeFromMemPool(getSize());
	}
}

GLTexture2D::~GLTexture2D() {
	if(TextureManager::getInstance()) {
		TextureManager::getInstance()->removeFromMemPool(getSize());
	}
}

GLTexture3D::~GLTexture3D() {
	if(TextureManager::getInstance()) {
		TextureManager::getInstance()->removeFromMemPool(getSize());
	}
}

GLTexture1D::GLTexture1D(GLint internalFormat, int width,
					int border, GLenum format, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum datatype)
:GLTexture(GL_TEXTURE_1D, internalFormat, width, border, format, datatype, minfilter, magfilter, wrap_s) {
	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
	//GLERROR(glDisable(target));
}

void GLTexture::bind(int active) {
	if(active < 0) {
		if(!autobind) {
			glGetIntegerv(GL_ACTIVE_TEXTURE, &active);
			active -= GL_TEXTURE0;
		} else {
			TextureManager::getInstance()->bind(this);
		}
	}
	if(active >= 0 && active != texslot) {
		TextureManager::getInstance()->bindToSlot(active, this);
	}
	for(QHash<GLenum, GLenum>::iterator it = params.begin(); it != params.end(); ++it) {
		//qDebug("%x %x %x",target, it.key(), it.value());
		GLERROR(glTexParameteri(target, it.key(), it.value()));
	}
}

void GLTexture1D::upload(void *data) {
	bind();
	GLERROR(glTexImage1D(target, 0, internalFormat, width, border, format, datatype, data));
	uploaded = true;
}


void GLTexture1D::loadNew(GLint internalFormat, int width,
					int border, GLenum format, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum datatype) {
	if(uploaded)
		TextureManager::getInstance()->removeFromMemPool(getSize());
	this->internalFormat = internalFormat;
	this->width = width;
	this->border = border;
	this->format = format;
	this->datatype = datatype;
	params[GL_TEXTURE_MIN_FILTER] = minfilter;
	params[GL_TEXTURE_MAG_FILTER] = magfilter;
	params[GL_TEXTURE_WRAP_S] = wrap_s;

	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
	//GLERROR(glDisable(target));
}

void GLTexture1D::loadNew(GLint internalFormat, int width,
					int border, GLenum format, GLenum datatype, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s) {
	loadNew(internalFormat, width, border, format, data, minfilter, magfilter, wrap_s, datatype);
}

GLTexture2D::GLTexture2D(GLint internalFormat, int width, int height,
					int border, GLenum format, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t, GLenum datatype)
:GLTexture(GL_TEXTURE_2D, internalFormat, width, border, format, datatype, minfilter, magfilter, wrap_s), height(height) {
	params[GL_TEXTURE_WRAP_T] = wrap_t;
	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
	//GLERROR(glDisable(target));
}

GLTexture2D::GLTexture2D(GLenum target, GLint internalFormat, int width, int height,
					int border, GLenum format, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t, GLenum datatype)
:GLTexture(target, internalFormat, width, border, format, datatype, minfilter, magfilter, wrap_s), height(height) {
	params[GL_TEXTURE_WRAP_T] = wrap_t;
	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
	//GLERROR(glDisable(target));
}

void GLTexture2D::upload(void *data) {
	bind();
	if(target == GL_TEXTURE_CUBE_MAP) {
		for(int i = 0; i < 6; i++) {
			GLERROR(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internalFormat, width, height, border, format, datatype, 0));
		}
	} else {
		GLERROR(glTexImage2D(target, 0, internalFormat, width, height, border, format, datatype, data));
	}
	uploaded = true;
}


void GLTexture2D::loadNew(GLint internalFormat, int width, int height,
					int border, GLenum format, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t, GLenum datatype) {
	if(uploaded)
		TextureManager::getInstance()->removeFromMemPool(getSize());
	this->internalFormat = internalFormat;
	this->width = width;
	this->height = height;
	this->border = border;
	this->format = format;
	this->datatype = datatype;
	params[GL_TEXTURE_MIN_FILTER] = minfilter;
	params[GL_TEXTURE_MAG_FILTER] = magfilter;
	params[GL_TEXTURE_WRAP_S] = wrap_s;
	params[GL_TEXTURE_WRAP_T] = wrap_t;

	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
}

void GLTexture2D::loadNew(GLint internalFormat, int width, int height,
					int border, GLenum format, GLenum datatype, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t) {
	loadNew(internalFormat, width, height, border, format, data, minfilter, magfilter, wrap_s, wrap_t, datatype);
}

GLTexture3D::GLTexture3D(GLint internalFormat, int width, int height, int depth,
					int border, GLenum format, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t, GLenum wrap_r, GLenum datatype)
:GLTexture(GL_TEXTURE_3D, internalFormat, width, border, format, datatype, minfilter, magfilter, wrap_s), height(height), depth(depth) {
	params[GL_TEXTURE_WRAP_T] = wrap_t;
	params[GL_TEXTURE_WRAP_R] = wrap_r;
	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
}

GLTexture3D::GLTexture3D(GLint internalFormat, int width, int height, int depth,
					int border, GLenum format, GLenum datatype, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t, GLenum wrap_r)
:GLTexture(GL_TEXTURE_3D, internalFormat, width, border, format, datatype, minfilter, magfilter, wrap_s), height(height), depth(depth) {
	params[GL_TEXTURE_WRAP_T] = wrap_t;
	params[GL_TEXTURE_WRAP_R] = wrap_r;
	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
}


void GLTexture3D::upload(void *data) {
	bind();
	GLERROR(glTexImage3D(target, 0, internalFormat, width, height, depth, border, format, datatype, data));
	uploaded = true;
}

void GLTexture3D::loadNew(GLint internalFormat, int width, int height, int depth,
					int border, GLenum format, GLenum datatype, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t, GLenum wrap_r) {
	loadNew(internalFormat, width, height, depth, border, format, data, minfilter, magfilter, wrap_s, wrap_t, wrap_r, datatype);
}

void GLTexture3D::loadNew(GLint internalFormat, int width, int height, int depth,
					int border, GLenum format, void* data,
					GLenum minfilter, GLenum magfilter,
					GLenum wrap_s, GLenum wrap_t, GLenum wrap_r, GLenum datatype) {
	if(uploaded)
		TextureManager::getInstance()->removeFromMemPool(getSize());
	this->internalFormat = internalFormat;
	this->width = width;
	this->height = height;
	this->depth = depth;
	this->border = border;
	this->format = format;
	this->datatype = datatype;
	params[GL_TEXTURE_MIN_FILTER] = minfilter;
	params[GL_TEXTURE_MAG_FILTER] = magfilter;
	params[GL_TEXTURE_WRAP_S] = wrap_s;
	params[GL_TEXTURE_WRAP_T] = wrap_t;
	params[GL_TEXTURE_WRAP_R] = wrap_r;

	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
}

void GLTexture3D::reload(void* data, int active, GLenum newdatatype) {
	if(newdatatype != GL_NONE)
		datatype = newdatatype;
	bind(active);
	GLERROR(glTexSubImage3D(target, 0, 0, 0, 0, width, height, depth, format, datatype, data));
}

void GLTexture2D::reload(void* data, int active, GLenum newdatatype) {
	if(newdatatype != GL_NONE)
		datatype = newdatatype;
	bind(active);
	GLERROR(glTexSubImage2D(target, 0, 0, 0, width, height, format, datatype, data));
}

void GLTexture1D::reload(void* data, int active, GLenum newdatatype) {
	if(newdatatype != GL_NONE)
		datatype = newdatatype;
	bind(active);
	GLERROR(glTexSubImage1D(target, 0, 0, width, format, datatype, data));
}

void GLTexture3D::resize(int w, int h, int d, void* data) {
	TextureManager::getInstance()->removeFromMemPool(getSize());
	width = w;
	height = h;
	depth = d;

	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
}

void GLTexture2D::resize(int w, int h, void* data) {
	TextureManager::getInstance()->removeFromMemPool(getSize());
	width = w;
	height = h;

	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
}

void GLTexture1D::resize(int w, void* data) {
	TextureManager::getInstance()->removeFromMemPool(getSize());
	width = w;

	upload(data);
	TextureManager::getInstance()->addToMemPool(getSize());
}

size_t GLTexture::getInternalFormatSize() const {
	switch(internalFormat) {
	case GL_RGBA32F:
	case GL_RGBA32I:
	case GL_RGBA32UI:
		return 16;
	case GL_RG32F:
	case GL_RG32UI:
	case GL_RG32I:
		return 8;
	case GL_LUMINANCE32F_ARB:
	case GL_ALPHA32F_ARB:
	case GL_DEPTH_COMPONENT32:
	case GL_DEPTH_COMPONENT32F:
	case GL_R32I:
	case GL_R32UI:
	case GL_RGBA:
		return 4;
	case GL_LUMINANCE_ALPHA:
	case GL_RG:
		return 2;
	default:
		return 1;
	}
}

size_t GLTexture1D::getSize() const {
	return (size_t)width*getInternalFormatSize();
}
size_t GLTexture2D::getSize() const {
	return (size_t)width*(size_t)height*getInternalFormatSize()*(size_t)(target == GL_TEXTURE_CUBE_MAP ? 6 : 1);
}
size_t GLTexture3D::getSize() const {
	return (size_t)width*(size_t)height*(size_t)depth*getInternalFormatSize();
}


GLTextureBuffer::GLTextureBuffer(GLVertexbufferf* vbo, GLenum internalFormat)
	:GLTexture(GL_TEXTURE_BUFFER, internalFormat, 0, 0, GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_NONE), vbo(vbo) {
	params.clear(); //buffer textures have no parameters
	setBuffer(vbo, internalFormat);
}

void GLTextureBuffer::setBuffer(GLVertexbufferf* new_vbo, GLenum new_internalFormat) {
	vbo = new_vbo;
	internalFormat = new_internalFormat;
	bool bound = isbound;
	if(!bound)
		bind();
	glTexBuffer(target, internalFormat, vbo ? vbo->handle() : 0);
}
