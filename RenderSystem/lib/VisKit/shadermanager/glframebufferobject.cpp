#include "glframebufferobject.h"
#include "gltexture.h"
#include <QtDebug>
#include "glbuffers.h"
#define DEBUG
#include "glerror.h"
#include "shader.h"

GLAbstractFBO* GLAbstractFBO::currentbound = 0;
bool GLFramebufferObject::resourcesinit = false;

GLAbstractFBO::GLAbstractFBO(int width, int height, GLenum depthtype):
	width(width), height(height), depthtype(depthtype) {

	GLERROR(glGenFramebuffersEXT(1, &handle));
	GLERROR(glGenRenderbuffersEXT(1, &depthhandle));
}

GLFramebufferObject::GLFramebufferObject(int width, int height, GLenum depthtype):
	GLAbstractFBO(width, height, depthtype),
	textures(33), img(0), data(0), premultiplied(true), vbo(0), shader(0) {

	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, handle));

	if(depthtype != GL_NONE) {
		GLERROR(glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthhandle));
		GLERROR(glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, depthtype, width, height));
		GLERROR(glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthhandle));
	}
	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

	for(int i = 0; i < 33; i++) {
		textures[i] = 0;
	}
}

void GLFramebufferObject::attachTexture(GLTexture* tex, GLenum attach, bool deleteCurrent) {
	int i = attach - GL_COLOR_ATTACHMENT0_EXT;
	bool wasbound = isBound();
	if(!wasbound)
		bind(false);
	if(textures[i] && tex != textures[i]) {
		if(!tex) {
			for(QVector<GLenum>::iterator it = buffers.begin(); it != buffers.end(); it++) {
				if((*it) == attach) {
					buffers.erase(it);
					break;
				}
			}
			GLERROR(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, attach, GL_TEXTURE_2D, 0, 0));
		}
		if(deleteCurrent)
			delete textures[i];
	}
	textures[i] = tex;
	if(tex) {
		if(attach != GL_DEPTH_ATTACHMENT_EXT && !buffers.contains(attach))
			buffers.push_back(attach);
		GLERROR(glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT, attach, tex->getHandle(), 0));
		//if(tex->getTarget() == GL_TEXTURE_CUBE_MAP) {
		//	for(int i = 0; i < 6; i++) {
		//		glFramebufferTextureFaceEXT(GL_FRAMEBUFFER_EXT, attach, tex->getHandle(), 0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i);
		//	}
		//}
	}
	checkStatus();
	if(!wasbound)
		release();
}

void GLFramebufferObject::detach(GLenum attach, bool deleteCurrent) {
	attachTexture(0, attach, deleteCurrent);
}

void GLFramebufferObject::attachTextureLayer(GLTexture* tex, int layer, GLenum attach, bool deleteCurrent) {
	int i = attach - GL_COLOR_ATTACHMENT0_EXT;
	bool wasbound = isBound();
	if(!wasbound)
		bind(false);
	if(textures[i] && tex != textures[i]) {
		if(!tex) {
			for(QVector<GLenum>::iterator it = buffers.begin(); it != buffers.end(); it++) {
				if((*it) == attach) {
					buffers.erase(it);
					break;
				}
			}
			GLERROR(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, attach, GL_TEXTURE_2D, 0, 0));
		}
		if(deleteCurrent)
			delete textures[i];
	}
	textures[i] = tex;
	if(tex) {
		if(attach != GL_DEPTH_ATTACHMENT_EXT && !buffers.contains(attach))
			buffers.push_back(attach);
		GLERROR(glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, attach, tex->getHandle(), 0, layer));
		//if(tex->getTarget() == GL_TEXTURE_CUBE_MAP) {
		//	for(int i = 0; i < 6; i++) {
		//		glFramebufferTextureFaceEXT(GL_FRAMEBUFFER_EXT, attach, tex->getHandle(), 0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i);
		//	}
		//}
	}
	checkStatus();
	if(!wasbound)
		release();
}

void GLFramebufferObject::bind(bool check, int drawbuffer) {
	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, handle));
	if(buffers.size() == 0) {
		GLERROR(glDrawBuffer(GL_NONE));
	}
	if(drawbuffer < 0 || drawbuffer >= buffers.size() && (buffers.size() > 1)) {
		GLERROR(glDrawBuffers(buffers.size(), buffers.data()));
	} else
		GLERROR(glDrawBuffer(buffers[drawbuffer]));

	if(check && buffers.size())
		checkStatus();
	currentbound = this;
}

void GLAbstractFBO::release() {
	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));
	currentbound = 0;
}

//bummed out of Qt
bool GLAbstractFBO::checkStatus() const
{
	GLERROR(GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT));
	switch(status) {
	case GL_NO_ERROR:
	case GL_FRAMEBUFFER_COMPLETE_EXT:
		return true;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		qDebug("GLFramebufferObject: Unsupported framebuffer format.");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
		qDebug("GLFramebufferObject: Framebuffer incomplete attachment.");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
		qDebug("GLFramebufferObject: Framebuffer incomplete, missing attachment.");
		break;
#ifdef GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT
	case GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT:
		qDebug("QGLFramebufferObject: Framebuffer incomplete, duplicate attachment.");
		break;
#endif
	case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		qDebug("GLFramebufferObject: Framebuffer incomplete, attached images must have same dimensions.");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
		qDebug("GLFramebufferObject: Framebuffer incomplete, attached images must have same format.");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
		qDebug("GLFramebufferObject: Framebuffer incomplete, missing draw buffer.");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
		qDebug("GLFramebufferObject: Framebuffer incomplete, missing read buffer.");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS_EXT:
		qDebug("GLFramebufferObject: Framebuffer incomplete, missing layer targets.");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_LAYER_COUNT_EXT:
		qDebug("GLFramebufferObject: Framebuffer incomplete, missing layer count.");
		break;
	default:
		qDebug() <<"QGLFramebufferObject: An undefined error has occurred: "<< status;
		break;
	}
	return false;
}

GLTexture* GLFramebufferObject::operator[](int index) {
	if(index < 0 || index >= 33) {
		return 0;
	}
	return textures[index];
}

void GLFramebufferObject::resize(int w, int h) {
	width = w;
	height = h;
	bool wasbound = isBound();

	if(!wasbound)
		bind(false);

	for(int i = 0; i < 33; i++) {
		if(textures[i]) {
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT + i, textures[i]->getTarget(), 0, 0);
		}
	}

	release();

	glDeleteFramebuffersEXT(1, &handle);
	glDeleteRenderbuffersEXT(1, &depthhandle);


	glGenFramebuffersEXT(1, &handle);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, handle);

	if(depthtype != GL_NONE) {
		glGenRenderbuffersEXT(1, &depthhandle);
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthhandle);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, depthtype, width, height);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthhandle);
	}

	for(int i = 0; i < 33; i++) {
		if(textures[i] && ((textures[i]->getTarget() == GL_TEXTURE_2D) ||
			(textures[i]->getTarget() == GL_TEXTURE_CUBE_MAP))) {
			((GLTexture2D*)textures[i])->resize(w, h);
			attachTexture(textures[i], GL_COLOR_ATTACHMENT0_EXT + i);
		}
	}

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	if(wasbound)
		bind();

	if(img)
		delete img;
	img = 0;

	if(data)
		delete [] data;
	data = 0;
}

bool GLAbstractFBO::isBound() const {
	return currentbound == this;
}

GLAbstractFBO::~GLAbstractFBO() {
	if(isBound())
		release();
	glDeleteFramebuffersEXT(1, &handle);
	glDeleteRenderbuffersEXT(1, &depthhandle);

}

GLFramebufferObject::~GLFramebufferObject() {
	if(isBound())
		release();
	for(int i = 0; i < 33; i++) {
		if(textures[i])
			delete textures[i];
	}
	if(shader) delete shader;
	if(vbo) delete vbo;
}

QImage& GLFramebufferObject::toImage() {
	if(!data)
		data = new unsigned char[width*height*4];
	bool wasbound = isBound();
	if(!wasbound)
		bind();
	glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, reinterpret_cast<GLvoid*>(data));
	if(!wasbound)
		release();
	if(!img)
		img = new QImage(data, width, height, premultiplied ? QImage::Format_ARGB32_Premultiplied : QImage::Format_ARGB32);

	mirrored = img->mirrored(false, true);
	return mirrored;
}

void GLFramebufferObject::setPremultiplied(bool b) {
	if(b != premultiplied) {
		if(img)
			delete img;
		premultiplied = b;
	}
}

GLAbstractFBO* GLAbstractFBO::current() {
	return GLAbstractFBO::currentbound;
}


GLMultiSampleFramebufferObject::GLMultiSampleFramebufferObject(int width, int height, GLenum depthtype, int samples):
	GLAbstractFBO(width, height, depthtype), samples(samples) {

	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, handle));
	GLERROR(glGenRenderbuffersEXT(1, &colorbufferhandle));
	GLERROR(glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, colorbufferhandle));
	GLERROR(glRenderbufferStorageMultisampleEXT(GL_RENDERBUFFER_EXT, samples, GL_RGBA, width, height));
	GLERROR(glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, colorbufferhandle));

	if(depthtype != GL_NONE) {
		GLERROR(glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthhandle));
		GLERROR(glRenderbufferStorageMultisampleEXT(GL_RENDERBUFFER_EXT, samples, depthtype, width, height));
		GLERROR(glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthhandle));
	}
	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));


}
void GLMultiSampleFramebufferObject::bind(bool check,int) {
	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, handle));

	if(check)
		checkStatus();

	currentbound = this;
}


void GLMultiSampleFramebufferObject::resize(int w, int h) {
	width = w;
	height = h;
	bool wasbound = isBound();


	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, handle));

	GLERROR(glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, colorbufferhandle));
	GLERROR(glRenderbufferStorageMultisampleEXT(GL_RENDERBUFFER_EXT, samples, GL_RGBA, width, height));
	GLERROR(glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, colorbufferhandle));

	if(depthtype != GL_NONE) {
		GLERROR(glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthhandle));
		GLERROR(glRenderbufferStorageMultisampleEXT(GL_RENDERBUFFER_EXT, samples, depthtype, width, height));
		GLERROR(glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthhandle));
	}

	GLERROR(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

	if(wasbound)
		bind();
}

void GLAbstractFBO::blit(GLAbstractFBO* other, GLenum buffers) {
	glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, handle);
	glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, other ? other->getHandle() : 0);
	glBlitFramebufferEXT(0, 0, width, height, 0, 0, width, height, buffers, GL_NEAREST);
	glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, 0);
	glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, 0);
}

void GLFramebufferObject::draw(int attachment, bool usebuiltinshader) {

	if(!resourcesinit) {
		Q_INIT_RESOURCE(fbo);
		resourcesinit = true;
	}

	if(!GLTexture::isAutoBind()) {
		qWarning("FBO Built-in draw only works if textures are automatically handled");
		return;
	}
	if(attachment < 0 || attachment > textures.size() || !textures[attachment]) {
		qWarning("No attachment at %d!", attachment);
	}
	if(!vbo) {
		vbo = new GLVertexbufferf(GL_QUADS, GL_STATIC_DRAW);
		(*vbo) << Vector3(-1, -1, 0) << Vector3(1, -1, 0) << Vector3(1, 1, 0) << Vector3(-1, 1, 0);
	}
	if(!shader && usebuiltinshader) {
		shader = new Shader("fboshader");
		shader->addFragmentShader(":/fbo/shaders/fbo.frag");
		shader->addVertexShader(":/fbo/shaders/fbo.vert");
		shader->addUniformSampler("fbo",textures[attachment]);
		shader->compileAndLink();
	}
	if(usebuiltinshader) {
		shader->getSamplers()["fbo"]->setTexture(textures[attachment]);
		shader->use();
	}
	vbo->draw();
	if(usebuiltinshader) {
		shader->release();
	}
}
