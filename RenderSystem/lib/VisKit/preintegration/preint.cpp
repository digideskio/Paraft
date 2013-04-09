#include "preint.h"
#include "shader.h"

#include "glbuffers.h"
#include "gltexture.h"
#include "glframebufferobject.h"

#include "QTFEditor.h"
#include "shadermanager.h"
#include <QDir>
static bool resourcesinit=false;
#define GLDEBUG(x) \
x; \
for(GLenum e = glGetError(); e != GL_NO_ERROR; e = glGetError()) { \
	qWarning("Error at line number %d, in file %s. glGetError() returned %s for call %s\n",__LINE__, __FILE__, gluErrorString(e), #x ); \
}


Preintegrator::Preintegrator(QTFEditor* qtfe, QObject* parent):QObject(parent),
uspecular(0), udiffuse(0), usteps(0), ubasesteps(0), udeltascale(0),
needsupdate(true), qtfe(qtfe), tfwidth(qtfe->getTFColorMapResolution()) {}

Preintegrator::Preintegrator(const GLenum *texslot, QTFEditor* qtfe, int, float basesteps, float steps, QObject* parent):QObject(parent),
uspecular(0), udiffuse(0), usteps(0), ubasesteps(0), udeltascale(0),
needsupdate(true), qtfe(qtfe), tfwidth(qtfe->getTFColorMapResolution()) {
	for(int i = 0; i < 4; ++i) {
		texslots[i] = texslot[i];
	}

	if(!resourcesinit) {
		Q_INIT_RESOURCE(shaders);
		resourcesinit = true;
	}

	preintshader = new Shader("preint");
	preintshader->addFragmentShader(":/preint.frag");
	preintshader->addVertexShader(":/preint.vert");

	preintshader->compileAndLink();

	preintshader->addUniformf("basesteps", 1, basesteps);
	preintshader->addUniformf("steps", 1, steps);
	preintshader->addUniformf("diffuse", 4, 1.0, 1.0, 1.0, 0.6);
	preintshader->addUniformf("specular", 4, 1.0, 1.0, 1.0, 0.2);
	preintshader->addUniformf("distscale", 1, 1.);
	preintshader->addUniformf("deltascale", 1, 1.);

	preintshader->addUniformf("width", 1, (double)tfwidth);
	preintshader->addUniformi("tf", 1, texslots[0] - GL_TEXTURE0);

	fbo = new GLFramebufferObject(tfwidth, tfwidth, GL_NONE);
	for(int i = 0; i < 3; ++i) {
		GLDEBUG(glActiveTexture(texslots[i + 1]));
		GLDEBUG(fbo->attachTexture(new GLTexture2D(GL_TEXTURE_2D, GL_RGBA32F_ARB, tfwidth, tfwidth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP), GL_COLOR_ATTACHMENT0_EXT + i));
		GLDEBUG((*fbo)[i]->bind());
		//GLDEBUG(glDisable(GL_TEXTURE_2D));
	}
	/*
	GLDEBUG(GLTexture2D *temptex = new GLTexture2D(GL_TEXTURE_2D, GL_RGBA32F_ARB, tfwidth, tfwidth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP));
	GLDEBUG(glActiveTexture(texslots[3]));
	GLDEBUG(fbo->attachTexture(temptex, GL_COLOR_ATTACHMENT2_EXT)); */

	//GLDEBUG(glDisable(GL_TEXTURE_2D));

	fbo->checkStatus();
	vbo = new GLVertexbufferf(GL_TRIANGLE_STRIP, GL_STATIC_DRAW);
	(*vbo) << Vector3(-1, -1, 0) << Vector3(1, -1, 0) << Vector3(-1, 1, 0) << Vector3(1, 1, 0);

	glActiveTexture(texslots[0]);
	tf = new GLTexture1D(GL_RGBA32F_ARB, tfwidth, 0, GL_RGBA, qtfe->getTFColorMap(), GL_LINEAR, GL_LINEAR, GL_CLAMP_TO_EDGE);
	glDisable(GL_TEXTURE_1D);

	connect(qtfe->getTFPanel(), SIGNAL(tfChanged(float*)), this, SLOT(tfChanged(float*)));
}


Preintegrator::Preintegrator(QTFEditor* qtfe, float basesteps, float steps, QObject* parent, GLenum textype):QObject(parent),
uspecular(0), udiffuse(0), usteps(0), ubasesteps(0),
needsupdate(true), qtfe(qtfe), tfwidth(qtfe->getTFColorMapResolution()) {
	if(!resourcesinit) {
		Q_INIT_RESOURCE(shaders);
		resourcesinit = true;
	}

	preintshader = new Shader("preint");
	preintshader->addFragmentShader(":/preint.frag");
	preintshader->addVertexShader(":/preint.vert");

	preintshader->compileAndLink();

	preintshader->addUniformf("basesteps", 1, basesteps);
	preintshader->addUniformf("steps", 1, steps);
	preintshader->addUniformf("diffuse", 4, 1.0, 1.0, 1.0, 0.6);
	preintshader->addUniformf("specular", 4, 1.0, 1.0, 1.0, 0.2);
	preintshader->addUniformf("distscale", 1, 1.);
	preintshader->addUniformf("deltascale", 1, 1.);
	preintshader->addUniformf("minstep", 1, 1.);

	tf = new GLTexture1D(GL_RGBA32F_ARB, tfwidth, 0, GL_RGBA, qtfe->getTFColorMap(), GL_LINEAR, GL_LINEAR, GL_CLAMP);
	preintshader->addUniformf("width", 1, (double)tfwidth);
	preintshader->addUniformSampler("tf", tf);

	fbo = new GLFramebufferObject(tfwidth, tfwidth, GL_NONE);
	for(int i = 0; i < 3; ++i) {
		GLDEBUG(fbo->attachTexture(new GLTexture2D(GL_TEXTURE_2D, textype, tfwidth, tfwidth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP), GL_COLOR_ATTACHMENT0_EXT + i));
		//GLDEBUG(glDisable(GL_TEXTURE_2D));
	}
	/*
	GLDEBUG(GLTexture2D *temptex = new GLTexture2D(GL_TEXTURE_2D, GL_RGBA32F_ARB, tfwidth, tfwidth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP));
	GLDEBUG(glActiveTexture(texslots[3]));
	GLDEBUG(fbo->attachTexture(temptex, GL_COLOR_ATTACHMENT2_EXT)); */

	//GLDEBUG(glDisable(GL_TEXTURE_2D));

	fbo->checkStatus();
	vbo = new GLVertexbufferf(GL_TRIANGLE_STRIP, GL_STATIC_DRAW);
	(*vbo) << Vector3(-1, -1, 0) << Vector3(1, -1, 0) << Vector3(-1, 1, 0) << Vector3(1, 1, 0);

	glDisable(GL_TEXTURE_1D);

	connect(qtfe->getTFPanel(), SIGNAL(tfChanged(float*)), this, SLOT(tfChanged(float*)));
}

Preintegrator::Preintegrator(int tfwidth, float basesteps, float steps, QObject* parent):QObject(parent),
uspecular(0), udiffuse(0), usteps(0), ubasesteps(0),
needsupdate(true), qtfe(qtfe), tfwidth(tfwidth) {
	if(!resourcesinit) {
		Q_INIT_RESOURCE(shaders);
		resourcesinit = true;
	}

	preintshader = new Shader("preint");
	preintshader->addFragmentShader(":/preint.frag");
	preintshader->addVertexShader(":/preint.vert");

	preintshader->compileAndLink();

	preintshader->addUniformf("basesteps", 1, basesteps);
	preintshader->addUniformf("steps", 1, steps);
	preintshader->addUniformf("diffuse", 4, 1.0, 1.0, 1.0, 0.6);
	preintshader->addUniformf("specular", 4, 1.0, 1.0, 1.0, 0.2);
	preintshader->addUniformf("distscale", 1, 1.);
	preintshader->addUniformf("deltascale", 1, 1.);
	preintshader->addUniformf("minstep", 1, 1.);

	tf = new GLTexture1D(GL_RGBA32F_ARB, tfwidth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP);
	preintshader->addUniformf("width", 1, (double)tfwidth);
	preintshader->addUniformSampler("tf", tf);

	fbo = new GLFramebufferObject(tfwidth, tfwidth, GL_NONE);
	for(int i = 0; i < 3; ++i) {
		GLDEBUG(fbo->attachTexture(new GLTexture2D(GL_TEXTURE_2D, GL_RGBA32F_ARB, tfwidth, tfwidth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP), GL_COLOR_ATTACHMENT0_EXT + i));
	}

	fbo->checkStatus();
	vbo = new GLVertexbufferf(GL_TRIANGLE_STRIP, GL_STATIC_DRAW);
	(*vbo) << Vector3(-1, -1, 0) << Vector3(1, -1, 0) << Vector3(-1, 1, 0) << Vector3(1, 1, 0);

}

Preintegrator::~Preintegrator() {
	delete preintshader;
	delete fbo;
	delete vbo;
}

void Preintegrator::update() {
	if(!needsupdate)
		return;
	needsupdate = false;
	tf->reload(qtfe->getTFColorMap());
	glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_CULL_FACE);
	glViewport(0, 0, tfwidth, tfwidth);

	Shader* current = Shader::current();
	//qDebug("preint steps: %f", preintshader->getFloats()["steps"]->v()[0]);
	//qDebug("preint basesteps: %f", preintshader->getFloats()["basesteps"]->v()[0]);
	preintshader->use();

	GLAbstractFBO* cfbo = GLAbstractFBO::current();

	fbo->bind();
	vbo->draw();


	if(current)
		current->use();
	else
		preintshader->release();

	if(cfbo)
		cfbo->bind();
	else
		fbo->release();

	glPopAttrib();
}

void Preintegrator::setBaseSteps(UniformFloat* u) {
	if(ubasesteps) {
		disconnect(ubasesteps, SIGNAL(valueChanged(float)), this, SLOT(setBaseSteps(float)));
	}
	if(!u)
		return;
	ubasesteps = u;
	connect(ubasesteps, SIGNAL(valueChanged(float)), this, SLOT(setBaseSteps(float)));
	setBaseSteps(u->v()[0]);
}

void Preintegrator::setSteps(UniformFloat* u) {
	if(usteps) {
		disconnect(usteps, SIGNAL(valueChanged(float)), this, SLOT(setSteps(float)));
	}
	if(!u)
		return;
	usteps = u;
	connect(usteps, SIGNAL(valueChanged(float)), this, SLOT(setSteps(float)));
	setSteps(u->v()[0]);
}

void Preintegrator::setDeltaScale(UniformFloat* u) {
	if(udeltascale) {
		disconnect(udeltascale, SIGNAL(valueChanged(float)), this, SLOT(setDeltaScale(float)));
	}
	if(!u)
		return;
	udeltascale = u;
	connect(udeltascale, SIGNAL(valueChanged(float)), this, SLOT(setDeltaScale(float)));
	setDeltaScale(u->v()[0]);
}

void Preintegrator::setDiffuse(UniformFloat* u) {
	if(udiffuse) {
		disconnect(udiffuse, SIGNAL(valueChanged(float, float, float, float)), this, SLOT(setDiffuse(float, float, float, float)));
	}
	if(!u)
		return;
	udiffuse = u;
	connect(udiffuse, SIGNAL(valueChanged(float, float, float, float)), this, SLOT(setDiffuse(float, float, float, float)));
	setDiffuse(u->v()[0], u->v()[1], u->v()[2], u->v()[3]);
}

void Preintegrator::setSpecular(UniformFloat* u) {
	if(uspecular) {
		disconnect(uspecular, SIGNAL(valueChanged(float, float, float, float)), this, SLOT(setSpecular(float, float, float, float)));
	}
	if(!u)
		return;
	uspecular = u;
	connect(uspecular, SIGNAL(valueChanged(float, float, float, float)), this, SLOT(setSpecular(float, float, float, float)));
	setSpecular(u->v()[0], u->v()[1], u->v()[2], u->v()[3]);
}

void Preintegrator::setSteps(float s) {
	preintshader->getFloats()["steps"]->setValues(s);
	needsupdate = true;
}

void Preintegrator::setDeltaScale(float s) {
	preintshader->getFloats()["deltascale"]->setValues(s);
	needsupdate = true;
}

void Preintegrator::setBaseSteps(float s) {
	preintshader->getFloats()["basesteps"]->setValues(s);
	needsupdate = true;
}

void Preintegrator::setDiffuse(float r, float g, float b, float a) {
	preintshader->getFloats()["diffuse"]->setValues(r,g,b,a);
	needsupdate = true;
}

void Preintegrator::setSpecular(float r, float g, float b, float a) {
	preintshader->getFloats()["specular"]->setValues(r,g,b,a);
	needsupdate = true;
}

void Preintegrator::forceUpdate() {
	needsupdate = true;
	update();
}

void Preintegrator::tfChanged(float* newtf) {
	//glActiveTexture(texslots[0]);
	needsupdate = true;
	emit updated();
}

Preintegrator3D::Preintegrator3D(const GLenum *texslot, QTFEditor* qtfe, int, int depth, float basesteps, float steps, QObject* parent)
:Preintegrator(qtfe, parent), depth(depth), logtable(true), logvalue(16) {
	for(int i = 0; i < 4; ++i) {
		texslots[i] = texslot[i];
	}

	if(!resourcesinit) {
		Q_INIT_RESOURCE(shaders);
		resourcesinit = true;
	}

	preintshader = new Shader("preint");
	preintshader->addFragmentShader(":/preint3d.frag");
	preintshader->addVertexShader(":/preint.vert");

	preintshader->compileAndLink();

	preintshader->addUniformf("basesteps", 1, basesteps);
	preintshader->addUniformf("steps", 1, steps);
	preintshader->addUniformf("diffuse", 4, 1.0, 1.0, 1.0, 0.6);
	preintshader->addUniformf("specular", 4, 1.0, 1.0, 1.0, 0.2);

	preintshader->addUniformf("width", 1, (double)tfwidth);
	preintshader->addUniformi("tf", 1, texslots[0] - GL_TEXTURE0);
	preintshader->addUniformf("distscale", 1, 1.);
	preintshader->addUniformf("deltascale", 1, 1.);
	preintshader->addUniformf("minstep", 1, 1.);


	fbo = new GLFramebufferObject(tfwidth, tfwidth, GL_NONE);
	GLDEBUG(glActiveTexture(texslots[1]));
	GLDEBUG(amb = new GLTexture3D(GL_RGBA32F_ARB, tfwidth, tfwidth, depth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR));
	amb->bind();
	GLDEBUG(glActiveTexture(texslots[2]));
	GLDEBUG(front = new GLTexture3D(GL_RGBA32F_ARB, tfwidth, tfwidth, depth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR));
	front->bind();

	/*
	GLDEBUG(GLTexture2D *temptex = new GLTexture2D(GL_TEXTURE_2D, GL_RGBA32F_ARB, tfwidth, tfwidth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP));
	GLDEBUG(glActiveTexture(texslots[3]));
	GLDEBUG(fbo->attachTexture(temptex, GL_COLOR_ATTACHMENT2_EXT)); */

	//GLDEBUG(glDisable(GL_TEXTURE_2D));

	vbo = new GLVertexbufferf(GL_TRIANGLE_STRIP, GL_STATIC_DRAW);
	(*vbo) << Vector3(-1, -1, 0) << Vector3(1, -1, 0) << Vector3(-1, 1, 0) << Vector3(1, 1, 0);

	glActiveTexture(texslots[0]);
	tf = new GLTexture1D(GL_RGBA32F_ARB, tfwidth, 0, GL_RGBA, qtfe->getTFColorMap(), GL_LINEAR, GL_LINEAR, GL_MIRRORED_REPEAT);
	glDisable(GL_TEXTURE_1D);

	connect(qtfe->getTFPanel(), SIGNAL(tfChanged(float*)), this, SLOT(tfChanged(float*)));
}

Preintegrator3D::Preintegrator3D(QTFEditor* qtfe, int depth, float basesteps, float steps, GLenum textype, QObject* parent, bool useliveshaders, const QString& path, ShaderManager* sm)
	:Preintegrator(qtfe, parent), depth(depth), logtable(true), logvalue(16) {
	if(!resourcesinit) {
		Q_INIT_RESOURCE(shaders);
		resourcesinit = true;
	}

	preintshader = new Shader("preint");
	if(!useliveshaders) {
		preintshader->addFragmentShader(":/preint3d.frag");
		preintshader->addVertexShader(":/preint.vert");
	} else {
		QDir dir(path);
		preintshader->addFragmentShader(dir.absoluteFilePath("preint3d.frag"));
		preintshader->addVertexShader(dir.absoluteFilePath("preint.vert"));
		if(sm)
			sm->addShader("preint", preintshader);
	}

	preintshader->compileAndLink();

	preintshader->addUniformf("basesteps", 1, basesteps);
	preintshader->addUniformf("steps", 1, steps);
	preintshader->addUniformf("diffuse", 4, 1.0, 1.0, 1.0, 0.6);
	preintshader->addUniformf("specular", 4, 1.0, 1.0, 1.0, 0.2);
	preintshader->addUniformf("distscale", 1, 1.);
	preintshader->addUniformf("deltascale", 1, 1.);
	preintshader->addUniformf("minstep", 1, 1.);

	tf = new GLTexture1D(GL_RGBA32F_ARB, tfwidth, 0, GL_RGBA, qtfe->getTFColorMap(), GL_LINEAR, GL_LINEAR, GL_CLAMP);
	preintshader->addUniformf("width", 1, (double)tfwidth);
	preintshader->addUniformSampler("tf", tf);

	fbo = new GLFramebufferObject(tfwidth, tfwidth, GL_NONE);
	GLDEBUG(amb = new GLTexture3D(textype, tfwidth, tfwidth, depth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP, GL_CLAMP));
	GLDEBUG(front = new GLTexture3D(textype, tfwidth, tfwidth, depth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP, GL_CLAMP));
	GLDEBUG(back = new GLTexture3D(textype, tfwidth, tfwidth, depth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP, GL_CLAMP));
	/*
	GLDEBUG(GLTexture2D *temptex = new GLTexture2D(GL_TEXTURE_2D, GL_RGBA32F_ARB, tfwidth, tfwidth, 0, GL_RGBA, 0, GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP));
	GLDEBUG(glActiveTexture(texslots[3]));
	GLDEBUG(fbo->attachTexture(temptex, GL_COLOR_ATTACHMENT2_EXT)); */

	//GLDEBUG(glDisable(GL_TEXTURE_2D));

	fbo->checkStatus();
	vbo = new GLVertexbufferf(GL_TRIANGLE_STRIP, GL_STATIC_DRAW);
	(*vbo) << Vector3(-1, -1, 0) << Vector3(1, -1, 0) << Vector3(-1, 1, 0) << Vector3(1, 1, 0);

	glDisable(GL_TEXTURE_1D);

	connect(qtfe->getTFPanel(), SIGNAL(tfChanged(float*)), this, SLOT(tfChanged(float*)));
}

void Preintegrator3D::update() {
	if(!needsupdate)
		return;
	needsupdate = false;
	tf->reload(qtfe->getTFColorMap());
	glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_CULL_FACE);
	glViewport(0, 0, tfwidth, tfwidth);

	Shader* current = Shader::current();
	preintshader->use();

	GLAbstractFBO* cfbo = GLFramebufferObject::current();

	float minstep = 1/(depth - 1.);
	if(logtable)
		minstep = (powf(logvalue, minstep) - 1)/(logvalue - 1.);
	minstep /= tfwidth;
	preintshader->getFloats()["minstep"]->setValues(minstep);
	qDebug("%f minstep", minstep);
	fbo->bind();
	//float oldsteps = preintshader->getFloats()["steps"]->v()[0];
	for(int i = 0; i < depth; ++i) {
		float distscale = logtable ? (powf(logvalue,i/(depth - 1.)) - 1)/(logvalue - 1.)
			: i/(depth - 1.);
		//qDebug("UPDATE LAWL: %f", distscale);

		preintshader->getFloats()["distscale"]->setValues(distscale);

		fbo->attachTextureLayer(amb, i, GL_COLOR_ATTACHMENT0_EXT);
		fbo->attachTextureLayer(front, i, GL_COLOR_ATTACHMENT2_EXT);
		fbo->attachTextureLayer(back, i, GL_COLOR_ATTACHMENT1_EXT);
		/*
		glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, amb->getHandle(), 0, i);
		glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, back->getHandle(), 0, i);
		glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, front->getHandle(), 0, i); */

		//glClear(GL_COLOR_BUFFER_BIT);

		if(fbo->checkStatus())
			vbo->draw();
		else {
			qFatal("DLKJALKDJFLDKJ");
			break;
		}
	}
/*	float* imagef = new float[256*256*512*4];

	glPushAttrib(GL_TEXTURE_BIT);

	//unsigned char* image = new unsigned char[256*256*512*4];
	glBindTexture(GL_TEXTURE_3D, amb->getHandle());
	glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, imagef);


	QImage(image, 256, 256, QImage::Format_ARGB32).save("0.png");
	QImage(image + 256*256*511*4, 256, 256, QImage::Format_ARGB32).save("511.png");
	QImage(image + 256*256*510*4, 256, 256, QImage::Format_ARGB32).save("510.png");
	QImage(image + 256*256*4, 256, 256, QImage::Format_ARGB32).save("1.png");

	for(int i=0;i<256*256*4;i++)
	{
		imagef[i]=0.0;
	}
	amb->reload(imagef);

	//delete [] image;
	delete [] imagef;

	glPopAttrib();

	//preintshader->getFloats()["steps"]->setValues(oldsteps);
*/
	if(current)
		current->use();
	else
		preintshader->release();

	if(cfbo)
		cfbo->bind();
	else
		fbo->release();

	glPopAttrib();
}

Preintegrator3D::~Preintegrator3D() {
	//delete amb;
	//delete front;
}

GLTexture* Preintegrator::getTex(int index) {
	if(!fbo)
		return 0;
	return (*fbo)[index];
}

GLTexture* Preintegrator3D::getTex(int index) {
	return index ? (index == 1 ? back : front) : amb;
}

void Preintegrator3D::setLogValue(float v) {
	logvalue = v;
	needsupdate = true;
}

void Preintegrator3D::setLogTable(bool v) {
	logtable = v;
	needsupdate = true;
}
