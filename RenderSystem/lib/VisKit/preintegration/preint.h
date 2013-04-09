#ifndef _PREINT_H_
#define _PREINT_H_

#include <GL/glew.h>
#include <QObject>

class QTFEditor;
class GLFramebufferObject;
class GLVertexbufferf;
class Shader;
class GLTexture1D;
class GLTexture3D;
class UniformFloat;
class GLTexture;
class ShaderManager;
class Preintegrator : public QObject {
	Q_OBJECT
protected:
	GLenum texslots[4];
	GLFramebufferObject* fbo;
	GLVertexbufferf* vbo;
	Shader* preintshader;
	GLTexture1D* tf;

	UniformFloat* uspecular;
	UniformFloat* udiffuse;
	UniformFloat* usteps;
	UniformFloat* ubasesteps;
	UniformFloat* udeltascale;

	bool needsupdate;
	QTFEditor* qtfe;
	int tfwidth;
	Preintegrator(QTFEditor* qtfe, QObject* parent);
public:
	Preintegrator(const GLenum *texslot, QTFEditor* qtfe, int tfwidth=1024, float basesteps=512.f, float steps=512.f, QObject* parent=0);
	Preintegrator(QTFEditor* qtfe, float basesteps=512.f, float steps=512.f, QObject* parent=0, GLenum textype=GL_RGBA32F_ARB);
	Preintegrator(int tfwidth, float basesteps=512.f, float steps=512.f, QObject* parent=0);
	virtual ~Preintegrator();

	virtual void update();
	void setBaseSteps(UniformFloat* u);
	void setSteps(UniformFloat* u);
	void setDiffuse(UniformFloat* u);
	void setSpecular(UniformFloat* u);
	void setDeltaScale(UniformFloat* u);
	virtual GLTexture* getTex(int index);
	GLTexture1D* getTF() { return tf; }

	void rebind();
	bool needsUpdate() const { return needsupdate; }
	void forceUpdate();
	int getTFWidth() const { return tfwidth; }

public slots:
	void tfChanged(float*);
	void setDiffuse(float r, float g, float b, float a);
	void setSpecular(float r, float g, float b, float a);
	void setSteps(float s);
	void setBaseSteps(float s);
	void setDeltaScale(float s);
signals:
	void updated();
};

class Preintegrator3D : public Preintegrator {
	Q_OBJECT
	GLTexture3D* amb, *front, *back;
protected:
	int depth;
	bool logtable;
	float logvalue;
public:
	Preintegrator3D(const GLenum *texslot, QTFEditor* qtfe, int tfwidth=256, int depth=512, float basesteps=512.f, float steps=512.f, QObject* parent=0);
	Preintegrator3D(QTFEditor* qtfe, int depth, float basesteps=512.f, float steps=512.f, GLenum textype=GL_RGBA32F_ARB, QObject* parent=0, bool useliveshaders=false, const QString& path=QString(), ShaderManager* sm=0);
	void setLogTable(bool v);
	~Preintegrator3D();
	virtual void update();
	virtual GLTexture* getTex(int index);
public slots:
	void setLogValue(float v);

};

#endif
