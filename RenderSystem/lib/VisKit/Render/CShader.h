#ifndef _CSHADER_H_
#define _CSHADER_H_

#define GL_GLEXT_PROTOTYPES
#ifdef WIN32
#include <windows.h>
#endif
#include <Qt>
#include <QGLWidget>

//#ifdef MAC_OSX
//#include <OpenGL/gl.h>
//#include <OpenGL/glu.h>
//#else
//#include <GL/gl.h>
//#include <GL/glu.h>
//#endif
#include <vector>
#include "CObject.h"

using namespace std;

extern bool verbose;

typedef struct {
	char name[80];
	int id;
	int	resID;
} CTexture;

class CProperties;
class QRenderWindow;

class CShader : public CObject{
public:
	CShader();
	virtual ~CShader();

	virtual void initialize() = 0;
	virtual void enableProgram(int numProgram) = 0;
	virtual void executeProgram(int numProgram) = 0;
	virtual void endProgram(int numProgram) = 0;
	virtual int numPrograms() = 0;
	virtual void loadProgram() = 0;
	virtual void loadPrograms() = 0;
	virtual void addProgram(const char* name, const char*entry = NULL)=0;

	virtual bool setShaderVariable1f(const char* name, float v1) = 0;
	virtual bool setShaderVariable3f(const char* name, float v1, float v2, float v3) = 0;
	virtual bool setShaderVariable4f(const char* name, float v1, float v2, float v3, float v4) = 0;
	
	void	setRenderWindow(QRenderWindow *r);
	int		getActiveProgram();
	void	setActiveProgram(int numProgram);
	
	virtual GLuint addTexture1D(const char *texname, void * data, int width, int clampx,  GLuint internalFormat, GLuint format, GLenum type, GLuint filter)=0;
	virtual GLuint addTexture2D(const char *texname, void * data, int width, int height, int clampx, int clampy,  GLuint internalFormat, GLuint format, GLenum type, GLuint filter)=0;
	virtual GLuint addTexture3D(const char *texname, void * data, int width, int height, int depth, int clampx, int clampy, int clampz, GLuint internalFormat, GLuint format, GLenum type, GLuint filter)=0;
	
protected:
	QRenderWindow	*m_pRenderer;
	
	int unnamedTextures;
	int currentInterpolation;
	int m_activeProgram;
	
	int findTexture(const char* texname);
	vector<CTexture> m_textures;
	void setTexture1D(GLuint name, void * data, int width, int clampx, GLuint internalFormat, GLuint format, GLenum type, GLuint filter);
	void setTexture2D(GLuint name, void * data, int width, int height, int clampx, int clampy, GLuint internalFormat, GLuint format, GLenum type, GLuint filter);
	void setTexture3D(GLuint name, void * data, int width, int height, int depth, int clampx, int clampy, int clampz, GLuint internalFormat, GLuint format, GLenum type, GLuint filter);
};



#endif

