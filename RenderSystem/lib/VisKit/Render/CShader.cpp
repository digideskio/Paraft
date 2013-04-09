#include <cstdio>
#ifdef MAC_OSX
#include <glew.h>
#else
#include <GL/glew.h>
#endif
#include "CShader.h"
#include "CProperties.h"
#include "QRenderWindow.h"
using namespace std;

CShader::CShader(){
	unnamedTextures = 0;
	currentInterpolation = GL_LINEAR;
	
	m_activeProgram = 0;
	m_textures.resize(0);	
}
CShader::~CShader(){
}
void CShader::setRenderWindow(QRenderWindow *r){
	m_pRenderer = r;
}
int CShader::findTexture(const char* texname){
	if(!texname) return -1;
	for(int i=0;i<m_textures.size();i++) 
	{
		if(strcmp(m_textures[i].name,texname)==0)
			return m_textures[i].id;
	}
	return -1;
}
void CShader::setTexture1D(GLuint name, void * data, int width, int clampx, GLuint internalFormat, GLuint format, GLenum type, GLuint filter) {
	glBindTexture(GL_TEXTURE_1D, name);

	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, clampx);

	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, filter);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, filter);
  
	float color[] = {0.0f, 0.0f, 0.0f, 0.0f};
	glTexParameterfv(GL_TEXTURE_1D, GL_TEXTURE_BORDER_COLOR, color);

	glTexImage1D(GL_TEXTURE_1D, 0, internalFormat, width, 0, format, type, data);

	if(glGetError() != GL_NO_ERROR) {
		fprintf(stderr,"[SHADER] An error (%d) occured when trying to open the texture 1D\n", glGetError());
	}
}
void CShader::setTexture2D(GLuint name, void * data, int width, int height, int clampx, int clampy, GLuint internalFormat, GLuint format, GLenum type, GLuint filter) {
	glBindTexture(GL_TEXTURE_2D, name);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, clampx);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, clampy);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);

	float color[] = {0.0f, 0.0f, 0.0f, 0.0f};
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);

	int error = glGetError();
	if(error != GL_NO_ERROR) {
		fprintf(stderr,"[SHADER] An error (%d) occured when trying to open the texture 2D\n", error);
	}
}
void CShader::setTexture3D(GLuint name, void * data, int width, int height, int depth, int clampx, int clampy, int clampz, GLuint internalFormat, GLuint format, GLenum type, GLuint filter) {
	glBindTexture(GL_TEXTURE_3D, name);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, clampx);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, clampy);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, clampz);
	//int mag = currentInterpolation;//GL_LINEAR;
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, filter);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, filter);

	float color[] = {0.0f, 0.0f, 0.0f, 0.0f};
	glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, color); 

	glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, format, type, data);
	int error = glGetError();
	if(error != GL_NO_ERROR) {
		fprintf(stderr,"[SHADER] An error (%d) occured when trying to open the texture 3D\n", error);
	}
}
int CShader::getActiveProgram() {
	return m_activeProgram;
}
void CShader::setActiveProgram(int numProgram) {
	m_activeProgram = numProgram;
	enableProgram(m_activeProgram);
}
