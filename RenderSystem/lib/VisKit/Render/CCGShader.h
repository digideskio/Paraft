#ifndef _CCGSHADER_H_
#define _CCGSHADER_H_

class CCGShader : public CShader{
protected:
	
public:
	virtual void	initialize(){}
	virtual void	enableProgram(int){}
	virtual void	executeProgram(int){}
	virtual void	endProgram(int){}
	virtual int		numPrograms(){}
	virtual void	loadProgram(){}
	virtual void	loadPrograms(){}
	virtual void	addProgram(const char* name, const char*entry = NULL){}

	virtual bool setShaderVariable1f(const char* name, float v1){}
	virtual bool setShaderVariable3f(const char* name, float v1, float v2, float v3){}
	virtual bool setShaderVariable4f(const char* name, float v1, float v2, float v3, float v4){}

	virtual GLuint addTexture1D(const char *texname, void * data, int width, int clampx,  GLuint internalFormat, GLuint format, GLenum type, GLuint filter){}
	virtual GLuint addTexture2D(const char *texname, void * data, int width, int height, int clampx, int clampy,  GLuint internalFormat, GLuint format, GLenum type, GLuint filter){}
	virtual GLuint addTexture3D(const char *texname , void * data, int width, int height, int depth, int clampx, int clampy, int clampz, GLuint internalFormat, GLuint format, GLenum type, GLuint filter){}
	
public:
	CCGShader();
	virtual ~CCGShader();
};

#endif

