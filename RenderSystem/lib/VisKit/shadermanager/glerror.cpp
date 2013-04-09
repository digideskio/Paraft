#include <GL/glew.h>
#include "glerror.h"
#include <QGLContext>
#include <QtDebug>
static bool printErrors = true;
void toggleGLErrors(bool v) {
	printErrors = v;
}
void printGLError(const char* file, int line) {
	if(printErrors && QGLContext::currentContext() && QGLContext::currentContext()->isValid()) {
		for(GLenum error = glGetError(); error != GL_NO_ERROR; error = glGetError())
			qDebug("GL Error in %s on line %d: %s", file, line, gluErrorString(error));
	}
}
