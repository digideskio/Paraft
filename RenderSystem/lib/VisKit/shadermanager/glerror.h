#ifndef GLERROR_H
#define GLERROR_H

#ifdef DEBUG
#define GLERROR(x) x; printGLError(__FILE__, __LINE__)
#else
#define GLERROR(x) x
#endif

void printGLError(const char* file, int line);
void toggleGLErrors(bool v);
#endif // GLERROR_H
