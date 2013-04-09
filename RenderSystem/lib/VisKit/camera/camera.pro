TEMPLATE = lib
DEPENDPATH += .
INCLUDEPATH += .
CONFIG += qt staticlib debug_and_release
DEFINES += _OPENGL_CAMERA

# Input
HEADERS += camera.h quaternion.h \
 vectors.h \
 matrices.h
SOURCES += camera.cpp quaternion.cpp \
 vectors.cpp \
 matrices.cpp

win32 {
    CONFIG(debug | debug_and_release) {
        TARGET = cameradebug
    }
    release {
        TARGET = camera
    }
    LIBS += -lglew32
}


unix {
    CONFIG(debug | debug_and_release) {  
        TARGET = cameradebug
    }
    release {
        TARGET = camera
    }
    LIBS += -lGLEW
}
