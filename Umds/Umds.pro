#-------------------------------------------------
#
# Project created by QtCreator 2013-08-22T16:38:32
#
#-------------------------------------------------

QMAKE_CXX       =  g++-4.8
QMAKE_CXXFLAGS  = -std=c++11
INCLUDEPATH    += /usr/local/include
INCLUDEPATH    += /usr/local/opt/qt5/include
LIBS           += -L/usr/local/lib
LIBS           += -L/usr/local/opt/qt5/lib
LIBS           += -lm -larmadillo

#QMAKE_CFLAGS_X86_64             -= -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5
#QMAKE_OBJECTIVE_CFLAGS_X86_64   -= -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5
#QMAKE_CXXFLAGS_X86_64           -= -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5
#QMAKE_LFLAGS_X86_64             -= -arch x86_64 -Xarch_x86_64 -mmacosx-version-min=10.5

QMAKE_LINK       = $$QMAKE_CXX
QMAKE_LINK_SHLIB = $$QMAKE_CXX

QT += core gui opengl widgets

TARGET = Umds
TEMPLATE = app

SOURCES += \
    main.cpp \
    MainWidget.cpp \
    Lamp.cpp \
    Node.cpp \
    ProjectionView.cpp

HEADERS += \
    MainWidget.h \
    Lamp.h \
    Node.h \
    ProjectionView.h \
    Utils.h
