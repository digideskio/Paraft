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

QT += core gui widgets

TARGET = Umds
TEMPLATE = app

SOURCES += main.cpp \
        MainWidget.cpp \
        Lamp.cpp

HEADERS += MainWidget.h \
        Lamp.h
