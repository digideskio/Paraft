#-------------------------------------------------
#
# Project created by QtCreator 2011-09-08T15:54:47
#
#-------------------------------------------------

QT       += core gui

TARGET = lamp_qt
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH += ../../lib/include \
               ../../lib/include/pan

LIBS += -L ../../lib/bin/libgsl.so \
        -L ../../lib/bin/libgslcblas.so \
        -L ../../lib/bin/libpanuseful.so \
        -L ../../lib/bin/libpanmath.so \
        -L ../../lib/bin/libpanmetric.so \
        -L ../../lib/bin/libpandconv.so \
        -L ../../lib/bin/libpanestimate.so \
        -L ../../lib/bin/libpanforce.so \
        -L ../../lib/bin/libpanlamp.so





