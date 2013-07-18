#-------------------------------------------------
#
# Project created by QtCreator 2011-09-08T15:54:47
#
#-------------------------------------------------

TARGET   = lamp_qt
TEMPLATE = app
QT      += core gui
SOURCES += main.cpp mainwindow.cpp
HEADERS += mainwindow.h
FORMS   += mainwindow.ui

INCLUDEPATH += /Users/Yang/Develop/Paraft/MDS/pan-0.1.2/include
LIBS += -L/Users/Yang/Develop/Paraft/MDS/pan-0.1.2/lib
LIBS += -lgsl -lgslcblas -lpanuseful -lpanmath -lpanmetric -lpandconv -lpanestimate -lpanforce -lpanlamp
