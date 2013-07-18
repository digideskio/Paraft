QMAKE_CXX        =  g++-4.8
QMAKE_CXXFLAGS   = -std=c++11
QMAKE_LINK       = $$QMAKE_CXX
QMAKE_LINK_SHLIB = $$QMAKE_CXX

TARGET   = lamp_qt
TEMPLATE = app
QT      += core gui
SOURCES += main.cpp mainwindow.cpp
HEADERS += mainwindow.h
FORMS   += mainwindow.ui

INCLUDEPATH += /Users/Yang/Develop/Paraft/MDS/pan-0.1.2/include
LIBS += -L/Users/Yang/Develop/Paraft/MDS/pan-0.1.2/lib
LIBS += -lgsl -lgslcblas -lpanuseful -lpanmath -lpanmetric -lpandconv -lpanestimate -lpanforce -lpanlamp
