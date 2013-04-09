TEMPLATE = app
TARGET = 
DEPENDPATH += .
INCLUDEPATH += .
CONFIG += debug_and_release

# Input
HEADERS += option.h \
	optionmanager.h \
	optiongroupbox.h \
	optionslider.h
SOURCES += option.cpp \
	optionmanager.cpp \
	optiongroupbox.cpp \
	optionslider.cpp


SOURCES += main.cpp \
	optiontest.cpp
HEADERS += optiontest.h
