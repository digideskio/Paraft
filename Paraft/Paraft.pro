QMAKE_CC     = gcc
QMAKE_CXX    = g++
#QMAKE_CC     = /usr/local/bin/mpicc
#QMAKE_CXX    = /usr/local/bin/mpic++
#LIBS        += -lmpi_cxx -lmpi -lopen-rte -lopen-pal -lutil
INCLUDEPATH += -I/usr/local/include

#QMAKE_CXXFLAGS += -fopenmp
#QMAKE_LFLAGS   += -fopenmp
LIBS += -L/usr/local/lib -lm -lopencv_core -lopencv_highgui

SOURCES += \
    Main.cpp \
    DataManager.cpp \
    FeatureTracker.cpp \
    BlockController.cpp \
    Metadata.cpp \
    SuperVoxel.cpp \
    SuperPixel.cpp

HEADERS += \
    DataManager.h \
    FeatureTracker.h \
    BlockController.h \
    Utils.h \
    Metadata.h \
    SuperVoxel.h \
    SuperPixel.h

OTHER_FILES += \
    vorts.config \
    jet.config \
    supervoxel.config
