#### MPI ################
QMAKE_CC     = /usr/local/bin/mpicc
QMAKE_CXX    = /usr/local/bin/mpic++
INCLUDEPATH += /usr/local/include
LIBS        += -L/usr/local/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -lutil
#########################

SOURCES += \
    Main.cpp \
    DataManager.cpp \
    FeatureTracker.cpp \
    BlockController.cpp

HEADERS += \
    DataManager.h \
    FeatureTracker.h \
    BlockController.h \
    Utils.h
