QMAKE_CXX       = g++-4.8
QMAKE_CXXFLAGS  = -std=c++11
INCLUDEPATH     = -I/usr/local/include
LIBS            = -L/usr/local/lib -lm -lopencv_core -lopencv_highgui

QMAKE_LINK       = $$QMAKE_CXX
QMAKE_LINK_SHLIB = $$QMAKE_CXX

SOURCES += \
    Main.cpp \
    DataManager.cpp \
    FeatureTracker.cpp \
    BlockController.cpp \
    Metadata.cpp \
    SuperVoxel.cpp

HEADERS += \
    DataManager.h \
    FeatureTracker.h \
    BlockController.h \
    Utils.h \
    Metadata.h \
    SuperVoxel.h

OTHER_FILES += \
    vorts.config \
    jet.config \
    supervoxel.config
