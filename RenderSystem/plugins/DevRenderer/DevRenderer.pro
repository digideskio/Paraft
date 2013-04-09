VISKIT_PATH = ../../lib/VisKit
include($${VISKIT_PATH}/VisIncludes.pro)

TEMPLATE = lib
QT += opengl
QT += network

INCLUDEPATH += ../.. \
    ./lib

win32 {
LIBS += -lglew32
}

macx {
DEFINES += NO_CXX11_STL
}

unix:!macx {
QMAKE_CXXFLAGS += -std=gnu++0x
}

HEADERS += \
    ../../PluginInterface.h \
    DevRenderer.h \
    VolumeRenderWindow.h \
    VolumeParser.h \
    RenderEffectPanel.h \
    ParameterEditor.h \
    PreIntegrator.h \
    lib/QParameterSet.h \
    lib/ParameterSet.h \
    lib/MSVectors.h \
    lib/MSGLTexture.h \
    lib/MSGLFramebufferObject.h \
    lib/GLShader.h \
    lib/Containers.h \
    RayCastingRenderer.h \
    PreIntegrationRenderer.h \
    lib/JsonParser.h \
    PreIntegratorGL.h \
    MainUI.h \
    VolumeMetadata.h \
    VolumeData.h \
    VolumeModel.h \
    VolumeDataBlock.h \
    SegmentedRayCastingRenderer.h \
    UDPListener.h \
    lib/GLBuffer.h \
    VolumeRenderer.h \
    SegmentedVolumeRenderer.h

SOURCES += \
    DevRenderer.cpp \
    VolumeRenderWindow.cpp \
    VolumeParser.cpp \
    PreIntegrator.cpp \
    lib/MSGLTexture.cpp \
    lib/MSGLFramebufferObject.cpp \
    lib/GLShader.cpp \
    RayCastingRenderer.cpp \
    PreIntegrationRenderer.cpp \
    lib/JsonParser.cpp \
    PreIntegratorGL.cpp \
    MainUI.cpp \
    VolumeMetadata.cpp \
    VolumeData.cpp \
    VolumeModel.cpp \
    VolumeDataBlock.cpp \
    SegmentedRayCastingRenderer.cpp \
    UDPListener.cpp \
    lib/GLBuffer.cpp \
    VolumeRenderer.cpp \
    SegmentedVolumeRenderer.cpp

DESTDIR = ..

OTHER_FILES += \
    shaders/preInt.frag \
    shaders/genPreInt.vert \
    shaders/genPreInt.frag \
    shaders/sliceRaycasting.frag \
    shaders/regularRaycasting.frag \
    shaders/raycasting.vert \
    shaders/SegmentedRayCasting.vert \
    shaders/SegmentedRayCasting.frag \
    shaders/Default.vert \
    shaders/CopyColor.frag \
    shaders/SegmentedRayCastingPreInt.frag
