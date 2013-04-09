CONFIG += cuda
CUDA_SOURCES += $${PWD}/streaksurface.cu
HEADERS += $${PWD}/streaksurface.h \
	$${PWD}/streaksurface.cuh
SOURCES += $${PWD}/streaksurface.cpp
INCLUDEPATH += $${PWD}
