#ifndef _QANIINTERFACE_H_
#define _QANIINTERFACE_H_

#include "camera.h"
#include "slicer.h"
#include <QImage>

class QTFEditor;

class QAniInterface {
public:
	QAniInterface() {}
	// for QAniEditor to fetch information
	// virtual ones are implemented in the inheriting class
	virtual Camera& getCamera()=0;
	virtual Slicer& getSlicer()=0;
	virtual size_t	getTotalSteps()=0;
	virtual size_t	getCurrentStep()=0;
	virtual QTFEditor * getTFEditor()=0;
	virtual QImage getTexture()=0;

};


#endif


