INCLUDEPATH += $${VISKIT_PATH}/camera \
	$${VISKIT_PATH}/shadermanager \
	$${VISKIT_PATH}/Render \
	$${VISKIT_PATH}/UI/QTFEditor \
	$${VISKIT_PATH}/UI/QColorPicker \
	$${VISKIT_PATH}/UI/QAniEditor \
	$${VISKIT_PATH}/optionmanager \
	$${VISKIT_PATH}/animtor
CAMERA_HEADERS += $${VISKIT_PATH}/camera/camera.h \
	$${VISKIT_PATH}/camera/vectors.h \
	$${VISKIT_PATH}/camera/matrices.h \
	$${VISKIT_PATH}/camera/quaternion.h
OPTION_MANAGER_HEADERS += $${VISKIT_PATH}/optionmanager/option.h \
	$${VISKIT_PATH}/optionmanager/optionmanager.h
SHADER_MANAGER_HEADERS += $${VISKIT_PATH}/shadermanager/shader.h \
	$${VISKIT_PATH}/shadermanager/shadermanager.h \
	$${VISKIT_PATH}/shadermanager/gltexture.h \
	$${VISKIT_PATH}/shadermanager/glframebufferobject.h
RENDER_HEADERS += $${VISKIT_PATH}/Render/CData.h \
	$${VISKIT_PATH}/Render/CStructuredMeshData.h \
	$${VISKIT_PATH}/Render/CImageData.h \
	$${VISKIT_PATH}/Render/CObject.h \
	$${VISKIT_PATH}/Render/CProperties.h \
	$${VISKIT_PATH}/Render/QRenderEffEditor.h \
	$${VISKIT_PATH}/Render/QRenderWindow.h \
	$${VISKIT_PATH}/Render/box.h \
	$${VISKIT_PATH}/Render/slicer.h \
	$${VISKIT_PATH}/Render/LIGHTPARAM.h
QANIEDITOR_HEADERS += $${VISKIT_PATH}/UI/QAniEditor/QAniEditor.h \
	$${VISKIT_PATH}/UI/QAniEditor/QAniClickable.h \
	$${VISKIT_PATH}/UI/QAniEditor/QAniKeyframe.h \
	$${VISKIT_PATH}/UI/QAniEditor/QAniInstance.h \
	$${VISKIT_PATH}/UI/QAniEditor/QAniGraph.h \
	$${VISKIT_PATH}/UI/QAniEditor/QAniInterface.h
ANIMATOR_HEADERS += $${VISKIT_PATH}/animator/animator.h \
	$${VISKIT_PATH}/animator/cameraanimator.h \
	$${VISKIT_PATH}/animator/animationmanager.h \
	$${VISKIT_PATH}/animator/floatanimator.h
QTFEDITOR_HEADERS += $${VISKIT_PATH}/UI/QTFEditor/histogram.h \
	$${VISKIT_PATH}/UI/QTFEditor/QHistogram.h \
	$${VISKIT_PATH}/UI/QTFEditor/QTFEditor.h \
	$${VISKIT_PATH}/UI/QTFEditor/QTFPanel.h \
	$${VISKIT_PATH}/UI/QTFEditor/QTFColormap.h
ALL_VISKIT_HEADERS += $$CAMERA_HEADERS \
	$$SHADER_MANAGER_HEADERS \
	$$RENDER_HEADERS \
	$$QTFEDITOR_HEADERS \
	$$QANIEDITOR_HEADERS \
	$$OPTION_MANAGER_HEADERS \
	$$ANIMATOR_HEADERS
INCLUDEPATH += $${VISKIT_PATH}/UI/QTFEditor \
	$${VISKIT_PATH}/UI/QColorPicker \
	$${VISKIT_PATH}/UI/QAniEditor \
	$${VISKIT_PATH}/camera \
	$${VISKIT_PATH}/Render \
	$${VISKIT_PATH}/shadermanager \
	$${VISKIT_PATH}/optionmanager \
	$${VISKIT_PATH}/animator \
	$${VISKIT_PATH}/preintegration \
	$${VISKIT_PATH}/util

CONFIG(debug, debug|release):DEFINES += DEBUG
win32 {
	LIBS += -lglew32
	CONFIG(debug, debug|release):LIBS += $${VISKIT_PATH}/Render/debug/Render.lib \
		$${VISKIT_PATH}/camera/debug/camera.lib \
		$${VISKIT_PATH}/shadermanager/debug/shadermanager.lib \
		$${VISKIT_PATH}/optionmanager/debug/optionmanager.lib \
		$${VISKIT_PATH}/UI/QTFEditor/debug/QTFEditor.lib \
		$${VISKIT_PATH}/UI/QColorPicker/debug/QColorPicker.lib \
		$${VISKIT_PATH}/UI/QAniEditor/debug/QAniEditor.lib \
		$${VISKIT_PATH}/preintegration/debug/preintegration.lib \
		$${VISKIT_PATH}/animator/debug/animator.lib
	CONFIG(release, debug|release):LIBS += $${VISKIT_PATH}/Render/release/Render.lib \
		$${VISKIT_PATH}/camera/release/camera.lib \
		$${VISKIT_PATH}/shadermanager/release/shadermanager.lib \
		$${VISKIT_PATH}/optionmanager/release/optionmanager.lib \
		$${VISKIT_PATH}/UI/QTFEditor/release/QTFEditor.lib \
		$${VISKIT_PATH}/UI/QColorPicker/release/QColorPicker.lib \
		$${VISKIT_PATH}/UI/QAniEditor/release/QAniEditor.lib \
		$${VISKIT_PATH}/preintegration/release/preintegration.lib \
		$${VISKIT_PATH}/animator/release/animator.lib
}
unix:LIBS += \
	$${VISKIT_PATH}/UI/QTFEditor/libQTFEditor.a \
	$${VISKIT_PATH}/UI/QColorPicker/libQColorPicker.a \
	$${VISKIT_PATH}/UI/QAniEditor/libQAniEditor.a \
	$${VISKIT_PATH}/Render/librender.a \
	$${VISKIT_PATH}/preintegration/libpreintegration.a \
	$${VISKIT_PATH}/shadermanager/libshadermanager.a \
	$${VISKIT_PATH}/camera/libcamera.a \
	$${VISKIT_PATH}/optionmanager/liboptionmanager.a \
	$${VISKIT_PATH}/animator/libanimator.a \
	-lGLEW
macx:LIBS += \
    $${VISKIT_PATH}/UI/QTFEditor/libQTFEditor.a \
    $${VISKIT_PATH}/UI/QColorPicker/libQColorPicker.a \
    $${VISKIT_PATH}/UI/QAniEditor/libQAniEditor.a \
    $${VISKIT_PATH}/Render/librender.a \
    $${VISKIT_PATH}/shadermanager/libshadermanager.a \
    $${VISKIT_PATH}/preintegration/libpreintegration.a \
    $${VISKIT_PATH}/camera/libcamera.a \
    $${VISKIT_PATH}/optionmanager/liboptionmanager.a \
	$${VISKIT_PATH}/animator/libanimator.a \
	-lGLEW
