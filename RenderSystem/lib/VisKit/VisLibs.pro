TEMPLATE = subdirs
CONFIG += ordered debug_and_release
CONFIG(debug, debug|release):DEFINES += DEBUG

SUBDIRS += 	camera \
			shadermanager \
			UI \
			optionmanager/optionmanager.pro \
			Render/renderlib.pro \
			animator \
			preintegration
