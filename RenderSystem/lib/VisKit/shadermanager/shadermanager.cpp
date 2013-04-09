#include <GL/glew.h>
#include "shadermanager.h"
#include "shaderslider.h"
#include <QGroupBox>
#include <QVBoxLayout>
#include <QDataStream>

Shader* ShaderManager::shaderdummy = 0;

ShaderManager::ShaderManager(int msgLevel):options(0), messageLevel(msgLevel) {
}

void ShaderManager::addShader(const QString& key, Shader* shader) {
	shaders[key] = shader;
}

void ShaderManager::useShader(const QString& key) {
	shaders[key]->use();
}

void ShaderManager::addShader(const QString& key, const QString& display) {
	shaders[key] = new Shader(display.isEmpty() ? key : display);
	createdShaders.push_back(shaders[key]);
}

void ShaderManager::reload() {
	if (messageLevel > 0) qDebug("Compiling Shaders.");
	for(QHash<QString, Shader*>::iterator it = shaders.begin(); it != shaders.end(); it++) {
		if(*it)
			(*it)->compileAndLink();
	}
	if (messageLevel > 0) qDebug("Done.");
}

Shader*& ShaderManager::operator[](const QString& index) {
	if(index != QString("current"))
		return shaders[index];
	shaderdummy = Shader::current();
	return shaderdummy;
}

void ShaderManager::useFixed() {
	if(Shader::currentbound)
		Shader::current()->release();
}

ShaderManager::~ShaderManager() {
	for(QList<Shader*>::iterator it = createdShaders.begin(); it != createdShaders.end(); it++) {
		delete (*it);
	}
}

void ShaderManager::accepted() {
	for(QHash<QString, Shader*>::iterator it = shaders.begin(); it != shaders.end(); it++) {
		(*it)->accepted();
	}
}

bool ShaderManager::noOptions() {
	for(QHash<QString, Shader*>::iterator it = shaders.begin(); it != shaders.end(); it++) {
		if(!((*it)->noOptions()))
			return false;
	}
	return true;
}

QGroupBox* ShaderManager::getOptions() {
	if(shaders.isEmpty() || noOptions())
		return 0;

	if(options) {
		for(QList<Shader*>::iterator it = createdShaders.begin(); it != createdShaders.end(); it++) {
			(*it)->getOptions(); //set all the options up
		}
		return options;
	}
	options = new QGroupBox(tr("Shader Options"));
	QVBoxLayout* layout = new QVBoxLayout;
	for(QList<Shader*>::iterator it = createdShaders.begin(); it != createdShaders.end(); it++) {
		if(!(*it)->noOptions())
			layout->addWidget((*it)->getOptions());
	}
	options->setLayout(layout);
	return options;
}

void ShaderManager::saveSettings(QIODevice* ios) {
	int count = 0;
	for(QHash<QString,Shader*>::Iterator it = shaders.begin(); it != shaders.end(); ++it) {
		if(it.value()->getHasOptions()) {
			++count;
		}
	}
	QDataStream ds(ios);
	ds << count;
	for(QHash<QString,Shader*>::Iterator it = shaders.begin(); it != shaders.end(); ++it) {
		if(it.value()->getHasOptions()) {
			ds << it.key();
			it.value()->saveSettings(ios);
		}
	}
}

void ShaderManager::loadSettings(QIODevice* ios) {
	int count = 0;
	QDataStream ds(ios);
	ds >> count;
	QString name;
	for(int i = 0; i < count; ++i) {
		ds >> name;
		qDebug("%s", name.toAscii().data());
		if(!shaders.contains(name)) { //skip
			int ucount;
			ds >> ucount;
			for(int j = 0; j < ucount; ++j) {
				ds >> name;
				int skip;
				ios->seek(ios->pos() + 4);
				ios->read((char*)&skip, 4);
				ios->seek(ios->pos() + skip);
			}
		} else {
			shaders[name]->loadSettings(ios);
		}
	}
}

void ShaderManager::setOptions() {
	for(QList<Shader*>::iterator it = createdShaders.begin(); it != createdShaders.end(); it++) {
		(*it)->setOptions();
	}
}
