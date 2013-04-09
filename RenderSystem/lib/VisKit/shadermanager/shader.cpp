#include <GL/glew.h>
#include "shader.h"
#include <QtDebug>
#include <QFile>
#include <cstdarg>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QGridLayout>
#include <QLabel>

#define DEBUG
#include "glerror.h"

#include "shadergroupbox.h"
#include "gltexture.h"
#include "shaderslider.h"
#include "texturemanager.h"
#include <QCheckBox>

#define getName() ( display.isEmpty() ? name : display )


Shader* Shader::currentbound = 0;
bool Shader::alwaysUpdate = false;
bool Shader::matricesInitialized = false;
Matrix4x4 Shader::mv;
Matrix4x4 Shader::vp;
Matrix4x4 Shader::mvp;

QStack<Matrix4x4> Shader::modelStack;
QStack<Matrix4x4> Shader::viewStack;
QStack<Matrix4x4> Shader::projectionStack;

Shader::Shader(const QString& name):QObject(), options(0), display(name),
		uniforms(new QHash<QString, Uniform*>()), floats(new QHash<QString, UniformFloat*>()),
		floatarrays(new QHash<QString, UniformFloatArray*>()),
		ints(new QHash<QString, UniformInt*>()),
		intarrays(new QHash<QString, UniformIntArray*>()),
		matrices(new QHash<QString, UniformMatrix*>()),
		matrixarrays(new QHash<QString, UniformMatrixArray*>()),
		samplers(new QHash<QString, UniformSampler*>()),
		programParameters(new QHash<GLenum, GLint>()), own(true), hasVariables(false) {
	GLERROR(id = glCreateProgram());
}

void Shader::addVertexShader(const QString& filename) {
	shaderfiles.append(new ShaderFile(filename, GL_VERTEX_SHADER));
	GLERROR(glAttachShader(id, shaderfiles.back()->id));
}

void Shader::addGeometryShader(const QString& filename) {
	shaderfiles.append(new ShaderFile(filename, GL_GEOMETRY_SHADER));
	GLERROR(glAttachShader(id, shaderfiles.back()->id));
}


void Shader::addTessControlShader(const QString& filename) {
	shaderfiles.append(new ShaderFile(filename, GL_TESS_CONTROL_SHADER));
	GLERROR(glAttachShader(id, shaderfiles.back()->id));
}

void Shader::addTessEvalShader(const QString& filename) {
	shaderfiles.append(new ShaderFile(filename, GL_TESS_EVALUATION_SHADER));
	GLERROR(glAttachShader(id, shaderfiles.back()->id));
}

ShaderFile::ShaderFile(const QString& f, GLenum type):filename(f) {
	GLERROR(id = glCreateShader(type));
}

void Shader::addFragmentShader(const QString& filename) {
	shaderfiles.append(new ShaderFile(filename, GL_FRAGMENT_SHADER));
	GLERROR(glAttachShader(id, shaderfiles.back()->id));
}

void Shader::addFile(ShaderFile* file) {
	shaderfiles.append(file);
}

Uniform::Uniform(UniformType type, const QString& name, int numParams, const QString& displayName):
		userModifiable(false), name(name), display(displayName), numParams(numParams), options(0), uoptions(BOTH), steps(10), updated(true), type(type) {
	sliders = new QSlider*[numParams];
}

void Uniform::setFieldNames(const char* first, ...) {
	fields.clear();
	fields.push_back(tr(first));
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		fields.push_back(tr(va_arg(ap, const char*)));
	}
	va_end(ap);
}

QWidget* UniformFloat::getOptions() {
	if(!userModifiable)
		return 0;

	if(options) {
		return options;
	}

	options = new ShaderGroupBox(getName());
	QGridLayout *layout = new QGridLayout();
	for(int i = 0; i < numParams; i++) {
		if(fields.size() > i) {
			layout->addWidget(new QLabel(fields[i]), i, 0);
		}
		spins[i] = new QDoubleSpinBox();
		if(ranges[i*2] != ranges[i*2 + 1]) {
			spins[i]->setRange(ranges[2*i], ranges[2*i + 1]);
		}
		spins[i]->setValue(vars[i]);
		if(uoptions & SPINBOX)
			layout->addWidget(spins[i], i, 2);
		if(uoptions & SLIDER) {
			if(ranges[i*2] != ranges[i*2+1]) { //range required for slider
				sliders[i] = new ShaderSlider(ranges[i*2], ranges[i*2+1], steps);
				connect(qobject_cast<ShaderSlider*>(sliders[i]), SIGNAL(newValue(double)),
							spins[i], SLOT(setValue(double)));
				connect(spins[i], SIGNAL(valueChanged(double)),
						qobject_cast<ShaderSlider*>(sliders[i]), SLOT(setNewValue(double)));
				qobject_cast<ShaderSlider*>(sliders[i])->setNewValue(vars[i]);
				layout->addWidget(sliders[i], i, 1);
			}
		}
		if(interactive)
			connect(spins[i], SIGNAL(valueChanged(double)), this, SLOT(accepted()));
	}
	options->setLayout(layout);
	connect(options, SIGNAL(shown()),
			this, SLOT(setOptions()));
	return options;
}

void UniformInt::shareValues(UniformInt *slave) {
	if(slave->numParams != numParams) {
		qWarning("Error: Trying to share values between incompatible shaders");
		return;
	}
	connect(this, SIGNAL(valueChanged(int)), slave, SLOT(setValues()));
	slave->master = this;
	slaves.push_back(slave);
}

void UniformFloat::shareValues(UniformFloat *slave) {
	if(slave->numParams != numParams) {
		qWarning("Error: Trying to share values between incompatible shaders");
		return;
	}
	connect(this, SIGNAL(valueChanged(float)), slave, SLOT(setValues()));
	slave->master = this;
	slaves.push_back(slave);
}

void UniformInt::stopSharing() {
	if(master) {
		master->disconnect(this, SLOT(setValues()));
		master->slaves.removeAll(this);
		master = 0;
	}
	for(QList<UniformInt*>::iterator it = slaves.begin(); it != slaves.end(); ++it) {
		disconnect(*it, SLOT(setValues()));
		(*it)->stopSharing();
	}
}

void UniformFloat::stopSharing() {
	if(master) {
		master->disconnect(this, SLOT(setValues()));
		master->slaves.removeAll(this);
		master = 0;
	}
	for(QList<UniformFloat*>::iterator it = slaves.begin(); it != slaves.end(); ++it) {
		disconnect(*it, SLOT(setValues()));
		(*it)->stopSharing();
	}
}

void UniformFloat::setValues(float *v, int num) {
	if(num > numParams)
		num = numParams;
	for(int i = 0; i < num; ++i) {
		vars[i] = v[i];
	}
	updated = true;
	if(Shader::current()) {
		set(Shader::current()->getId());
	}
	switch(numParams) {
		case 1:
			emit valueChanged(vars[0]);
			break;
		case 2:
			emit valueChanged(vars[0], vars[1]);
			break;
		case 3:
			emit valueChanged(vars[0], vars[1], vars[2]);
			break;
		case 4:
			emit valueChanged(vars[0], vars[1], vars[1], vars[3]);
			break;
		default:
			emit valueChanged(vars[0]);
			qWarning("Error: Invalid number of parameters %s:%d", __FILE__, __LINE__);
	}

}

void UniformInt::setValues(int *v, int num) {
	if(num > numParams)
		num = numParams;
	for(int i = 0; i < num; ++i) {
		vars[i] = v[i];
	}
	updated = true;
	if(Shader::current()) {
		set(Shader::current()->getId());
	}
	switch(numParams) {
		case 1:
			emit valueChanged(vars[0]);
			break;
		case 2:
			emit valueChanged(vars[0], vars[1]);
			break;
		case 3:
			emit valueChanged(vars[0], vars[1], vars[2]);
			break;
		case 4:
			emit valueChanged(vars[0], vars[1], vars[1], vars[3]);
			break;
		default:
			emit valueChanged(vars[0]);
			qWarning("Error: Invalid number of parameters %s:%d", __FILE__, __LINE__);
	}

}

void UniformInt::setValues() {
	if(!master)
		return;

	setValues(master->vars, numParams);
}

void UniformFloat::setValues() {
	if(!master)
		return;

	setValues(master->vars, numParams);
}


QWidget* UniformInt::getOptions() {
	if(!userModifiable)
		return 0;

	if(options) {
		return options;
	}

	QLayout *l;
	if(numParams == 1 && (uoptions & CHECK)) {
		QVBoxLayout* layout = new QVBoxLayout;
		options = new ShaderGroupBox(QString());
		check = new QCheckBox();
		check->setText(getName());
		layout->addWidget(check);
		l = layout;
	} else {
		QGridLayout* layout = new QGridLayout();
		options = new ShaderGroupBox(getName());
		for(int i = 0; i < numParams; i++) {
			if(fields.size() > i) {
				layout->addWidget(new QLabel(fields[i]), i, 0);
			}
			spins[i] = new QSpinBox();
			if(ranges[i*2] != ranges[i*2 + 1]) {
				spins[i]->setRange(ranges[2*i], ranges[2*i + 1]);
			}
			if(uoptions & SPINBOX)
				layout->addWidget(spins[i], i, 2);

			if(uoptions & SLIDER) {
				if(ranges[i*2] != ranges[i*2+1]) {
					sliders[i] = new QSlider(Qt::Horizontal);
					sliders[i]->setRange(ranges[i*2], ranges[i*2 + 1]);
					connect(sliders[i], SIGNAL(sliderMoved(int)),
							spins[i], SLOT(setValue(int)));
					connect(spins[i], SIGNAL(valueChanged(int)),
							sliders[i], SLOT(setValue(int)));
					layout->addWidget(sliders[i], i, 1);
				}
			}
		}
		l = layout;
	}
	options->setLayout(l);
	connect(options, SIGNAL(shown()),
			this, SLOT(setOptions()));
	return options;
}

UniformSampler::UniformSampler(const QString &name, GLTexture *tex)
	:UniformInt(name, 1, 0), tex(tex) {
	type = SAMPLER;
	vars = new int[1];
	vars[0] = tex && tex->isBound() ? tex->getSlot() : 0;
	if(tex)
		connect(tex, SIGNAL(deleted()), this, SLOT(textureDeleted()));
}

void UniformSampler::set(GLuint shader) {
	if(tex) {
		if(!tex->isBound()) {
			TextureManager::getInstance()->bind(tex);
			updated = true;
		}
		vars[0] = tex->getSlot();
		//qDebug("%s: set to slot %d", name.toAscii().data(), vars[0]);
	}
	UniformInt::set(shader);
}

void UniformSampler::setTexture(GLTexture *texture) {
	if(tex) {
		disconnect(tex, SIGNAL(deleted()), this, SLOT(textureDeleted()));
	}
	tex = texture;
	if(!tex) {
		vars[0] = 0;
	} else {
		connect(tex, SIGNAL(deleted()), this, SLOT(textureDeleted()));
	}
	updated = true;
	if(tex && Shader::current()) {
		set(Shader::current()->getId());
	}
}

void UniformSampler::textureDeleted() {
	disconnect(tex, SIGNAL(deleted()), this, SLOT(textureDeleted()));
	tex = 0;
	updated = true;
}


void UniformInt::setValues(int first, ...) {
	vars[0] = first;
	updated = true;
	if(numParams == 1) {
		if(Shader::current()) {
			set(Shader::current()->getId());
		}
		return;
	}
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		vars[i] = va_arg(ap, int);
	}
	va_end(ap);

	if(Shader::current()) {
		set(Shader::current()->getId());
	}
	switch(numParams) {
		case 1:
			emit valueChanged(vars[0]);
			break;
		case 2:
			emit valueChanged(vars[0], vars[1]);
			break;
		case 3:
			emit valueChanged(vars[0], vars[1], vars[2]);
			break;
		case 4:
			emit valueChanged(vars[0], vars[1], vars[1], vars[3]);
			break;
		default:
			emit valueChanged(vars[0]);
			qWarning("Error: Invalid number of parameters %s:%d", __FILE__, __LINE__);
	}
}

void UniformFloat::setValues(float first, ...) {
	vars[0] = first;
	updated = true;
	if(numParams == 1) {
		if(Shader::current()) {
			set(Shader::current()->getId());
		}
		return;
	}
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		vars[i] = (float)va_arg(ap, double);
	}
	va_end(ap);
	if(Shader::current()) {
		set(Shader::current()->getId());
	}
	switch(numParams) {
		case 1:
			emit valueChanged(vars[0]);
			break;
		case 2:
			emit valueChanged(vars[0], vars[1]);
			break;
		case 3:
			emit valueChanged(vars[0], vars[1], vars[2]);
			break;
		case 4:
			emit valueChanged(vars[0], vars[1], vars[1], vars[3]);
			break;
		default:
			emit valueChanged(vars[0]);
			qWarning("Error: Invalid number of parameters %s:%d", __FILE__, __LINE__);
	}
}

void UniformFloatArray::addValues(float first, ...) {
	float t;
	t = first;
	updated = true;
	numSets++;
	vars.push_back(t);
	if(numParams == 1) {
		if(Shader::current()) {
			set(Shader::current()->getId());
		}
		return;
	}
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		vars.push_back((float)va_arg(ap, double));
	}
	va_end(ap);
	if(Shader::current()) {
		set(Shader::current()->getId());
	}
}

void UniformIntArray::addValues(int first, ...) {
	int t;
	t = first;
	updated = true;
	numSets++;
	vars.push_back(t);
	if(numParams == 1) {
		if(Shader::current()) {
			set(Shader::current()->getId());
		}
		return;
	}
	va_list ap;
	va_start(ap, first);
	for(int i = 1; i < numParams; i++) {
		vars.push_back(va_arg(ap, int));
	}
	va_end(ap);
	if(Shader::current()) {
		set(Shader::current()->getId());
	}
}

void UniformMatrixArray::addValues(Matrix4x4& m) {
	for (int j = 0; j < rows; ++j) {
		for (int i = 0; i < cols; ++i) {
			vars.push_back(m[i][j]);
		}
	}
	updated = true;
	numSets++;
	if(Shader::current()) {
		set(Shader::current()->getId());
	}
}

void UniformInt::accepted() {
	if(userModifiable) {
		if(numParams == 1 && (uoptions & CHECK)) {
			vars[0] = check->isChecked() ? 1 : 0;
		} else {
			for(int i = 0; i < numParams; i++) {
				vars[i] = spins[i]->value();
			}
		}
		updated = true;
	}
	switch(numParams) {
		case 1:
			emit valueChanged(vars[0]);
			break;
		case 2:
			emit valueChanged(vars[0], vars[1]);
			break;
		case 3:
			emit valueChanged(vars[0], vars[1], vars[2]);
			break;
		case 4:
			emit valueChanged(vars[0], vars[1], vars[1], vars[3]);
			break;
		default:
			emit valueChanged(vars[0]);
			qWarning("Error: Invalid number of parameters %s:%d", __FILE__, __LINE__);
	}
}

void UniformFloat::setOptions() {
	if(!userModifiable)
		return;

	for(int i = 0; i < numParams; i++) {
		spins[i]->setValue(vars[i]);
	}
}

void UniformInt::setOptions() {
	if(!userModifiable)
		return;

	if(numParams == 1 && (uoptions & CHECK)) {
		check->setChecked(vars[0]);
	} else {
		for(int i = 0; i < numParams; i++) {
			spins[i]->setValue(vars[i]);
		}
	}
}

void UniformFloat::accepted() {
	if(userModifiable) {
		for(int i = 0; i < numParams; i++) {
			vars[i] = (float)spins[i]->value();
		}
		updated = true;
	}
	switch(numParams) {
		case 1:
			emit valueChanged(vars[0]);
			break;
		case 2:
			emit valueChanged(vars[0], vars[1]);
			break;
		case 3:
			emit valueChanged(vars[0], vars[1], vars[2]);
			break;
		case 4:
			emit valueChanged(vars[0], vars[1], vars[1], vars[3]);
			break;
		default:
			emit valueChanged(vars[0]);
			qWarning("Error: Invalid number of parameters %s:%d", __FILE__, __LINE__);
	}
}

void UniformInt::setRanges(int first, ...) {
	va_list ap;
	va_start(ap, first);
	ranges[0] = first;
	ranges[1] = va_arg(ap, int);
	for(int i = 1; i < numParams; i++) {
		ranges[i*2] = va_arg(ap, int);
		ranges[i*2 + 1] = va_arg(ap, int);
	}
	va_end(ap);
}

void UniformFloat::setRanges(float first, ...) {
	va_list ap;
	va_start(ap, first);
	ranges[0] = first;
	ranges[1] = (float)va_arg(ap, double);
	for(int i = 1; i < numParams; i++) {
		ranges[i*2] = (float)va_arg(ap, double);
		ranges[i*2 + 1] = (float)va_arg(ap, double);
	}
	va_end(ap);
}

void UniformFloat::set(GLuint shader) {
	GLERROR(GLint loc = glGetUniformLocation(shader, name.toAscii().data()));
	if(loc == -1) {
		return;
	}
	//qDebug("%s, %d %f", name.toAscii().data(), numParams, vars[0]);
	switch(numParams) {
		case 1:
			GLERROR(glUniform1fv(loc, 1, vars));
			break;
		case 2:
			GLERROR(glUniform2fv(loc, 1, vars));
			break;
		case 3:
			GLERROR(glUniform3fv(loc, 1, vars));
			break;
		case 4:
			GLERROR(glUniform4fv(loc, 1, vars));
			break;
		default:
			qDebug("Invaid shader parameter size (%s).", name.toAscii().data());
	}
	updated = false;
}

void UniformFloatArray::set(GLuint shader) {
	if(!numSets)
		return;
	GLERROR(GLint loc = glGetUniformLocation(shader, name.toAscii().data()));
	if(loc == -1) {
		return;
	}
	//qDebug("%s, %d %f", name.toAscii().data(), numParams, vars[0]);
	switch(numParams) {
		case 1:
			GLERROR(glUniform1fv(loc, numSets, vars.data()));
			break;
		case 2:
			GLERROR(glUniform2fv(loc, numSets, vars.data()));
			break;
		case 3:
			GLERROR(glUniform3fv(loc, numSets, vars.data()));
			break;
		case 4:
			GLERROR(glUniform4fv(loc, numSets, vars.data()));
			break;
		default:
			qDebug("Invaid shader parameter size (%s).", name.toAscii().data());
	}
	updated = false;
}

void UniformIntArray::set(GLuint shader) {
	if(!numSets)
		return;
	GLERROR(GLint loc = glGetUniformLocation(shader, name.toAscii().data()));
	if(loc == -1) {
		return;
	}
	//qDebug("%s, %d %f", name.toAscii().data(), numParams, vars[0]);
	switch(numParams) {
		case 1:
			GLERROR(glUniform1iv(loc, numSets, vars.data()));
			break;
		case 2:
			GLERROR(glUniform2iv(loc, numSets, vars.data()));
			break;
		case 3:
			GLERROR(glUniform3iv(loc, numSets, vars.data()));
			break;
		case 4:
			GLERROR(glUniform4iv(loc, numSets, vars.data()));
			break;
		default:
			qDebug("Invaid shader parameter size (%s).", name.toAscii().data());
	}
	updated = false;

}

void UniformMatrixArray::set(GLuint shader) {
	if(!numSets)
		return;
	GLERROR(GLint loc = glGetUniformLocation(shader, name.toAscii().data()));
	if(loc == -1) {
		return;
	}
	if(cols == 2) {
		if(rows == 2) {
			glUniformMatrix2fv(loc, numSets, false, vars.data());
		} else if(rows == 3) {
			glUniformMatrix2x3fv(loc, numSets, false, vars.data());
		} else if(rows == 4) {
			glUniformMatrix2x4fv(loc, numSets, false, vars.data());
		} else {
			qDebug("Bad Matrix Size");
		}
	} else if(cols == 3) {
		if(rows == 2) {
			glUniformMatrix3x2fv(loc, numSets, false, vars.data());
		} else if(rows == 3) {
			glUniformMatrix3fv(loc, numSets, false, vars.data());
		} else if(rows == 4) {
			glUniformMatrix3x4fv(loc, numSets, false, vars.data());
		} else {
			qDebug("Bad Matrix Size");
		}
	} else if(cols == 4) {
		if(rows == 2) {
			glUniformMatrix4x2fv(loc, numSets, false, vars.data());
		} else if(rows == 3) {
			glUniformMatrix4x3fv(loc, numSets, false, vars.data());
		} else if(rows == 4) {
			glUniformMatrix4fv(loc, numSets, false, vars.data());
		} else {
			qDebug("Bad Matrix Size");
		}
	} else {
		qDebug("Bad Matrix Size");
	}
	updated = false;
}

void Shader::addUniformi(const QString& name, int numParams, ...) {
	va_list ap;
	va_start(ap, numParams);
	int* params = new int[numParams];
	for(int i = 0; i < numParams; i++) {
		params[i] = va_arg(ap, int);
	}
	va_end(ap);

	if((*ints)[name])
		delete (*ints)[name];
	if(samplers->contains(name)) {
		samplers->remove(name);
	}
	(*ints)[name] = new UniformInt(name, numParams, params);
	(*uniforms)[name] = (*ints)[name];
}


void Shader::addUniformf(const QString& name, int numParams, ...) {
	va_list ap;
	va_start(ap, numParams);
	float* params = new float[numParams];
	for(int i = 0; i < numParams; i++) {
		params[i] = (float)va_arg(ap, double);
	}
	va_end(ap);

	if((*floats)[name])
		delete (*floats)[name];
	(*floats)[name] = new UniformFloat(name, numParams, params);
	(*uniforms)[name] = (*floats)[name];
}

void UniformInt::set(GLuint shader) {
	GLERROR(GLint loc = glGetUniformLocation(shader, name.toAscii().data()));
	if(loc == -1) {
		return;
	}
	switch(numParams) {
		case 1:
			GLERROR(glUniform1iv(loc, 1, (const GLint*)vars));
			break;
		case 2:
			GLERROR(glUniform2iv(loc, 1, (const GLint*)vars));
			break;
		case 3:
			GLERROR(glUniform3iv(loc, 1, (const GLint*)vars));
			break;
		case 4:
			GLERROR(glUniform4iv(loc, 1, (const GLint*)vars));
			break;
		default:
			qDebug("Invalid shader parameter size (%s).", name.toAscii().data());
	}
	updated = false;
}

void ShaderFile::checkShader() {
	int p;
	GLERROR(glGetShaderiv(id, GL_COMPILE_STATUS, (GLint*)&p));
	if(!p) {
		qDebug("Compile failed for: %s", filename.toAscii().data());
	}
	GLERROR(glGetShaderiv(id, GL_INFO_LOG_LENGTH, (GLint*)&p));
	if(p > 1) {
		char* infolog = new char[p];
		int l;
		GLERROR(glGetShaderInfoLog(id, 65535, (GLsizei*)&l, infolog));
		qDebug("%s", infolog);
		delete [] infolog;

	}
}

void ShaderFile::readFile() {
	QFile file(filename);
	if(!file.exists())
		qDebug("Error: shader %s does not exist", filename.toAscii().data());
	file.open(QIODevice::ReadOnly);
	int length = file.size();

	char* s = new char[length];

	file.read(s, file.size());
	file.close();
	GLERROR(glShaderSource(id, 1, (const char**)&s, (const GLint*)&length));
	delete [] s;
}

void Shader::checkProgram() {
	int p;
	GLERROR(glGetProgramiv(id, GL_LINK_STATUS, (GLint*)&p));
	if(!p) {
		qWarning("Link failed for program %s", display.toAscii().data());
	}
	GLERROR(glGetProgramiv(id, GL_INFO_LOG_LENGTH, (GLint*)&p));
	if(p > 1) {
		char* infolog = new char[p];
		int l;
		GLERROR(glGetProgramInfoLog(id, 65535, (GLsizei*)&l, infolog));
		qDebug("%s", infolog);
		delete [] infolog;

	}
	glValidateProgram(id);
	GLERROR(glGetProgramiv(id, GL_VALIDATE_STATUS, (GLint*)&p));
	if(!p) {
		qWarning("Validate failed for program %s", display.toAscii().data());
		GLERROR(glGetProgramiv(id, GL_INFO_LOG_LENGTH, (GLint*)&p));
		if(p > 1) {
			char* infolog = new char[p];
			int l;
			GLERROR(glGetProgramInfoLog(id, 65535, (GLsizei*)&l, infolog));
			qDebug("%s", infolog);
			delete [] infolog;

		}
	}
}

void Shader::addUniformSampler(const QString &name, GLTexture *tex) {
	if((*uniforms).contains(name))
		delete ((*uniforms)[name]);
	UniformSampler* newsampler = new UniformSampler(name, tex);
	(*uniforms)[name] = newsampler;
	(*ints)[name] = newsampler;
	(*samplers)[name] = newsampler;
}


void Shader::release() {
	GLERROR(glUseProgram(0));
	currentbound = 0;
}

void Shader::use() {
	currentbound = this;
	GLERROR(glUseProgram(id));
	setUniforms();
	setDefaultMatrices();
}

void Shader::compileAndLink() {
	for(QList<ShaderFile*>::iterator it = shaderfiles.begin(); it != shaderfiles.end(); it++) {
		(*it)->compile();
	}
	for(QHash<GLenum, GLint>::iterator it = programParameters->begin(); it != programParameters->end(); it++) {
		GLERROR(glProgramParameteriEXT(id, it.key(), it.value()));
	}

	char* buf;
	int size;
	GLenum type;
	int active, maxlen;

	GLERROR(glLinkProgram(id));

	GLERROR(glGetProgramiv(id, GL_ACTIVE_ATTRIBUTES, &active));
	GLERROR(glGetProgramiv(id, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &maxlen));
	buf = new char[maxlen + 1];

	for(int i = 0; i < active; ++i) {
		GLERROR(glGetActiveAttrib(id, i, maxlen+1, 0, &size, &type, buf));
		if(attribs.contains(buf) || (buf[0] == 'g' && buf[1] == 'l'))
			continue;
		addAttrib(buf);
	}

	delete [] buf;



	GLuint t = 1;
	for(QStringList::iterator it = attribs.begin(); it != attribs.end(); ++it) {
		GLERROR(glBindAttribLocation(id, t, (*it).toAscii().data()));
		//qDebug("binding %s to slot %d", (*it).toAscii().data(), t);
		++t;
	}
	GLERROR(glLinkProgram(id));
	checkProgram();


	GLERROR(glGetProgramiv(id, GL_ACTIVE_UNIFORMS, &active));
	GLERROR(glGetProgramiv(id, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxlen));
	buf = new char[maxlen + 1];

	for(int i = 0; i < active; ++i) {
		GLERROR(glGetActiveUniform(id, i, maxlen+1, 0, &size, &type, buf));
		//check to see if we already have this one or it's a builtin
		if(uniforms->contains(buf) || (buf[0] == 'g' && buf[1] == 'l'))
			continue;

		//qDebug("Uniform: %s", buf);

		//add the uniform
		switch(type) {
			case GL_FLOAT:
				addUniformf(buf, 1, 0.f);
				break;
			case GL_FLOAT_VEC2:
				addUniformf(buf, 2, 0.f, 0.f);
				break;
			case GL_FLOAT_VEC3:
				addUniformf(buf, 3, 0.f, 0.f, 0.f);
				break;
			case GL_FLOAT_VEC4:
				addUniformf(buf, 4, 0.f, 0.f, 0.f, 0.f);
				break;
			case GL_INT:
			case GL_BOOL:
				addUniformi(buf, 1, 0);
				break;
			case GL_SAMPLER_1D:
			case GL_SAMPLER_2D:
			case GL_SAMPLER_3D:
			case GL_SAMPLER_CUBE:
			case GL_SAMPLER_1D_SHADOW:
			case GL_SAMPLER_2D_SHADOW:
				addUniformSampler(buf, 0);
				break;
			case GL_INT_VEC2:
			case GL_BOOL_VEC2:
				addUniformi(buf, 2, 0, 0);
				break;
			case GL_INT_VEC3:
			case GL_BOOL_VEC3:
				addUniformi(buf, 3, 0, 0, 0);
				break;
			case GL_INT_VEC4:
			case GL_BOOL_VEC4:
				addUniformi(buf, 4, 0, 0, 0, 0);
				break;
			case GL_FLOAT_MAT2:
				addUniformMatrix(buf, 2, 2);
				break;
			case GL_FLOAT_MAT3:
				addUniformMatrix(buf, 3, 3);
				break;
			case GL_FLOAT_MAT4:
				addUniformMatrix(buf, 4, 4);
				break;
			case GL_FLOAT_MAT2x3:
				addUniformMatrix(buf, 2, 3);
				break;
			case GL_FLOAT_MAT2x4:
				addUniformMatrix(buf, 2, 4);
				break;
			case GL_FLOAT_MAT3x2:
				addUniformMatrix(buf, 3, 2);
				break;
			case GL_FLOAT_MAT3x4:
				addUniformMatrix(buf, 3, 4);
				break;
			case GL_FLOAT_MAT4x2:
				addUniformMatrix(buf, 4, 2);
				break;
			case GL_FLOAT_MAT4x3:
				addUniformMatrix(buf, 4, 3);
				break;
			default:
				//wtf
				break;
		}
	}
	delete [] buf;

	GLERROR(glGetProgramiv(id, GL_ACTIVE_ATTRIBUTES, &active));
	GLERROR(glGetProgramiv(id, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &maxlen));
	buf = new char[maxlen + 1];

	attribIndices.clear();
	for(int i = 0; i < active; ++i) {
		GLERROR(glGetActiveAttrib(id, i, maxlen+1, 0, &size, &type, buf));
		if(!attribs.contains(buf))
			continue;
		attribIndices[buf] = glGetAttribLocation(id, buf);
		//qDebug("Attrib %s set to slot: %d", buf, attribIndices[buf]);
	}

	delete [] buf;

	Shader* lastbound = currentbound;
	use();
	setUniforms(true);
	if(lastbound) {
		lastbound->use();
	} else {
		release();
	}
//	GLERROR();
}

bool Shader::isBound() {
	return currentbound == this;
}

void Shader::setUniforms(bool force) {
	for(QHash<QString, Uniform*>::iterator it = uniforms->begin(); it != uniforms->end(); it++) {
		if((*it)->hasUpdate() || force || alwaysUpdate) {
			(*it)->set(id);
		}
	}

}

void Shader::addProgramParameter(GLenum pname, GLint value) {
	(*programParameters)[pname] = value;
}

ShaderFile& ShaderFile::operator=(const ShaderFile& rhs) {
	id = rhs.id;
	filename = rhs.filename;
	type = rhs.type;
	return *this;
}

Shader& Shader::operator=(const Shader& rhs) {
	id = rhs.id;
	shaderfiles = rhs.shaderfiles;
	return *this;
}

void ShaderFile::compile() {
	readFile();
	GLERROR(glCompileShader(id));
	checkShader();
}

ShaderFile::ShaderFile(const ShaderFile& orig): filename(orig.filename),
	id(orig.id), type(orig.type) {
}


Shader::~Shader() {
	for(QList<ShaderFile*>::iterator it = shaderfiles.begin(); it != shaderfiles.end(); it++) {
		delete (*it);
	}
	if(!own)
		return;
	for(QHash<QString, Uniform*>::iterator it = uniforms->begin(); it != uniforms->end(); it++) {
		delete (*it);
	}
	delete uniforms;
	delete programParameters;
	delete floats;
	delete matrices;
	delete floatarrays;
	delete ints;
	delete samplers;
}

void Shader::accepted() {
	for(QHash<QString, Uniform*>::iterator it = uniforms->begin(); it != uniforms->end(); it++) {
		(*it)->accepted();
	}
}

void Shader::setName(const QString& name) {
	display = name;
}

bool Shader::noOptions() {
	if(!hasVariables)
		return true;
	for(QHash<QString, Uniform*>::iterator it = uniforms->begin(); (it != uniforms->end()); it++) {
		if((*it)->isModifiable())
			return false;
	}
	return true;
}

QWidget* Shader::getOptions() {
	if(uniforms->isEmpty() || noOptions())
		return 0;

	if(options) {
		return options;
	}
	options = new QGroupBox(display);
	QVBoxLayout* layout = new QVBoxLayout();
	for(QHash<QString, Uniform*>::iterator it = uniforms->begin(); it != uniforms->end(); it++) {
		if((*it)->isModifiable())
			layout->addWidget((*it)->getOptions()); //set all the values
	}
	options->setLayout(layout);
	return options;
}

Uniform* Shader::operator[](const QString& index) {
	if(uniforms->contains(index))
		return (*uniforms)[index];
	return 0;
}

void Shader::addAttrib(const QString& name) {
	if(!attribs.contains(name)) {
		attribs.append(name);
		//qDebug("added attrib: %s", name.toAscii().data());
	}
}

void Shader::setAttribute(const QString& name, int numParams, ...) {
	va_list ap;
	va_start(ap, numParams);
	float t[4];
	for(int i = 0; i < numParams; i++) {
		t[i] = (float)va_arg(ap, double);
	}
	va_end(ap);
//	GLERROR(GLint loc = glGetAttribLocation(id, name.toAscii().data()));

	if(getAttribLocation(name) == -1) {
#ifdef DEBUG
		//qDebug("Warning: Tried to set inactive attribute %s", name.toAscii().data());
#endif
		return;
	}
	GLint loc = getAttribLocation(name);
	switch(numParams) {
		case 1:
			GLERROR(glVertexAttrib1f(loc, t[0]));
			break;
		case 2:
			GLERROR(glVertexAttrib2f(loc, t[0], t[1]));
			break;
		case 3:
			GLERROR(glVertexAttrib3f(loc, t[0], t[1], t[2]));
			break;
		case 4:
			GLERROR(glVertexAttrib4f(loc, t[0], t[1], t[2], t[3]));
			break;
		default:
			qDebug("error: invalid number of parameters!");
	}
	//delete [] t;
}

GLint Shader::getAttribLocation(const QString& name) const {
	//return glGetAttribLocation(id, name.toAscii().data());
	return attribIndices.contains(name) ? attribIndices[name] : -1;
}

void Shader::shareUniforms(Shader* other) {
	if(own) {
		delete uniforms;
		delete floats;
		delete ints;
		delete programParameters;
		delete floatarrays;
		delete samplers;
	}
	own = false;
	uniforms = other->uniforms;
	programParameters = other->programParameters;
	floats = other->floats;
	ints = other->ints;
	floatarrays = other->floatarrays;
	matrices = other->matrices;
	samplers = other->samplers;
}

void Shader::addUniformMatrix(const QString &name, int cols, int rows) {
	if(matrices->contains(name))
		delete (*matrices)[name];
	(*matrices)[name] = new UniformMatrix(name, cols, rows);
	(*uniforms)[name] = (*matrices)[name];
}

void Shader::addUniformFloatArray(const QString &name, int numParams) {
	if(floatarrays->contains(name))
		delete (*floatarrays)[name];
	(*floatarrays)[name] = new UniformFloatArray(name, numParams);
	(*uniforms)[name] = (*floatarrays)[name];
}

void Shader::addUniformIntArray(const QString &name, int numParams) {
	if(intarrays->contains(name))
		delete (*intarrays)[name];
	(*intarrays)[name] = new UniformIntArray(name, numParams);
	(*uniforms)[name] = (*intarrays)[name];
}

void Shader::addUniformMatrixArray(const QString &name, int cols, int rows) {
	if(matrixarrays->contains(name))
		delete (*matrixarrays)[name];
	(*matrixarrays)[name] = new UniformMatrixArray(name, cols, rows);
	(*uniforms)[name] = (*matrixarrays)[name];
}



void UniformMatrix::set(GLuint shader) {
	GLERROR(GLint loc = glGetUniformLocation(shader, name.toAscii().data()));
	if(loc == -1) {
		return;
	}
	if(cols == 2) {
		if(rows == 2) {
			GLERROR(glUniformMatrix2fv(loc, 1, false, vars));
		} else if(rows == 3) {
			GLERROR(glUniformMatrix2x3fv(loc, 1, false, vars));
		} else if(rows == 4) {
			GLERROR(glUniformMatrix2x4fv(loc, 1, false, vars));
		} else {
			qDebug("Bad Matrix Size");
		}
	} else if(cols == 3) {
		if(rows == 2) {
			GLERROR(glUniformMatrix3x2fv(loc, 1, false, vars));
		} else if(rows == 3) {
			GLERROR(glUniformMatrix3fv(loc, 1, false, vars));
		} else if(rows == 4) {
			GLERROR(glUniformMatrix3x4fv(loc, 1, false, vars));
		} else {
			qDebug("Bad Matrix Size");
		}
	} else if(cols == 4) {
		if(rows == 2) {
			GLERROR(glUniformMatrix4x2fv(loc, 1, false, vars));
		} else if(rows == 3) {
			GLERROR(glUniformMatrix4x3fv(loc, 1, false, vars));
		} else if(rows == 4) {
			GLERROR(glUniformMatrix4fv(loc, 1, false, vars));
		} else {
			qDebug("Bad Matrix Size");
		}
	} else {
		qDebug("Bad Matrix Size");
	}
	updated = false;

}

void UniformMatrix::setValues(const Matrix4x4& m) {
	m.set(vars, cols, rows);
	updated = true;
	if(Shader::current()) {
		set(Shader::current()->getId());
	}
}

void Shader::trySetUniformf(const QString& name, int numParams, ...) {
	va_list ap;
	va_start(ap, numParams);
	float* t = new float[numParams];
	for(int i = 0; i < numParams; i++) {
		t[i] = (float)va_arg(ap, double);
	}
	va_end(ap);

	if(!current() || !(current()->getFloats().contains(name))) {
		delete [] t;
		return;
	}
	for(int i = 0; i < numParams; ++i) {
		current()->getFloats()[name]->v()[i] = t[i];
		current()->getFloats()[name]->set(current()->getId());
	}
	delete [] t;

}

void Shader::trySetUniformi(const QString& name, int numParams, ...) {
	va_list ap;
	va_start(ap, numParams);
	float* t = new float[numParams];
	for(int i = 0; i < numParams; i++) {
		t[i] = va_arg(ap, int);
	}
	va_end(ap);

	if(!current() || !(current()->getInts().contains(name))) {
		delete [] t;
		return;
	}
	for(int i = 0; i < numParams; ++i) {
		current()->getInts()[name]->v()[i] = t[i];
		current()->getInts()[name]->set(current()->getId());
	}
	delete [] t;

}


void Shader::trySetUniformMatrix(const QString& name, const Matrix4x4& mat) {
	if(!current() || !(current()->getMatrices().contains(name))) {
		return;
	}
	current()->getMatrices()[name]->setValues(mat);

}

void UniformFloat::setInteractive(bool i) {
	interactive = i;
	if(options) {
		if(interactive) {
			for(int i = 0; i < numParams; ++i) {
				connect(spins[i], SIGNAL(valueChanged(double)), this, SLOT(accepted()));
			}
		} else {
			for(int i = 0; i < numParams; ++i)
				disconnect(spins[i], SIGNAL(valueChanged(double)), this, SLOT(accepted()));
		}

	}
}

void Shader::updateMatrices() {
	mv = viewStack.top()*modelStack.top();
	vp = projectionStack.top()*viewStack.top();
	mvp = projectionStack.top()*viewStack.top()*modelStack.top();
	setDefaultMatrices();
}

void Shader::initMatrices() {
	if(matricesInitialized)
		return;
	matricesInitialized = true;

	modelStack.push(Matrix4x4::identity);
	viewStack.push(Matrix4x4::identity);
	projectionStack.push(Matrix4x4::identity);
	setDefaultMatrices();
}

void Shader::setMatrix(QStack<Matrix4x4> &stack, const Matrix4x4& m) {
	if(!matricesInitialized)
		initMatrices();

	//an 'empty' stack
	if(stack.size() == 1) {
		stack.push(m);
	} else {
		stack.top() = m;
	}

	updateMatrices();
}

void Shader::pushMatrix(QStack<Matrix4x4> &stack, const Matrix4x4 &m) {
	if(!matricesInitialized)
		initMatrices();

	stack.push(m);

	updateMatrices();
}

void Shader::popMatrix(QStack<Matrix4x4> &stack) {
	if(!matricesInitialized)
		initMatrices();
	if(stack.size() < 2)
		return;

	stack.pop();

	updateMatrices();
}

void Shader::setModel(const Matrix4x4 &m) {
	setMatrix(modelStack, m);
}

void Shader::setProjection(const Matrix4x4 &m) {
	setMatrix(projectionStack, m);
}

void Shader::setView(const Matrix4x4 &m) {
	setMatrix(viewStack, m);
}

void Shader::pushModel(const Matrix4x4 &m) {
	pushMatrix(modelStack, m);
}

void Shader::pushView(const Matrix4x4 &m) {
	pushMatrix(viewStack, m);
}

void Shader::pushProjection(const Matrix4x4 &m) {
	pushMatrix(projectionStack, m);
}

void Shader::popProjection() {
	popMatrix(projectionStack);
}

void Shader::popView() {
	popMatrix(viewStack);
}

void Shader::popModel() {
	popMatrix(modelStack);
}

const Matrix4x4& Shader::getTop(QStack<Matrix4x4> &stack) {
	if(!matricesInitialized)
		initMatrices();

	return stack.top();
}

const Matrix4x4& Shader::getModel() {
	return getTop(modelStack);
}

const Matrix4x4& Shader::getView() {
	return getTop(viewStack);
}

const Matrix4x4& Shader::getProjection() {
	return getTop(projectionStack);
}

const Matrix4x4& Shader::getModelView() {
	return mv;
}

const Matrix4x4& Shader::getModelViewProjection() {
	return mvp;
}

const Matrix4x4& Shader::getViewProjection() {
	return vp;
}

void Shader::setDefaultMatrices() {
	trySetUniformMatrix("model", getModel());
	trySetUniformMatrix("modelView", getModelView());
	trySetUniformMatrix("view", getView());
	trySetUniformMatrix("viewProjection", getViewProjection());
	trySetUniformMatrix("projection", getProjection());
	trySetUniformMatrix("modelViewProjection", getModelViewProjection());
}

void UniformFloat::saveSettings(QIODevice* ios) {
	ios->write((char*)&type, sizeof(UniformType));
	int size = numParams*4 + 4;
	ios->write((char*)&size, 4);
	ios->write((char*)&numParams, 4);
	ios->write((char*)vars, numParams*4);
}
void UniformFloat::loadSettings(QIODevice* ios) {
	int params = 0;
	ios->read((char*)&params, 4);
	if(params != numParams)
		qFatal("Error: Attempted to load bad settings");
	ios->read((char*)vars, numParams*4);
	updated = true;
	setOptions();
	switch(numParams) {
	case 1:
		emit valueChanged(vars[0]);
		break;
	case 2:
		emit valueChanged(vars[0], vars[1]);
		break;
	case 3:
		emit valueChanged(vars[0], vars[1], vars[2]);
		break;
	case 4:
		emit valueChanged(vars[0], vars[1], vars[1], vars[3]);
		break;
	default:
		emit valueChanged(vars[0]);
		qWarning("Error: Invalid number of parameters %s:%d", __FILE__, __LINE__);
	}
}
void UniformInt::saveSettings(QIODevice* ios) {
	ios->write((char*)&type, sizeof(UniformType));
	int size = numParams*4 + 4;
	ios->write((char*)&size, 4);
	ios->write((char*)&numParams, 4);
	ios->write((char*)vars, numParams*4);
}
void UniformInt::loadSettings(QIODevice* ios) {
	int params;
	ios->read((char*)&params, 4);
	if(params != numParams)
		qFatal("Error: Attempted to load bad settings");
	ios->read((char*)vars, numParams*4);
	updated = true;
	setOptions();
	switch(numParams) {
	case 1:
		emit valueChanged(vars[0]);
		break;
	case 2:
		emit valueChanged(vars[0], vars[1]);
		break;
	case 3:
		emit valueChanged(vars[0], vars[1], vars[2]);
		break;
	case 4:
		emit valueChanged(vars[0], vars[1], vars[1], vars[3]);
		break;
	default:
		emit valueChanged(vars[0]);
		qWarning("Error: Invalid number of parameters %s:%d", __FILE__, __LINE__);
	}
}

void UniformFloatArray::saveSettings(QIODevice* ios) {
	ios->write((char*)&type, sizeof(UniformType));
	int size = numSets*4 + 4;
	ios->write((char*)&size, 4);
	ios->write((char*)&numSets, 4);
	ios->write((char*)vars.data(), numSets*4);
}
void UniformFloatArray::loadSettings(QIODevice* ios) {
	int params;
	ios->read((char*)&params, 4);
	if(params != numSets)
		qFatal("Error: Attempted to load bad settings");
	ios->read((char*)vars.data(), numSets*4);
	updated = true;
}
void UniformIntArray::saveSettings(QIODevice* ios) {
	ios->write((char*)&type, sizeof(UniformType));
	int size = numSets*4 + 4;
	ios->write((char*)&size, 4);
	ios->write((char*)&numSets, 4);
	ios->write((char*)vars.data(), numSets*4);
}
void UniformIntArray::loadSettings(QIODevice* ios) {
	int params;
	ios->read((char*)&params, 4);
	if(params != numSets)
		qFatal("Error: Attempted to load bad settings");
	ios->read((char*)vars.data(), numSets*4);
	updated = true;
}

void UniformSampler::loadSettings(QIODevice*) {
}
void UniformSampler::saveSettings(QIODevice*) {
}

void UniformMatrix::saveSettings(QIODevice* ios) {
	ios->write((char*)&type, sizeof(UniformType));
	int size = rows*cols*4 + 8;
	ios->write((char*)&size, 4);
	ios->write((char*)&rows, 4);
	ios->write((char*)&cols, 4);
	ios->write((char*)vars, rows*cols*4);
}
void UniformMatrix::loadSettings(QIODevice* ios) {
	int nrows, ncols;
	ios->read((char*)&nrows, 4);
	ios->read((char*)&ncols, 4);
	if(nrows != rows || ncols != cols)
		qFatal("Error: Attempted to load bad settings");
	ios->read((char*)vars, rows*cols*4);
	updated = true;
}

void UniformMatrixArray::saveSettings(QIODevice* ios) {
	ios->write((char*)&type, sizeof(UniformType));
	int size = rows*cols*4 + 8;
	ios->write((char*)&size, 4);
	ios->write((char*)&rows, 4);
	ios->write((char*)&cols, 4);
	ios->write((char*)&numSets, 4);
	ios->write((char*)vars.data(), numSets*rows*cols*4);
}
void UniformMatrixArray::loadSettings(QIODevice* ios) {
	int nrows, ncols, nSets;
	ios->read((char*)&nrows, 4);
	ios->read((char*)&ncols, 4);
	ios->read((char*)&nSets, 4);
	if(nrows != rows || ncols != cols || nSets != numSets)
		qFatal("Error: Attempted to load bad settings");
	ios->read((char*)vars.data(), numSets*rows*cols*4);
	updated = true;
}

void Shader::loadSettings(QIODevice* ios) {
	if(!own)
		return;
	QDataStream ds(ios);
	QString uniform;
	int numUniforms, size;
	Uniform::UniformType type;
	ds >> numUniforms;
	for(int i = 0; i < numUniforms; ++i) {
		ds >> uniform;
		ios->read((char*)&type, sizeof(Uniform::UniformType));
		ios->read((char*)&size, 4);
		if(uniforms->contains(uniform) && type == (*uniforms)[uniform]->getType()) {
			(*uniforms)[uniform]->loadSettings(ios);
		} else {
			ios->seek(ios->pos() + size);
		}
	}
}

void Shader::saveSettings(QIODevice* ios) {
	if(!own)
		return;
	int numUnis=0;
	for(QHash<QString,Uniform*>::iterator it = uniforms->begin(); it != uniforms->end(); ++it) {
		if(it.value()->isModifiable())
			++numUnis;
	}
	QDataStream ds(ios);
	ds << numUnis;
	for(QHash<QString,Uniform*>::iterator it = uniforms->begin(); it != uniforms->end(); ++it) {
		if(it.value()->isModifiable()) {
			ds << it.key();
			it.value()->saveSettings(ios);
		}
	}
}

void Shader::setOptions() {
	if(!own)
		return;
	for(QHash<QString,Uniform*>::iterator it = uniforms->begin(); it != uniforms->end(); ++it) {
		(*it)->setOptions();
	}
}

