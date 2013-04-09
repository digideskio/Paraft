#ifndef _SHADER_H_
#define _SHADER_H_

#include <GL/glew.h>

#include <QHash>
#include <QString>
#include <QList>
#include <QStringList>
#include <QObject>
#include <QVector>
#include <QStack>

#include "../camera/matrices.h"

class QSpinBox;
class QDoubleSpinBox;
class QGroupBox;
class QSlider;
class QCheckBox;
class QWidget;
class ShaderGroupBox;

class ShaderFile {
	QString filename;
	GLuint id;
	GLenum type;
	void checkShader();
	void readFile();
	public:
		ShaderFile(const QString&, GLenum);
		ShaderFile(const ShaderFile&);
		void compile();
	ShaderFile& operator=(const ShaderFile& rhs);
	friend class Shader;
};


class Uniform : public QObject {
	Q_OBJECT
	public:
		enum UniformOption { SPINBOX = 1, SLIDER = 2, BOTH = 3, CHECK = 4 };
		enum UniformType { FLOAT, INT, SAMPLER, MATRIX, FLOAT_ARRAY, INT_ARRAY, MATRIX_ARRAY };

	protected:
		bool userModifiable;
		QString name;
		QString display;
		QStringList fields;
		int numParams;
		ShaderGroupBox* options;
		UniformOption uoptions;
		QSlider** sliders;
		int steps;
		bool updated;
		bool interactive;
		UniformType type;
	public:
		Uniform(UniformType type, const QString& name, int numParams, const QString& displayName = QString());
		const QString& getName() const { return name; }
		void setModifiable(bool mod) { userModifiable = mod; }
		void setFieldNames(const char* first, ...);
		bool isModifiable() { return userModifiable; }
		void setDisplayType(UniformOption type) { uoptions = type; }
		void setDisplayName(const QString& displayName) { display = displayName; }
		void setSteps(int s) { steps = s; }
		virtual void setInteractive(bool) {}
		virtual void set(GLuint shader)=0;
		virtual QWidget* getOptions()=0;
		virtual void accepted()=0;
		bool hasUpdate() const { return updated; }
		UniformType getType() const { return type; }
		virtual ~Uniform() {
			delete [] sliders;
		}
		virtual void loadSettings(QIODevice* ios)=0;
		virtual void saveSettings(QIODevice* ios)=0;
	public slots:
		virtual void setValues() {}
		virtual void setOptions() {}
};

class UniformFloat : public Uniform {
	Q_OBJECT

	float* vars;
	float* ranges;
	UniformFloat* master;
	QList<UniformFloat*> slaves;
	QDoubleSpinBox** spins;
	public:
		UniformFloat(const QString& name, int numParams, float* vars, const QString& displayName = QString()):
			Uniform(FLOAT, name, numParams, displayName), vars(vars), master(0) {
			ranges = new float[numParams*2];
			for(int i = 0; i < numParams*2; i++) {
				ranges[i] = 0;
			}

			spins = new QDoubleSpinBox*[numParams];
			for(int i = 0; i < numParams; i++) {
				spins[i] = 0;
			}
		}
		void setRanges(float first, ...);
		~UniformFloat() {
			stopSharing();
			delete [] vars;
			delete [] ranges;
			delete [] spins; //the actual spin boxes are taken care of by Qt
		}
		float* v() {
			updated = true;
			return vars;
		}
		QWidget* getOptions();
		void set(GLuint shader);
		void setValues(float first, ...);
		void setValues(float* v, int num);
		void setInteractive(bool);
		virtual void loadSettings(QIODevice* ios);
		virtual void saveSettings(QIODevice* ios);
		void shareValues(UniformFloat* slave);
		void stopSharing();
	public slots:
		void setOptions();
		void accepted();
		void setValues();
	signals:
		void valueChanged(float v1, float v2=-1.f, float v3=-1.f, float v4=-1.f);
};

class UniformInt : public Uniform {
	Q_OBJECT
protected:
	int* vars;
	int* ranges;
	QSpinBox** spins;
	QCheckBox* check;
	UniformInt* master;
	QList<UniformInt*> slaves;
	public:
		UniformInt(const QString& name, int numParams, int* vars, const QString& displayName = QString()):
			Uniform(INT, name, numParams, displayName), vars(vars), master(0) {
			ranges = new int[numParams*2];
			for(int i = 0; i < numParams*2; i++) {
				ranges[i] = 0;
			}

			spins = new QSpinBox*[numParams];
			for(int i = 0; i < numParams; i++) {
				spins[i] = 0;
			}
		}
		void setRanges(int first, ...);
		~UniformInt() {
			stopSharing();
			delete [] vars;
			delete [] ranges;
			delete [] spins;
		}
		int* v() {
			updated = true;
			return vars;
		}
		QWidget* getOptions();
		virtual void set(GLuint shader);
		void setValues(int first, ...);
		void setValues(int* v, int num);
		virtual void loadSettings(QIODevice* ios);
		virtual void saveSettings(QIODevice* ios);
		void shareValues(UniformInt* slave);
		void stopSharing();
	public slots:
		void accepted();
		void setOptions();
		void setValues();
	signals:
		void valueChanged(int v1, int v2=-1, int v3=-1, int v4=-1);
};

class GLTexture;
class UniformSampler : public UniformInt {
	Q_OBJECT
	GLTexture* tex;
public:
	UniformSampler(const QString& name, GLTexture* tex);
	void setTexture(GLTexture* tex);
	virtual void set(GLuint shader);
	virtual void loadSettings(QIODevice* ios);
	virtual void saveSettings(QIODevice* ios);
public slots:
	void textureDeleted();
};

class UniformMatrix : public Uniform {
	Q_OBJECT
	float* vars;
	int rows;
	int cols;
public:
	UniformMatrix(const QString& name, int cols, int rows, const QString& displayname = QString()):
	Uniform(MATRIX, name, cols*rows, displayname), vars(new float[rows*cols]), rows(rows), cols(cols) {
		for(int i = 0; i < (rows < cols ? rows : cols); i++) {
			vars[i*rows + i] = 1;
		}
	}
	void set(GLuint shader);
	void setValues(const Matrix4x4& m);
	float* v() {
		updated = true;
		return vars;
	}
	QWidget* getOptions() { return 0; }
	void accepted() {}
	~UniformMatrix() {
		delete vars;
	}
	virtual void loadSettings(QIODevice* ios);
	virtual void saveSettings(QIODevice* ios);
};

class UniformFloatArray : public Uniform {
	Q_OBJECT
	QVector<float> vars;
	int numSets;
public:
	UniformFloatArray(const QString& name, int numParams):Uniform(FLOAT_ARRAY, name, numParams), vars(numParams), numSets(0) {}
	void set(GLuint shader);
	void addValues(float first, ...);
	QVector<float>& v() {
		updated = true;
		return vars;
	}
	QWidget* getOptions() { return 0; }
	void accepted() {}
	virtual void loadSettings(QIODevice* ios);
	virtual void saveSettings(QIODevice* ios);
};

class UniformIntArray : public Uniform {
	Q_OBJECT
	QVector<int> vars;
	int numSets;
public:
	UniformIntArray(const QString& name, int numParams):Uniform(INT_ARRAY, name, numParams), vars(numParams), numSets(0) {}
	void set(GLuint shader);
	void addValues(int first, ...);
	QVector<int>& v() {
		updated = true;
		return vars;
	}
	QWidget* getOptions() { return 0; }
	void accepted() {}
	virtual void loadSettings(QIODevice* ios);
	virtual void saveSettings(QIODevice* ios);
};

class UniformMatrixArray : public Uniform {
	Q_OBJECT
	QVector<float> vars;
	int rows;
	int cols;
	int numSets;
public:
	UniformMatrixArray(const QString& name, int cols, int rows, const QString& displayname = QString()):
	Uniform(MATRIX_ARRAY, name, cols*rows, displayname), rows(rows), cols(cols) {}
	void set(GLuint shader);
	void addValues(Matrix4x4& m);
	QVector<float>& v() {
		updated = true;
		return vars;
	}
	QWidget* getOptions() { return 0; }
	void accepted() {}
	virtual void loadSettings(QIODevice* ios);
	virtual void saveSettings(QIODevice* ios);
};

class Shader : public QObject {
	Q_OBJECT

	GLuint id;
	QList<ShaderFile*> shaderfiles;
	QGroupBox *options;
	QString display;
	QHash<QString, Uniform*>* uniforms;
	QHash<QString, UniformFloat*>* floats;
	QHash<QString, UniformFloatArray*>* floatarrays;
	QHash<QString, UniformInt*>* ints;
	QHash<QString, UniformIntArray*>* intarrays;
	QHash<QString, UniformMatrix*>* matrices;
	QHash<QString, UniformMatrixArray*>* matrixarrays;
	QHash<QString, int> attribIndices;
	QHash<QString, UniformSampler*>* samplers;
	QStringList attribs;
	QHash<GLenum, GLint>* programParameters;
	void checkProgram();

	bool own, hasVariables;

	static QStack<Matrix4x4> modelStack;
	static QStack<Matrix4x4> viewStack;
	static QStack<Matrix4x4> projectionStack;

	static Matrix4x4 mv;
	static Matrix4x4 mvp;
	static Matrix4x4 vp;

	static bool matricesInitialized;
	static void initMatrices();
	static void updateMatrices();
	static void setMatrix(QStack<Matrix4x4>& stack, const Matrix4x4& m);
	static void popMatrix(QStack<Matrix4x4>& stack);
	static void pushMatrix(QStack<Matrix4x4>& stack, const Matrix4x4& m);
	static const Matrix4x4& getTop(QStack<Matrix4x4>& stack);
	static void setDefaultMatrices();

	static Shader* currentbound;
	static bool alwaysUpdate;
	friend class ShaderManager;

	public:
		Shader(const QString& name=QString());
		~Shader();
		void addVertexShader(const QString&);
		void addFragmentShader(const QString&);
		void addGeometryShader(const QString&);
		void addTessControlShader(const QString&);
		void addTessEvalShader(const QString&);
		void addFile(ShaderFile*);
		void setName(const QString& name);
		void compileAndLink();
		void use();
		void setUniforms(bool force=false);
		void setAttribute(const QString& name, int numParams, ...);
		void addAttrib(const QString& name);
		GLint getAttribLocation(const QString& name) const;
		void addUniformi(const QString& name, int numParams, ...);
		void addUniformf(const QString& name, int numParams, ...);
		void addUniformSampler(const QString& name, GLTexture* tex);
		void addUniformMatrix(const QString& name, int cols, int rows);
		void addUniformFloatArray(const QString& name, int numParams);
		void addUniformIntArray(const QString& name, int numParams);
		void addUniformMatrixArray(const QString& name, int cols, int rows);
		void addProgramParameter(GLenum pname, GLint value);
		void shareUniforms(Shader* other);
		void release();
		GLuint getId() {
			return id;
		}
		QWidget* getOptions();
		QHash<QString, UniformFloat*>& getFloats() { return *floats; }
		QHash<QString, UniformFloatArray*>& getFloatArrays() { return *floatarrays; }
		QHash<QString, UniformInt*>& getInts() { return *ints; }
		QHash<QString, UniformIntArray*>& getIntArrays() { return *intarrays; }
		QHash<QString, UniformMatrix*>& getMatrices() { return *matrices; }
		QHash<QString, UniformMatrixArray*>& getMatrixArrays() { return *matrixarrays; }
		QHash<QString, UniformSampler*>& getSamplers() { return *samplers; }
		Uniform* operator[](const QString& index);
		Shader& operator=(const Shader& rhs);
		void setHasOptions(bool o) {
			hasVariables = o;
		}
		bool getHasOptions() const { return hasVariables; }
		bool noOptions();
		bool isBound();
		static void setAlwaysUpdate(bool b) {
			alwaysUpdate = b;
		}

		static Shader* current() { return currentbound; }
		static void trySetUniformf(const QString& name, int numParams, ...);
		static void trySetUniformi(const QString& name, int numParams, ...);
		static void trySetUniformMatrix(const QString& name, const Matrix4x4& values);
		static void setModel(const Matrix4x4& m);
		static void setView(const Matrix4x4& m);
		static void setProjection(const Matrix4x4& m);
		static void pushView(const Matrix4x4& m);
		static void pushModel(const Matrix4x4& m);
		static void pushProjection(const Matrix4x4& m);
		static void popView();
		static void popProjection();
		static void popModel();
		static const Matrix4x4& getModel();
		static const Matrix4x4& getView();
		static const Matrix4x4& getProjection();
		static const Matrix4x4& getModelView();
		static const Matrix4x4& getViewProjection();
		static const Matrix4x4& getModelViewProjection();

		void loadSettings(QIODevice* ios);
		void saveSettings(QIODevice* ios);
		void setOptions();

	public slots:
		void accepted();
};
#endif
