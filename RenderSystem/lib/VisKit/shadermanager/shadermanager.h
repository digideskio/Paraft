#ifndef _SHADERMANAGER_H_
#define _SHADERMANAGER_H_


#include "shader.h"
#include <QHash>
#include <QString>
#include <QList>
#include <QObject>
#include <QIODevice>
class QGroupBox;

class ShaderManager : public QObject {
	Q_OBJECT

	QHash<QString, Shader*> shaders;
	QList<Shader*> createdShaders;
	QGroupBox* options;
	int messageLevel;

	bool noOptions();
	void checkglErrors(const QString& shader);

	static Shader* shaderdummy;

	public:
		ShaderManager(int msgLevel = 1);
		~ShaderManager();
		void addShader(const QString&, Shader*);
		void addShader(const QString&, const QString& =QString());
		void useShader(const QString&);
		void useFixed();
		void reload();
		Shader*& operator[](const QString& index);
		QGroupBox* getOptions();
		void saveSettings(QIODevice* ios);
		void loadSettings(QIODevice* ios);

	public slots:
		void accepted();
		void setOptions();
};

#endif
