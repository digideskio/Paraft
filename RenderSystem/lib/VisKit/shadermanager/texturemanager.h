#ifndef TEXTUREMANAGER_H
#define TEXTUREMANAGER_H

#include <QObject>
#include <QVector>
#include <QHash>
#include <QQueue>

class GLTexture;
class QGLContext;
class TextureManager : public QObject
{
Q_OBJECT

	QVector<GLTexture*> texslots;
	QVector<bool> reservedslots;
	QQueue<GLTexture*> queue;
	size_t memPool;
	static QHash<const QGLContext*, TextureManager*> managers;
	bool printUsageChanges;

protected:
	explicit TextureManager(QObject *parent = 0);

public:
	static TextureManager* getInstance();
	bool bindToSlot(int slot, GLTexture* tex);
	bool bind(GLTexture* tex);
	void remove(GLTexture* tex);
	int getFreeSlot() const;
	void reserveSlot(int slot);
	void freeSlot(int slot);
	void addToMemPool(size_t size);
	void removeFromMemPool(size_t size);
	size_t getMemUsage() const { return memPool; }
	void togglePrintUsageChanges(bool v) { printUsageChanges = v; }

signals:

public slots:

};

#endif // TEXTUREMANAGER_H
