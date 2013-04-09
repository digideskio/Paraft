#include <GL/glew.h>
#include <QGLContext>
#include "texturemanager.h"
#include "gltexture.h"

QHash<const QGLContext*, TextureManager*> TextureManager::managers;

TextureManager::TextureManager(QObject *parent):
	QObject(parent),
	memPool(0),
	printUsageChanges(false)
{
	int maxtextures;
	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxtextures);
	texslots.resize(maxtextures);
	reservedslots.resize(maxtextures);
	for(int i = 0; i < maxtextures; ++i) {
		texslots[i] = 0;
		reservedslots[i] = false;
	}
}

TextureManager* TextureManager::getInstance() {
	const QGLContext* cur = QGLContext::currentContext();
	if(!cur)
		return 0;
	if(!managers.contains(cur)) {
		managers[cur] = new TextureManager();
	}
	return managers[cur];
}

bool TextureManager::bindToSlot(int slot, GLTexture *tex) {
	if(queue.contains(tex)) {
		queue.removeAll(tex);
	}
	queue.push_back(tex);
	if(slot < 0 || slot >= texslots.size())
		return false;
	if(tex->isBound()) {
		texslots[tex->getSlot()] = 0;
	}

	if(texslots[slot]) {
		texslots[slot]->setBound(false);
		texslots[slot]->setSlot(-1);
	}

	glActiveTexture(GL_TEXTURE0 + slot);
	tex->managerBind();
	tex->setSlot(slot);
	texslots[slot] = tex;
	return true;
}

bool TextureManager::bind(GLTexture* tex) {
	if(tex->isBound()) {
		if(queue.contains(tex)) {
			queue.removeAll(tex);
		}
		queue.push_back(tex);
		glActiveTexture(GL_TEXTURE0 + tex->getSlot());
		return true;
	}

	if(queue.size() == texslots.size()) { //all slots are full
		GLTexture* ousted = queue.front();
		queue.pop_front();
		return bindToSlot(ousted->getSlot(), tex);
	}

	for(int i = 0; i < texslots.size(); ++i) {
		if(!texslots[i] && !reservedslots[i]) {
			return bindToSlot(i, tex);
		}
	}
	return false; //should never get here
}

void TextureManager::remove(GLTexture *tex) {
	if(tex->isBound()) {
		texslots[tex->getSlot()] = 0;
	}
	if(queue.contains(tex)) {
		queue.removeAll(tex);
	}
}

int TextureManager::getFreeSlot() const {
	for(int i = 0; i < texslots.size(); ++i) {
		if(!texslots[i] && !reservedslots[i])
			return i;
	}
	return -1;
}

void TextureManager::reserveSlot(int slot) {
	if(texslots[slot]) {
		texslots[slot]->setBound(false);
		texslots[slot]->setSlot(-1);
		texslots[slot] = 0;
	}
	reservedslots[slot] = true;
}

void TextureManager::freeSlot(int slot) {
	reservedslots[slot] = false;
}

void TextureManager::addToMemPool(size_t change) {
	memPool += change;
	if(printUsageChanges)
		qDebug("New mem size: %zu (%zuKB, %fMB)", memPool, memPool/1024, memPool/(1024.*1024.));
}

void TextureManager::removeFromMemPool(size_t change) {
	memPool -= change;
	if(printUsageChanges)
		qDebug("New mem size: %zu (%zuKB, %fMB)", memPool, memPool/1024, memPool/(1024.*1024.));

}
