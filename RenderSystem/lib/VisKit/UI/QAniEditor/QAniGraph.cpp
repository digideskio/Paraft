#include "QAniGraph.h"
#include "QAniEditor.h"
#include "QAniKeyframe.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif


QAniGraphNode::QAniGraphNode() : m_keyframe(NULL), m_hovered(false), m_highlighted(false), m_current(false), parent(NULL) {
}

QAniGraphNode::~QAniGraphNode() {
	if (m_keyframe) {
		if (m_keyframe->getLeftInstance()) delete m_keyframe->getLeftInstance();
		delete m_keyframe;
	}
}


/* class QAniGraph
 *
 *
 */

QAniGraph::QAniGraph(QWidget * parent, const QGLWidget * shareWidget)
:QGLWidget(parent, shareWidget), m_aniEditor((QAniEditor*)parent) {
	setWindowFlags(Qt::Tool);
	setWindowTitle(tr("Animation Graph"));
	this->setSizePolicy(QSizePolicy::MinimumExpanding,QSizePolicy::MinimumExpanding);
	setMinimumSize(400,300);
	resize(400,300);

	root = new QAniGraphNode();
	root->m_rect = QRectF(-2.0, 0.0, 1.0, 1.0);
	head = &(root->children);
	m_offsetx = -2.0;	
	m_offsety = 0.0;
	m_zoom = 5.0;

	m_mousePressed = false;
	m_hoveredNode = NULL;
	setMouseTracking(true);
	makeCurrent();
}

QAniGraph::~QAniGraph() {
	delete root;
}

void QAniGraph::initializeGL() {
	glewInit();

	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glDisable(GL_TEXTURE_2D);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void QAniGraph::resizeGL(int width, int height) {
	m_width = width;
	m_height = height;
	glViewport(0,0,width,height);
}

void QAniGraph::paintGL() {
	glClearColor(1.0,1.0,1.0,1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(m_offsetx, m_offsetx + m_zoom * (double)m_width / (double)m_height,
		   m_offsety, m_offsety + m_zoom);
	glScalef(1.0,-1.0,1.0);
	glTranslatef(0.0,-m_zoom,0.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glColor3f(0.0,0.0,0.0);
	glLineWidth(1.0);
	renderText(root->x()+0.2, root->y()+root->width()-0.17, 0.0, tr("S"), QFont("Arial", 200/m_zoom*frameGeometry().height()/300));
	glBegin(GL_LINE_STRIP);
	for (int i = 0; i < 21; ++i) {
		glVertex2f((root->x()+root->width()/2)+root->width()/2*cos(M_PI/10*i), (root->y()+root->height()/2)+root->width()/2*sin(M_PI/10*i));
	}
	glEnd();

	drawRecursiveGraph(head);
}

void QAniGraph::drawRecursiveGraph(QList<QAniGraphNode *> * nodes) {
	for (int i = 0; i < nodes->size(); ++i) {
		QAniGraphNode * node = (*nodes)[i];
		QAniGraphNode * parent = node->parent;

		if (node->m_highlighted && (parent->m_highlighted || parent == root)) {
			glLineWidth(4.0);
			glColor3f(0.5,0.0,0.0);
		}
		else if (node->m_current && (parent->m_current || parent == root)) {
			glLineWidth(4.0);
			glColor3d(0.7,0.7,0.0);
		}
		else {
			glLineWidth(1.0);
			glColor3d(0.0,0.0,0.0);
		}

		// draw edge between parent and the node
		double depth = 0.0;
		if (node->m_highlighted && (parent->m_highlighted || parent == root)) depth = 1.0;
		glEnable(GL_DEPTH_TEST);
		glBegin(GL_LINE_STRIP);
		if (i > 0) {
			glVertex3d(parent->x() + parent->width() / 2, parent->y() + parent->height(), depth);
			glVertex3d(parent->x() + parent->width() / 2, node->y() + node->height() / 2, depth);
			glVertex3d(node->x(), node->y() + node->height() / 2, depth);
		}
		else {
			glVertex3d(parent->x() + parent->width(), parent->y() + parent->height() / 2, depth);
			glVertex3d(node->x(), node->y() + node->height() / 2, depth);
		}
		glEnd();
		glDisable(GL_DEPTH_TEST);

		float r = (float)(*nodes)[i]->m_keyframe->getSnapshot().width() / (float)(*nodes)[i]->m_keyframe->getSnapshot().height();
		float w, h;
		if (r > 1.0) { w = 1/r/2; h = 0.5; }
		else { h = 1/r/2; w = 0.5; }

		// draw highlight if the node is in the hovered path
		if ((*nodes)[i]->m_current) {
			glBegin(GL_TRIANGLE_FAN);
			glColor4f(1.0,1.0,0.0,1.0); glVertex2d(node->x() + node->width()/2, node->y() + node->height()/2);
			glColor4f(1.0,1.0,0.0,0.0); glVertex2d(node->x()-0.3, node->y() - 0.3);
			glColor4f(1.0,1.0,0.0,0.0); glVertex2d(node->x()-0.3, node->y() + node->height() + 0.3);
			glColor4f(1.0,1.0,0.0,0.0); glVertex2d(node->x() + node->width() + 0.3, node->y() + node->height() + 0.3);
			glColor4f(1.0,1.0,0.0,0.0); glVertex2d(node->x() + node->width() + 0.3, node->y() - 0.3);
			glColor4f(1.0,1.0,0.0,0.0); glVertex2d(node->x()-0.3, node->y() - 0.3);
			glEnd();
		}

		// draw highlight if the node is in the hovered path
		if ((*nodes)[i]->m_highlighted) {			
			glBegin(GL_TRIANGLE_FAN);
			glColor4f(1.0,0.0,0.0,1.0); glVertex2d(node->x() + node->width()/2, node->y() + node->height()/2);
			glColor4f(1.0,0.0,0.0,0.0); glVertex2d(node->x()-0.3, node->y() - 0.3);
			glColor4f(1.0,0.0,0.0,0.0); glVertex2d(node->x()-0.3, node->y() + node->height() + 0.3);
			glColor4f(1.0,0.0,0.0,0.0); glVertex2d(node->x() + node->width() + 0.3, node->y() + node->height() + 0.3);
			glColor4f(1.0,0.0,0.0,0.0); glVertex2d(node->x() + node->width() + 0.3, node->y() - 0.3);
			glColor4f(1.0,0.0,0.0,0.0); glVertex2d(node->x()-0.3, node->y() - 0.3);
			glEnd();
		}

		// draw the node
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, (*nodes)[i]->m_keyframe->getTexture());
		glBegin(GL_QUADS);
		glTexCoord2f(0.5-w,0.5-h); glVertex2d(node->x(), node->y());
		glTexCoord2f(0.5-w,0.5+h); glVertex2d(node->x(), node->y() + node->height());
		glTexCoord2f(0.5+w,0.5+h); glVertex2d(node->x() + node->width(), node->y() + node->height());
		glTexCoord2f(0.5+w,0.5-h); glVertex2d(node->x() + node->width(), node->y());
		glEnd();
		glDisable(GL_TEXTURE_2D);
	
		drawRecursiveGraph(&((*nodes)[i]->children));
	}
}

void QAniGraph::mousePressEvent(QMouseEvent * event) {
	if (m_hoveredNode) {
		clearCurrentRecurrsiveGraph(head);
		m_currentNodes.clear();
		QAniGraphNode * prev = m_hoveredNode;
		QList<QAniKeyframe*> keyframeList;

		QAniKeyframe * keyframe = NULL;
		m_hoveredNode->m_current = true;
		m_hoveredNode->m_keyframe->cloneKeyframe(keyframe);
		keyframe->setTexture(keyframe->getSnapshot());
		keyframeList.push_back(keyframe);
		if (m_hoveredNode->m_keyframe->getLeftInstance()) {
			QAniInstance * instance = NULL;
			m_hoveredNode->m_keyframe->getLeftInstance()->cloneInstance(instance);
			keyframe->setLeftInstance(instance);
		}

		for (QAniGraphNode * node = m_hoveredNode->parent; node != NULL; node = node->parent) {
			for (int i = 0; i < node->children.size(); ++i) {
				if (node->children[i] == prev) m_currentNodes.push_front(i);
			}
			prev = node;

			if (node != root) {
				keyframe = NULL;
				node->m_current = true;
				node->m_keyframe->cloneKeyframe(keyframe);
				keyframe->setTexture(keyframe->getSnapshot());
				keyframeList.push_front(keyframe);
				if (node->m_keyframe->getLeftInstance()) {
					QAniInstance * instance = NULL;
					node->m_keyframe->getLeftInstance()->cloneInstance(instance);
					keyframe->setLeftInstance(instance);
				}
			}
		}
/*		for (QAniGraphNode * node = m_hoveredNode; node->children.size() > 0; node = node->children.first()) {
			keyframe = NULL;
			node->children.first()->m_current = true;
			node->children.first()->m_keyframe->cloneKeyframe(keyframe);
			keyframe->setTexture(keyframe->getSnapshot());
			keyframeList.push_back(keyframe);
			if (node->children.first()->m_keyframe->getLeftInstance()) {
				QAniInstance * instance = NULL;
				node->children.first()->m_keyframe->getLeftInstance()->cloneInstance(instance);
				keyframe->setLeftInstance(instance);
			}
		}*/
		emit pickedTimelineFromGraph(&keyframeList);
		updateGL();
	}
	else {
		m_oldx = event->x();
		m_oldy = event->y();
		setCursor(Qt::ClosedHandCursor);
		m_mousePressed = true;
	}
}

void QAniGraph::mouseMoveEvent(QMouseEvent * event) {
	if (m_mousePressed) {
		m_offsetx -= (double)(event->x() - m_oldx) / (double)m_height * m_zoom;
		m_offsety += (double)(event->y() - m_oldy) / (double)m_height * m_zoom;
		m_oldx = event->x();
		m_oldy = event->y();
		updateGL();
	}
	else { // hover
		double x = m_offsetx + m_zoom * (double)event->x() / (double)m_height;
		double y = - m_offsety + m_zoom - m_zoom * (1.0 - (double)event->y() / (double)m_height);
		m_hoveredNode = NULL;
		hoverRecursiveGraph(head, x, y);
		if (m_hoveredNode) {
			setCursor(Qt::PointingHandCursor);
			QAniGraphNode * node;
			for (node = m_hoveredNode; node != root; node = node->parent) node->m_highlighted = true;
//			for (node = m_hoveredNode; node->children.size() > 0; node = node->children.first()) node->children.first()->m_highlighted = true;
		}
		else setCursor(Qt::OpenHandCursor);
		updateGL();
	}
}

void QAniGraph::hoverRecursiveGraph(QList<QAniGraphNode *> * nodes, double x, double y) {
	for (int i = 0; i < nodes->size(); ++i) {
		(*nodes)[i]->m_highlighted = false;
		if ((*nodes)[i]->contains(x, y)) {
			(*nodes)[i]->m_hovered = true;
			m_hoveredNode = (*nodes)[i];
		}
		else (*nodes)[i]->m_hovered = false;

		hoverRecursiveGraph(&((*nodes)[i]->children), x, y);
	}
}

void QAniGraph::mouseReleaseEvent(QMouseEvent *) {
	setCursor(Qt::OpenHandCursor);
	m_mousePressed = false;
}

void QAniGraph::wheelEvent(QWheelEvent * event) {
	double x = m_offsetx + m_zoom * (double)event->x() / (double)m_height;
	double y = - m_offsety + m_zoom - m_zoom * (1.0 - (double)event->y() / (double)m_height);
	m_zoom /= (1.0 + (double)(event->delta()/120)/10.0);
	m_offsetx = x - m_zoom * (double)event->x() / (double)m_height;
	m_offsety = -(y - m_zoom) - m_zoom * (1.0 - (double)event->y() / (double)m_height);
	updateGL();
}

void QAniGraph::buildGraph(QList<QAniKeyframe*> * KeyframeList){
	int changeIdx;
	QList<QAniGraphNode *> * nodes = head;
	QAniGraphNode * parent = root;
	clearCurrentRecurrsiveGraph(head);
	m_currentNodes.clear();
	for (changeIdx = 0; changeIdx < KeyframeList->size() && nodes->size() > 0; ++changeIdx) {
		bool match = false;
		for (int i = 0; i < nodes->size(); ++i) {
			if (((*KeyframeList)[changeIdx])->totallyEqual((*nodes)[i]->m_keyframe)) {
				if (((*KeyframeList)[changeIdx])->getLeftInstance()) {
					if (*(((*KeyframeList)[changeIdx])->getLeftInstance()) == *((*nodes)[i]->m_keyframe->getLeftInstance())) {
						parent = (*nodes)[i];
						parent->m_current = true;
						m_currentNodes.push_back(i);
						nodes = &(parent->children);
						match = true;
					}
				}
				else if (((*KeyframeList)[changeIdx])->getLeftInstance() == NULL && (*nodes)[i]->m_keyframe->getLeftInstance() == NULL) {
					parent = (*nodes)[i];
					parent->m_current = true;
					m_currentNodes.push_back(i);
					nodes = &(parent->children);
					match = true;
				}
			}
		}
		if (!match) break;
	}

	while (changeIdx < KeyframeList->size()) {
		QAniGraphNode * node = new QAniGraphNode();
		QAniKeyframe * keyframe = (*KeyframeList)[changeIdx];
		// copy the keyframe
		keyframe->cloneKeyframe(node->m_keyframe);

		// copy the instances if existed
		if (keyframe->getLeftInstance()) {
			QAniInstance * instance = NULL;
			keyframe->getLeftInstance()->cloneInstance(instance);
			node->m_keyframe->setLeftInstance(instance);
		}

		if (parent->children.size() > 0) node->m_rect = QRectF(parent->m_rect.x() + 2.0, parent->m_rect.y() + 2.0 * (double)getBreadth(parent), 1.0, 1.0);
		else node->m_rect = QRectF(parent->m_rect.x() + 2.0, parent->m_rect.y(), 1.0, 1.0);
		node->m_keyframe->setTexture(node->m_keyframe->getSnapshot());
		node->parent = parent;
		node->m_current = true;
		m_currentNodes.push_back(nodes->size());
		nodes->push_back(node);
		parent = nodes->last();
		nodes = &(nodes->last()->children);
		changeIdx++;
	}

	updatePosition(root);
	updateGL();
}

void QAniGraph::aniUpdate() {
	updateGL();
}

int QAniGraph::getBreadth(QAniGraphNode * node) {
	if (node->children.size() == 0) return 1;
	int branch = 0;
	for (int i = 0; i < node->children.size(); ++i) {
		branch += getBreadth(node->children[i]);
	}
	return branch;
}

void QAniGraph::updatePosition(QAniGraphNode * node) {
	int currentBreadth = 0;
	for (int i = 0; i < node->children.size(); ++i) {
		QAniGraphNode * child = node->children[i];
		child->m_rect = QRectF(node->x() + 2.0, node->m_rect.y() + 2.0 * (double)currentBreadth, 1.0, 1.0);
		updatePosition(child);
		currentBreadth += getBreadth(child);
	}
}

void QAniGraph::clearCurrentRecurrsiveGraph(QList<QAniGraphNode *> * nodes) {
	for (int i = 0; i < nodes->size(); ++i) {
		(*nodes)[i]->m_current = false;
		clearCurrentRecurrsiveGraph(&((*nodes)[i]->children));
	}
}

void QAniGraph::saveGraph(QFile & file) {
	int size = head->size();
	file.write((char *)&size, 4);
	saveRecursiveGraph(file, head);

	size = m_currentNodes.size();
	file.write((char *)&size, 4);
	for (int i = 0; i < size; ++i)
		file.write((char *)&(m_currentNodes[i]), 4);
}

void QAniGraph::saveRecursiveGraph(QFile & file, QList<QAniGraphNode *> * nodes) {
	for (int i = 0; i < nodes->size(); ++i) {
		QAniGraphNode * node = (*nodes)[i];
		int size = node->children.size();
		file.write((char*)&size, 4);
		saveRecursiveGraph(file, &(node->children));
		node->m_keyframe->save(file);
		if (node->m_keyframe->getLeftInstance()) node->m_keyframe->getLeftInstance()->save(file);
	}
}

void QAniGraph::loadGraph(QFile & file) {
	int size;
	file.read((char *)&size, 4);

	if (size < 1) return;

	removeALL();
	m_currentNodes.clear();

	// load the whole graph
	loadRecursiveGraph(file, root, size);

	// load the current timeline
	file.read((char *)&size, 4);
	for (int i = 0; i < size; ++i) {
		int idx;
		file.read((char *)&idx, 4);
		m_currentNodes.push_back(idx);
	}

	// build current timeline
	QAniKeyframe * keyframe = NULL;
	QList<QAniKeyframe*> keyframeList;
	QList<QAniGraphNode *> * nodes = head;
	for (int i = 0; i < m_currentNodes.size(); ++i) {
		QAniGraphNode * node = (*nodes)[m_currentNodes[i]];
		node->m_current = true;

		keyframe = NULL;
		node->m_keyframe->cloneKeyframe(keyframe);
		keyframe->setTexture(keyframe->getSnapshot());
		keyframeList.push_back(keyframe);
		if (node->m_keyframe->getLeftInstance()) {
			QAniInstance * instance = NULL;
			node->m_keyframe->getLeftInstance()->cloneInstance(instance);
			keyframe->setLeftInstance(instance);
		}
		nodes = &(node->children);
	}
	emit pickedTimelineFromGraph(&keyframeList);


	updatePosition(root);
	updateGL();
}

void QAniGraph::loadRecursiveGraph(QFile & file, QAniGraphNode * parent, int nodesNum) {
	QAniGraphNode * node;
	for (int i = 0; i < nodesNum; ++i) {
		node = new QAniGraphNode();
		node->parent = parent;
		int size;
		file.read((char *)&size, 4);
		loadRecursiveGraph(file, node, size);
		node->m_keyframe = QAniKeyframe::load(file);
		if (node->m_keyframe->getLeftInstance() != NULL) {
			QAniInstance * instance = QAniInstance::load(file);
			node->m_keyframe->setLeftInstance(instance);
		}
		parent->children.push_back(node);
	}
}

void QAniGraph::removeALL() {
	removeRecursiveGraph(head);
}

void QAniGraph::removeRecursiveGraph(QList<QAniGraphNode *> * nodes) {
	for (int i = 0; i < nodes->size(); ++i) {
		QAniGraphNode * node = (*nodes)[i];
		removeRecursiveGraph(&(node->children));
		delete node;
	}
	nodes->clear();
}
