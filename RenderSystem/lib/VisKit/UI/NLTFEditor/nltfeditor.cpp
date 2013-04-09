#include "nltfeditor.h"
#include "tfnode.h"
#include <cmath>
#include <QPaintEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>

NLTFEditor::NLTFEditor(QWidget *parent)
	: QWidget(parent), head(new TFNode(0, 0)), tail(new TFNode(1, 1)), selected(0), highlighted(0), moved(false)
{
	head->next = tail;
	tail->prev = head;

	setMouseTracking(true);
	setFocusPolicy(Qt::StrongFocus);
}

NLTFEditor::~NLTFEditor()
{
	TFNode* cur = head;
	while(cur) {
		TFNode* next = cur->next;
		delete cur;
		cur = next;
	}
}


void NLTFEditor::paintEvent(QPaintEvent *e) {
	QPainter painter(this);
	for(TFNode* cur = head; cur->next; cur = cur->next) {
		painter.drawLine(cur->getPoint(this), cur->next->getPoint(this));
	}
	painter.setBrush(Qt::SolidPattern);
	for(TFNode* cur = head; cur; cur = cur->next) {
		if(cur == selected) {
			painter.setBrush(QColor(255, 255, 0));
		} else if(cur == highlighted) {
			painter.setBrush(QColor(0, 255, 255));
		} else {
			painter.setBrush(QColor(0, 0, 0));
		}
		cur->draw(&painter);
	}
	QWidget::paintEvent(e);
}

void NLTFEditor::mousePressEvent(QMouseEvent *) {
	moved = false;
	if(highlighted) {
		selected = highlighted;
	}
}

void NLTFEditor::mouseReleaseEvent(QMouseEvent *) {
	update();
	if(moved)
		functionChanged();
}

void NLTFEditor::mouseMoveEvent(QMouseEvent *e) {
	if(!e->buttons() && head->next != tail) {
		TFNode* closest = head->next;
		QPointF curpoint = head->next->getPoint(this);
		float curdist = fabs(curpoint.x() - e->x()) + fabs(curpoint.y() - e->y());
		for(TFNode* cur = head->next; cur != tail; cur = cur->next) {
			QPointF point = cur->getPoint(this);
			float dist = fabs(point.x() - e->x()) + fabs(point.y() - e->y());
			if(dist < curdist) {
				closest = cur;
				curpoint = point;
				curdist = dist;
			}
		}
		if(!(fabs(curpoint.x() - e->x()) <= 4 && fabs(curpoint.y() - e->y()) <= 4)) {
			closest = 0;
		}

		if(closest != highlighted) {
			highlighted = closest;
			update();
		}
	} else {
		if(highlighted && highlighted == selected) {
			selected->setPositionValue(e->x()/(float)width(), 1.f - e->y()/(float)height());
			moved = true;
			update();
		}
	}
}

void NLTFEditor::mouseDoubleClickEvent(QMouseEvent *e) {
	if(highlighted) return;

	selected = insertNode(e->x()/(float)width(), 1.f - e->y()/(float)height());
	highlighted = selected;
	update();
	functionChanged();
}


void NLTFEditor::keyReleaseEvent(QKeyEvent *e) {
	switch(e->key()) {
	case Qt::Key_Delete:
		if(selected) {
			delete selected;
			selected = 0;
			update();
			functionChanged();
		}
	default:
		e->ignore();
	}
}

TFNode* NLTFEditor::insertNode(float position, float value) {
	TFNode* cur;
	for(cur = head; cur->next && cur->next->position < position; cur = cur->next);
	return new TFNode(position, value, cur, cur->next);
}

const QList<QPair<float,float> >& NLTFEditor::getNodes() const {
	return mapping;
}


void NLTFEditor::functionChanged() {
	mapping.clear();
	for(TFNode* cur = head; cur; cur = cur->next) {
		mapping.push_back(QPair<float,float>(cur->position, cur->value));
	}
	emit mappingChanged();
}
