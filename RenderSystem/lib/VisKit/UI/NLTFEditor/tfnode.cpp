#include "tfnode.h"
#include <QPaintDevice>
#include <QPainter>
#include <QPointF>

QPointF TFNode::getPoint(QPaintDevice* device) const {
	return QPointF(device->width()*position, device->height()*(1.f - value));
}

void TFNode::draw(QPainter *painter) const {
	QPainterPath path;
	path.addEllipse(painter->device()->width()*position - 3, painter->device()->height()*(1.f - value) - 3, 6, 6);
	painter->drawPath(path);
}

void TFNode::setPositionValue(float p_position, float p_value) {
	position = p_position;
	value = p_value;

	if(prev) {
		if(position <= prev->position) {
			position = prev->position;
		}
		if(value <= prev->value) {
			value = prev->value;
		}
	}
	if(next) {
		if(position >= next->position) {
			position = next->position;
		}
		if(value >= next->value) {
			value = next->value;
		}
	}
}


