#ifndef TFNODE_H
#define TFNODE_H

class QPointF;
class QPaintDevice;
class QPainter;

struct TFNode
{
	TFNode* prev;
	TFNode* next;
	float position;
	float value;
	TFNode(float p_position, float p_value, TFNode* p_prev=0, TFNode* p_next=0):prev(p_prev), next(p_next), position(p_position), value(p_value) {
		if(prev) {
			prev->next = this;
			if(value <= prev->value)
				value = prev->value;
		}
		if(next) {
			next->prev = this;
			if(value >= next->value)
				value = next->value;
		}
	}
	void setPositionValue(float position, float value);
	~TFNode() {
		if(prev) {
			prev->next = next;
		}
		if(next) {
			next->prev = prev;
		}
	}
	QPointF getPoint(QPaintDevice*) const;
	void draw(QPainter*) const;
};

#endif // TFNODE_H
