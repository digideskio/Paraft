#ifndef NODE_H
#define NODE_H

#include <QGraphicsItem>
#include <QList>

class ProjectionView;
class QGraphicsSceneMouseEvent;

class Node : public QGraphicsItem {
public:
    Node(ProjectionView *projView, int label);

    QRectF boundingRect() const;
    QPainterPath shape() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    bool advance();

protected:
    const int RADIUS = 10;
    const std::vector<Qt::GlobalColor> colorScheme {
        Qt::gray, Qt::red, Qt::green, Qt::blue, Qt::cyan, Qt::magenta, Qt::yellow,
        Qt::darkRed, Qt::darkGreen, Qt::darkBlue, Qt::darkCyan, Qt::darkMagenta, Qt::darkYellow,
    };

    QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

private:
    ProjectionView *projView;
    QPointF newPos;
    int label;
};

#endif // NODE_H
