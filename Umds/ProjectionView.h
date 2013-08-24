#ifndef PROJECTIONVIEW_H
#define PROJECTIONVIEW_H

#include <QDebug>
#include <QGraphicsView>
#include "Node.h"

class ProjectionView : public QGraphicsView {
    Q_OBJECT
public:
    ProjectionView(QWidget *parent = 0);
    ~ProjectionView();

    void addNode(int nodeId, int nodeLabel);
    std::vector<double> getProjSeed();

protected:
    void drawBackground(QPainter *painter, const QRectF &rect);

private:
    QGraphicsScene *scene = nullptr;
    QList<Node*> seeds;
};

#endif // PROJECTIONVIEW_H
