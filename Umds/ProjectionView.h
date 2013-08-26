#ifndef PROJECTIONVIEW_H
#define PROJECTIONVIEW_H

#include <QDebug>
#include <QGraphicsView>
#include <QMouseEvent>
#include "Node.h"

class ProjectionView : public QGraphicsView {
    Q_OBJECT
public:
    ProjectionView(QWidget *parent = 0);
    ~ProjectionView();

    void addNode(int nodeLabel, int x, int y, bool isSeed);
//    void clearScene();
    std::vector<double> getProjSeed();

protected:
    void drawBackground(QPainter *painter, const QRectF &rect);

private:
    QGraphicsScene *scene = nullptr;
    QList<Node*> seeds;
};

#endif // PROJECTIONVIEW_H
