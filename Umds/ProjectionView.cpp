#include "ProjectionView.h"

ProjectionView::ProjectionView(QWidget *parent) : QGraphicsView(parent) {
    scene = new QGraphicsScene(this);
    scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    scene->setSceneRect(-200, -200, 400, 400);
    setScene(scene);
    setCacheMode(CacheBackground);
    setViewportUpdateMode(BoundingRectViewportUpdate);
    setRenderHint(QPainter::Antialiasing);
    setTransformationAnchor(AnchorUnderMouse);
//    scale(qreal(0.8), qreal(0.8));
    setMinimumSize(400, 400);
//    setWindowTitle(tr("Elastic Nodes"));
}

void ProjectionView::drawBackground(QPainter *painter, const QRectF &rect) {
    QRectF sceneRect = this->sceneRect();
    QLinearGradient gradient(sceneRect.topLeft(), sceneRect.bottomRight());
    gradient.setColorAt(0, Qt::white);
    gradient.setColorAt(1, Qt::lightGray);
    painter->fillRect(rect.intersected(sceneRect), gradient);
    painter->setBrush(Qt::NoBrush);
    painter->drawRect(sceneRect);
}
