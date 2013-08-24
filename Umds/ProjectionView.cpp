#include "ProjectionView.h"

ProjectionView::ProjectionView(QWidget *parent) : QGraphicsView(parent) {
    int w = width();
    int h = height();
    qDebug() << "w: " << w << "h: " << h;
    scene = new QGraphicsScene(this);
    scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    scene->setSceneRect(-w/2, -h/2, w, h);
    fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);
    setScene(scene);
    setCacheMode(CacheBackground);
    setViewportUpdateMode(BoundingRectViewportUpdate);
    setRenderHint(QPainter::Antialiasing);
    setTransformationAnchor(AnchorUnderMouse);
    setWindowTitle(tr("Visual Space"));
}

ProjectionView::~ProjectionView() {
    if (!seeds.isEmpty())
        for (auto seed : seeds)
            delete seed;
}

void ProjectionView::addNode(int nodeId, int nodeLabel) {
    Node *node = new Node(this);
    node->setPos(10*nodeId, -10*nodeId);
    seeds.push_back(node);
    scene->addItem(node);
}

void ProjectionView::drawBackground(QPainter *painter, const QRectF &rect) {

}
