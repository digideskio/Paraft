#include "ProjectionView.h"

ProjectionView::ProjectionView(QWidget *parent) : QGraphicsView(parent) {
    int w = width();
    int h = height();
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

void ProjectionView::addNode(int nodeId, int nodeLabel, int x, int y) {
    Node *node = new Node(this, nodeLabel);
    node->setPos(x, y);
    seeds.push_back(node);
    scene->addItem(node);
}

std::vector<double> ProjectionView::getProjSeed() {
    std::vector<double> projSeed;
    for (Node *seed : seeds) {
        projSeed.push_back(seed->pos().x());
        projSeed.push_back(seed->pos().y());
    }
    return projSeed;
}

void ProjectionView::drawBackground(QPainter *painter, const QRectF &rect) {

}
