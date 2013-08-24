#ifndef PROJECTIONVIEW_H
#define PROJECTIONVIEW_H

#include <QGraphicsView>

class ProjectionView : public QGraphicsView {
    Q_OBJECT
public:
    ProjectionView(QWidget *parent = 0);

protected:
    void drawBackground(QPainter *painter, const QRectF &rect);

private:
    QGraphicsScene *scene = nullptr;
};

#endif // PROJECTIONVIEW_H
