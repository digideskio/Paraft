#ifndef UDPLISTENER_H
#define UDPLISTENER_H

#include <QtCore>
#include <QtNetwork>

class UDPListener : public QObject
{
    Q_OBJECT
public:
    void initSocket();

public slots:
    void read();

signals:
    void handEvent(int x, int y, int state);

private:
    QUdpSocket *_udpSocket;

};

#endif // UDPLISTENER_H
