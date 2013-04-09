#include "JsonParser.h"
#include "UDPListener.h"

void UDPListener::initSocket()
{
    _udpSocket = new QUdpSocket(this);
    _udpSocket->bind(QHostAddress::Any, 3000);
    connect(_udpSocket, SIGNAL(readyRead()), this, SLOT(read()));

}

void UDPListener::read()
{
    Json::Parser parser;
    while (_udpSocket->hasPendingDatagrams())
    {
        QByteArray datagram;
        datagram.resize(_udpSocket->pendingDatagramSize());
        _udpSocket->readDatagram(datagram.data(), datagram.size());
        //for (int i = 0; i < datagram.size(); i++)
        //qDebug("%s", datagram.data());
        std::string doc(datagram.data(), datagram.size());
        Json::Value root;
        parser.parse(doc, root);
        parser.write(doc, root);
        qDebug("%s", doc.c_str());
        int x = root[0]["Palm"]["X"].toInt();
        int y = root[0]["Palm"]["Y"].toInt();
        int state = root[0]["State"].toInt();
        qDebug("palm: %d, %d (%d)", x, y, state);
        emit handEvent(x, y, state);
    }
}
