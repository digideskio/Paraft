#ifndef DEVRENDERER_H
#define DEVRENDERER_H

#define PLUGIN_NAME "DevRenderer"
#define PLUGIN_VERSION "0.0.1"

#include "PluginInterface.h"

#include "JsonParser.h"

//#include "VolumeParser.h"
#include "VolumeRenderWindow.h"
#include "MainUI.h"

//#include "VolumeMetadata.h"
#include "VolumeModel.h"

////
#include "UDPListener.h"

class DevRenderer : public PluginInterface
{
    Q_OBJECT
    Q_INTERFACES(PluginInterface)

public:
    ~DevRenderer();
    void init();

protected:
    //virtual void subWindowActivated(const QString &windowTitle, QWidget *widget);
    //virtual void subWindowClosed(const QString &windowTitle, QWidget *widget);
    void subWindowActivated(QWidget *widget);
    void subWindowClosed(QWidget *widget);

    bool request(const QString &method, QVariant &arg);

    void updateRecentFileList(const QString &filePath);
    void writePreferencesFile();

    void test();    ////

    QString _pluginPath;

    //MainInterface *m_mainInterface;
    VolumeParser m_volumeParser;        ////

    //QRenderEffEditor *m_renderEffEditor;
    //QTFEditor *m_tfEditor;

    MainUI *_mainUI;

    QString _prefFileName;
    Json::Value _pref;
    QList<QString> _recentFiles;
    QMenu *_recentFilesMenu;


    VolumeRenderWindow *_activeWindow;

    ////
    UDPListener *_udpClient;


public slots:
    void act();
    void open(const QString &path);
    void open();
    void openRecentFile(int index);
    //void open(int index);

    ////
    void cameraTrack(int x, int y, int state);
};

class IndexedAction : public QAction
{
    Q_OBJECT
public:
    IndexedAction(const QString &text, int index, QObject *parent)
        : QAction(text, parent),_index(index)
    {
        connect(this, SIGNAL(triggered()), this, SLOT(triggerIndex()));
    }
protected:
    int _index;
protected slots:
    void triggerIndex() { emit triggered(_index); }
signals:
    void triggered(int index);
};

#endif // DEVRENDERER_H
