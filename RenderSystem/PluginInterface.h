/*
 * PluginInterface.h
 *
 * The interface for plugins.
 *
 * Usage:
 * 1. Declare a plugin class that inherits from PluginInterface.
 * 2. Use the Q_INTERFACES(PluginInterface) macro to tell Qt's meta-object system about the interfaces.
 * 3. Export the plugin using the Q_EXPORT_PLUGIN2() macro.
 */

#ifndef PLUGININTERFACE_H
#define PLUGININTERFACE_H

#include "MainInterface.h"

class PluginInterface : public QObject {
    Q_OBJECT

public:
    virtual ~PluginInterface() {}
    virtual void init() {}
    virtual void subWindowActivated(QWidget *widget) { Q_UNUSED(widget); }
    virtual void subWindowClosed(QWidget *widget) { Q_UNUSED(widget); }
    virtual void keyPressEvent(QKeyEvent *event) { Q_UNUSED(event); }

    QString getPluginName() const { return _pluginName; }
    QString getPluginVersion() const { return _pluginVersion; }
    void    setMainInterface(MainInterface *mainInterface) { _mainInterface = mainInterface; }

    virtual bool request(const QString &method, QVariant &arg) { Q_UNUSED(method); Q_UNUSED(arg); return false; }

protected:
    void setPluginName(const QString &pluginName) { _pluginName = pluginName; }                // set it in init()
    void setPluginVersion(const QString &pluginVersion) { _pluginVersion = pluginVersion; }    // set it in init()

    QString         getPluginPath() const { return _mainInterface->getPluginsPath() + QString("/") + _pluginName; }
    QMenu        *& getMenu(const QString &title) { return _mainInterface->getMenu(title); }
    QAction      *& getAction(const QString &text) { return _mainInterface->getAction(text); }
    QDockWidget   * addDockWidget(const QString &title, QWidget *widget, Qt::DockWidgetArea area = Qt::RightDockWidgetArea, bool show = true) { return _mainInterface->addDockWidget(title, widget, area, show);}
    QMdiSubWindow * addSubWindow(QWidget *widget) { return _mainInterface->addSubWindow(widget, this); }
    void            tabifyDockWidget(QDockWidget *first, QDockWidget *second) { _mainInterface->tabifyDockWidget(first, second); }

    MainInterface * getMainInterface() { return _mainInterface; }

private:
    MainInterface * _mainInterface;
    QString         _pluginName;
    QString         _pluginVersion;
};

Q_DECLARE_INTERFACE(PluginInterface, "SimpleRenderer.PluginInterface")

#endif // PLUGININTERFACE_H
