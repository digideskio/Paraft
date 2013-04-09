#ifndef MAININTERFACE_H
#define MAININTERFACE_H

#include <QtGui>

class PluginInterface;

class MainInterface {
public:
    virtual QDockWidget *addDockWidget(const QString &title, QWidget *widget, Qt::DockWidgetArea area = Qt::RightDockWidgetArea, bool show = true) = 0;
    virtual QMdiSubWindow *addSubWindow(QWidget *widget, PluginInterface *plugin) = 0;
    virtual void addToolBar(QToolBar *toolbar, Qt::ToolBarArea area = Qt::TopToolBarArea) = 0;
    virtual PluginInterface *currentPlugin() = 0;
    virtual PluginInterface *getActivePlugin() = 0;         // plugin with currently active subwindow
    virtual PluginInterface *getPlugin(const QString &name) = 0;
    virtual QAction *&getAction(const QString &text) = 0;
    virtual QMenu *&getMenu(const QString &title) = 0;
    virtual QString getPluginsPath() = 0;
    virtual void tabifyDockWidget(QDockWidget *first, QDockWidget *second) = 0;
};

#endif // MAININTERFACE_H
