#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui>
#include "MainInterface.h"
#include "PluginInterface.h"

class MainWindow;

class PluginManager {
public:
    PluginManager(MainWindow *mainWindow);
    ~PluginManager();
    bool loadPlugins();
    bool readList(QString fileName = QString());
    bool writeList(QString fileName = QString());
    int size() { return m_plugins.size(); }
    int count() { return m_plugins.size(); }
    QString name(int index) { return m_plugins[index].name; }
    QString version(int index) { return m_plugins[index].version; }
    QString fileName(int index) { return m_plugins[index].fileName; }
    bool isEnabled(int index) { return m_plugins[index].enabled; }
    void setEnabled(int index, bool enabled) { m_plugins[index].enabled = enabled; }
    bool isLoaded(int index) { return m_plugins[index].loaded; }
    PluginInterface *getPlugin(int index) { return m_plugins[index].instance; }
    PluginInterface *getPlugin(const QString &name);

private:
    struct Plugin {
        Plugin(const QString &_name, const QString &_version, const QString &_fileName, bool _enabled)
            : name(_name), fileName(_fileName), version(_version), enabled(_enabled), loaded(false), instance(NULL) {}
        QString name;
        QString version;
        QString fileName;
        bool enabled;
        bool loaded;
        PluginInterface *instance;
    };

    MainWindow *m_mainWindow;
    QString m_pluginsPath;
    QList<Plugin> m_plugins;
};

class WindowAction : public QAction {
    Q_OBJECT
public:
    WindowAction(const QString &text, QMdiSubWindow *window, QObject *parent = 0)
        : QAction(text, parent), m_window(window) { setCheckable(true); connect(this, SIGNAL(triggered()), this, SLOT(check())); }
protected:
    QMdiSubWindow *m_window;
public slots:
    void check() { setChecked(true); emit triggered(m_window); if (m_window->isMinimized()) m_window->showNormal(); }
    void windowStateChanged(Qt::WindowStates oldState, Qt::WindowStates newState);
signals:
    void triggered(QMdiSubWindow *window);
};

class SubWindow : public QMdiSubWindow {
    Q_OBJECT
public:
    SubWindow(QWidget *parent = 0, Qt::WindowFlags flags = 0) : QMdiSubWindow(parent, flags) { /*setFocusPolicy(Qt::StrongFocus);*/ }
    void setParentPlugin(PluginInterface *plugin) { m_parentPlugin = plugin; }
    PluginInterface *getParentPlugin() { return m_parentPlugin; }
    void setWidget(QWidget *widget) { QMdiSubWindow::setWidget(widget); setFocusProxy(widget); }
protected:
    virtual void keyPressEvent(QKeyEvent *e) { qDebug("SubWindow::keyPressEvent(%d)", e->key()); QMdiSubWindow::keyPressEvent(e); }
    virtual void closeEvent(QCloseEvent *) { emit windowClosed(this); }
    virtual void focusInEvent(QFocusEvent *e) { qDebug("SubWindow::focusInEvent()"); QMdiSubWindow::focusInEvent(e); }
    PluginInterface *m_parentPlugin;
signals:
    void windowClosed(QMdiSubWindow *window);
};

class AboutPluginsWindow : public QWidget {
    Q_OBJECT
public:
    AboutPluginsWindow(PluginManager *pluginManager, QWidget *parent = 0);
protected:
    PluginManager *m_pluginManager;
    QVBoxLayout *m_layout;
    QTableWidget *m_table;
    QHBoxLayout *m_layoutBottom;
    QPushButton *m_btnOK, *m_btnCancel;
    QPushButton *m_btnClose;
    bool m_dirty;
public slots:
    void show();
    void cellChanged(int row, int column);
    void btnOK();
    void btnCancel();
    void btnClose();
};

class MainWindow : public QMainWindow, public MainInterface {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

    // MainInterface
    QString getPluginsPath();   // { return QString(PLUGINS_PATH); }
    QMenu *&getMenu(const QString &title) { return m_menus[title]; }
    QAction *&getAction(const QString &text) { return m_actions[text]; }
    QDockWidget *addDockWidget(const QString &title, QWidget *widget, Qt::DockWidgetArea area, bool show);
    QMdiSubWindow *addSubWindow(QWidget *widget, PluginInterface *plugin);
    void addToolBar(QToolBar *toolbar, Qt::ToolBarArea area);
    void tabifyDockWidget(QDockWidget *first, QDockWidget *second);

    PluginInterface *currentPlugin() { return qobject_cast<SubWindow *>(m_mdiArea->currentSubWindow())->getParentPlugin(); }
    PluginInterface *getActivePlugin() {
        QMdiSubWindow *currentSubWindow = m_mdiArea->currentSubWindow();
        if (currentSubWindow == 0)
            return 0;
        else
            return qobject_cast<SubWindow *>(currentSubWindow)->getParentPlugin();
    }

    PluginInterface *getPlugin(const QString &name) { return m_pluginManager->getPlugin(name); }

protected:
    void initLayout();
    void keyPressEvent(QKeyEvent *e);
    void focusOutEvent(QFocusEvent *e) { Q_UNUSED(e); qDebug("MainWindow::focusOutEvent"); }

    QHash<QString, QMenu *> m_menus;
    QHash<QString, QAction *> m_actions;
    QList<QDockWidget *> m_dockWidgets;
    QList<SubWindow *> m_subWindows;
    QList<WindowAction *> m_windowActions;
    QMdiArea *m_mdiArea;

    AboutPluginsWindow *m_aboutPluginsWindow;
    PluginManager *m_pluginManager;
    int _x, _y, _state;


public slots:
    void actTabbedView(bool);
    void actCloseSubWindow();
    void actCloseAllSubWindows();
    void actAboutPlugins();
    void subWindowActivated(QMdiSubWindow *window);
    void subWindowClosed(QMdiSubWindow *window);
    void actionView(int index); //
    void actionWindow(int index);   //
    void onExit();
    void test(bool);
    void test2(int idx);
    void test3();
};

#endif // MAINWINDOW_H
