#include "MainWindow.h"

#define PLUGINS_PATH "./plugins"
#define PLUGINS_LIST_FILE "plugins.txt"

PluginManager::PluginManager(MainWindow *mainWindow)
    : m_mainWindow(mainWindow),
      m_pluginsPath(mainWindow->getPluginsPath()) {
}

PluginManager::~PluginManager() {
    qDebug("~PluginManager()");
    for (int i = 0; i < m_plugins.size(); i++) {
        if (m_plugins[i].loaded)
            delete m_plugins[i].instance;
    }
}

bool PluginManager::loadPlugins() {
    bool dirty = false;
    for (int i = 0; i < m_plugins.size(); i++) {
        Plugin &plugin = m_plugins[i];
        if (!plugin.enabled) continue;
        QPluginLoader pluginLoader(plugin.fileName);
        QObject *pluginInstance = pluginLoader.instance();
        if (pluginInstance == NULL) {
            qDebug("Loading \"%s\" failed", plugin.fileName.toAscii().constData());
            continue;
        }
        plugin.instance = qobject_cast<PluginInterface *>(pluginInstance);
        if (plugin.instance == NULL) {
            qDebug("Loading \"%s\" failed", plugin.fileName.toAscii().constData());
            continue;
        }

        plugin.instance->setMainInterface(m_mainWindow);
        plugin.instance->init();
        if (plugin.name != plugin.instance->getPluginName()) {
            dirty = true;
            plugin.name = plugin.instance->getPluginName();
        }
        if (plugin.version != plugin.instance->getPluginVersion()) {
            dirty = true;
            plugin.version = plugin.instance->getPluginVersion();
        }
        plugin.loaded = true;

        qDebug("Loading \"%s\" successful", plugin.fileName.toAscii().constData());
    }
    if (dirty && !writeList()) {
        qDebug("Writing to plugins file failed.");
        return false;
    }
    return true;
}

// read plugins list from "plugins.txt"
bool PluginManager::readList(QString fileName) {
    if (fileName == "")
        fileName = QString(PLUGINS_LIST_FILE);
    QDir pluginsDir(m_pluginsPath);
    QFile file(pluginsDir.filePath(fileName));
    if (!file.open(QIODevice::ReadOnly)) {
        // no "plugins.txt", try every file in plugins directory
        foreach (QString fileName, pluginsDir.entryList(QDir::Files))
            m_plugins.append(Plugin(QString(""), QString(""), pluginsDir.filePath(fileName), true));
        return false;
    }
    while (!file.atEnd()) {
        QString line = file.readLine();
        if (line.contains(QRegExp("^\\s#"))) continue;  // comment
        QStringList list = line.split(",");
        Plugin plugin(list[1].trimmed(), list[2].trimmed(), list[3].trimmed(), list[0].trimmed().toInt() != 0);
        //
        //qDebug("%d,%s,%s,%s", plugin.isEnabled, plugin.name.toAscii().constData(), plugin.version.toAscii().constData(), plugin.fileName.toAscii().constData());
        m_plugins.append(plugin);
    }
    return true;
}

// write plugins list to "plugins.txt"
bool PluginManager::writeList(QString fileName)
{
    if (fileName == "")
        fileName = QString(PLUGINS_LIST_FILE);
    QDir pluginsDir(m_pluginsPath);
    QFile file(pluginsDir.filePath(fileName));
    if (!file.open(QIODevice::WriteOnly)) return false;
    for (int i = 0; i < m_plugins.size(); i++)
    {
        Plugin &plugin = m_plugins[i];
        file.write(QString("%1, %2, %3, %4\r\n").arg(plugin.enabled ? 1 : 0).arg(plugin.name, plugin.version, plugin.fileName).toAscii().constData());
    }
    qDebug("Writing to \"%s\" successful.", fileName.toAscii().constData());
    return true;
}

PluginInterface *PluginManager::getPlugin(const QString &name)
{
    for (int i = 0; i < m_plugins.size(); i++)
    {
        if (m_plugins[i].name == name)
            return m_plugins[i].instance;
    }
    return 0;
}

void WindowAction::windowStateChanged(Qt::WindowStates oldState, Qt::WindowStates newState)
{
    setChecked(newState & Qt::WindowActive);
    if (!(oldState & Qt::WindowMinimized) && (newState & Qt::WindowMinimized))
        //m_mainWindow->setSubWindowFocus(m_index);
        //(*m_windowList)[m_index + 1 < m_windowList->size() ? m_index + 1 : 0]->setFocus();
        m_window->mdiArea()->activateNextSubWindow();
    //MainWindow::mainWindow->setWindowTitle(WINDOW_TITLE);
}

AboutPluginsWindow::AboutPluginsWindow(PluginManager *pluginManager, QWidget* parent)
    : QWidget(parent), m_pluginManager(pluginManager)
{
    setWindowFlags(Qt::Dialog);
    setWindowModality(Qt::WindowModal);

    setWindowTitle(tr("About Plugins"));
    resize(640, 480);

    m_layout = new QVBoxLayout();
    //m_tableView = new QTableView(this);
    //m_tableView->setModel(new PluginsTableModel());

    m_table = new QTableWidget(m_pluginManager->count(), 3, this);

    m_table->verticalHeader()->setVisible(false);
    m_table->setHorizontalHeaderItem(0, new QTableWidgetItem(tr("Name")));
    m_table->setHorizontalHeaderItem(1, new QTableWidgetItem(tr("Version")));
    m_table->setHorizontalHeaderItem(2, new QTableWidgetItem(tr("Location")));
    m_table->setColumnWidth(0, 200);
    m_table->setColumnWidth(1, 80);
    m_table->setColumnWidth(2, 300);

    for (int i = 0; i < m_table->rowCount(); i++)
    {
        QTableWidgetItem *name = new QTableWidgetItem(m_pluginManager->name(i));
        QTableWidgetItem *version = new QTableWidgetItem(m_pluginManager->version(i));
        QTableWidgetItem *location = new QTableWidgetItem(m_pluginManager->fileName(i));
        name->setFlags(Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsTristate);
        version->setFlags(Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | Qt::ItemIsEnabled);
        location->setFlags(Qt::ItemIsEditable | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | Qt::ItemIsEnabled);
        m_table->setItem(i, 0, name);
        m_table->setItem(i, 1, version);
        m_table->setItem(i, 2, location);

        if (!m_pluginManager->isEnabled(i))
            m_table->item(i, 0)->setCheckState(Qt::Unchecked);
        else
        {
            if (!m_pluginManager->isLoaded(i))
                m_table->item(i, 0)->setCheckState(Qt::PartiallyChecked);
            else
                m_table->item(i, 0)->setCheckState(Qt::Checked);
        }
    }

    connect(m_table, SIGNAL(cellChanged(int, int)), this, SLOT(cellChanged(int, int)));

    m_layout->addWidget(m_table);

    m_layoutBottom = new QHBoxLayout();
    m_layoutBottom->addStretch();
    //m_btnOK = new QPushButton(tr("OK"));
    //m_btnCancel = new QPushButton(tr("Cancel"));
    m_btnClose = new QPushButton(tr("Close"));
    //connect(m_btnOK, SIGNAL(clicked()), this, SLOT(btnOK()));
    //connect(m_btnCancel, SIGNAL(clicked()), this, SLOT(btnCancel()));
    connect(m_btnClose, SIGNAL(clicked()), this, SLOT(btnClose()));
    //m_layoutBottom->addWidget(m_btnOK);
    //m_layoutBottom->addWidget(m_btnCancel);
    m_layoutBottom->addWidget(m_btnClose);

    m_layout->addLayout(m_layoutBottom);


    setLayout(m_layout);
}

void AboutPluginsWindow::show()
{
    m_dirty = false;
    QWidget::show();
}

void AboutPluginsWindow::cellChanged(int row, int column)
{
    m_dirty = true;
    switch (column)
    {
    case 0:
        m_pluginManager->setEnabled(row, m_table->item(row, column)->checkState() != Qt::Unchecked);
        break;
    }
}

void AboutPluginsWindow::btnOK()
{
    this->hide();
}

void AboutPluginsWindow::btnCancel()
{
    this->hide();
}

void AboutPluginsWindow::btnClose()
{
    if (m_dirty) m_pluginManager->writeList();
    this->hide();
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
#ifdef Q_WS_MAC
    QDir::setCurrent("../../..");
#endif

    //
    //mainWindow = this;

    setWindowTitle("RenderSystem (x64)");

    initLayout();

    //loadPlugins();
    m_pluginManager = new PluginManager(this);
    m_pluginManager->readList();
    m_pluginManager->loadPlugins();

    m_aboutPluginsWindow = new AboutPluginsWindow(m_pluginManager, this);

    //
    connect(m_mdiArea, SIGNAL(subWindowActivated(QMdiSubWindow *)), this, SLOT(subWindowActivated(QMdiSubWindow *)));
}

MainWindow::~MainWindow()
{
    foreach (QMenu *menu, m_menus)
        if (menu) delete menu;
    foreach (QAction *action, m_actions)
        if (action) delete action;

    foreach (QDockWidget *dock, m_dockWidgets)
        delete dock;

    delete m_mdiArea;

    delete m_pluginManager;

    qDebug("Exiting");
}

QString MainWindow::getPluginsPath()
{
    return QString(PLUGINS_PATH);
}

QDockWidget *MainWindow::addDockWidget(const QString &title, QWidget *widget, Qt::DockWidgetArea area, bool show)
{
    QDockWidget *dock = new QDockWidget(title, this);
    dock->setWidget(widget);
    QMainWindow::addDockWidget(area, dock);
    if (show)
        dock->show();
    else
        dock->hide();
    m_dockWidgets.append(dock);

    /*
    //Action *act = new Action(title, m_dockWidgetList.size(), this);
    QAction *act = new QAction(title, this);
    act->setCheckable(true);
    //act->setChecked(show);
    m_actions[QString("View.%1").arg(title)] = act;
    m_menus["View"]->addAction(act);
    //connect(act, SIGNAL(triggeredIdx(int)), this, SLOT(actionView(int)));

    connect(dock, SIGNAL(visibilityChanged(bool)), act, SLOT(setChecked(bool)));
    connect(act, SIGNAL(triggered(bool)), dock, SLOT(setVisible(bool)));
    */

    QAction *act = dock->toggleViewAction();
    m_actions[QString("View.%1").arg(title)] = act;
    m_menus["View"]->addAction(act);

    return dock;
}

//QMdiSubWindow *MainWindow::addSubWindow(const QString &windowTitle, QWidget *widget, PluginInterface *plugin)
QMdiSubWindow *MainWindow::addSubWindow(QWidget *widget, PluginInterface *plugin)
{
    //QMdiSubWindow *win = new QMdiSubWindow();
    SubWindow *win = new SubWindow();
    win->setAttribute(Qt::WA_DeleteOnClose, true);
    //win->setWindowTitle(windowTitle);
    win->setWindowTitle(widget->windowTitle());
    win->setWidget(widget);
    //widget->setParent(win);
    //win->layout()->addWidget(widget);
    win->setParentPlugin(plugin);
    m_mdiArea->addSubWindow(win);
    m_subWindows.append(win);

    //WindowAction *act = new WindowAction(windowTitle, m_mdiSubWindowList.size(), this, this);
    //WindowAction *act = new WindowAction(QString("&%1 %2").arg(m_subWindows.size()).arg(windowTitle), win, this);
    WindowAction *act = new WindowAction(QString("&%1 %2").arg(m_subWindows.size()).arg(widget->windowTitle()), win, this);
    //act->setCheckable(true);
    //act->setChecked(true);
    //QString actName(QString("Window.%1").arg(windowTitle));
    //m_actions[QString("Window.%1").arg(windowTitle)] = act;

    //m_actions[actName]->setCheckable(true);
    //m_actions[actName]->setChecked(true);
    m_menus["Window"]->addAction(act);
    m_windowActions.append(act);

    //act->setCheckWhenTriggered();

    connect(win, SIGNAL(windowStateChanged(Qt::WindowStates, Qt::WindowStates)), act, SLOT(windowStateChanged(Qt::WindowStates, Qt::WindowStates)));
    //connect(act, SIGNAL(triggered()), win, SLOT(setFocus()));
    connect(act, SIGNAL(triggered(QMdiSubWindow *)), m_mdiArea, SLOT(setActiveSubWindow(QMdiSubWindow *)));

    connect(win, SIGNAL(windowClosed(QMdiSubWindow *)), this, SLOT(subWindowClosed(QMdiSubWindow *)));

    //if (!m_actions["Window.Close"]->isEnabled())
    m_actions["Window.Close"]->setEnabled(true);
    //if (!m_actions["Window.CloseAll"]->isEnabled())
    m_actions["Window.CloseAll"]->setEnabled(true);

    return win;
}

void MainWindow::addToolBar(QToolBar *toolbar, Qt::ToolBarArea area)
{
    QMainWindow::addToolBar(area, toolbar);

    ////
}

void MainWindow::tabifyDockWidget(QDockWidget *first, QDockWidget *second)
{
    QMainWindow::tabifyDockWidget(first, second);
}

/*void MainWindow::setSubWindowFocus(int index)
{
    m_mdiArea->activateNextSubWindow();
    //m_mdiSubWindowList[index + 1 < m_mdiSubWindowList.size() ? index + 1 : 0]->setFocus();
}*/

void MainWindow::initLayout()
{
    resize(1901, 1032);

    m_mdiArea = new QMdiArea(this);
    m_mdiArea->setDocumentMode(true);
    //m_mdiArea->setTabsClosable(true);       // Qt-4.8.0
    m_mdiArea->setTabsMovable(true);        // Qt-4.8.0
    setCentralWidget(m_mdiArea);

    m_menus["File"] = new QMenu(tr("&File"), this);
    m_menus["View"] = new QMenu(tr("&View"), this);
    m_menus["Tools"] = new QMenu(tr("&Tools"), this);
    m_menus["Window"] = new QMenu(tr("&Window"), this);
    m_menus["Help"] = new QMenu(tr("&Help"), this);

    menuBar()->addMenu(m_menus["File"]);
    menuBar()->addMenu(m_menus["View"]);
    menuBar()->addMenu(m_menus["Tools"]);
    menuBar()->addMenu(m_menus["Window"]);
    menuBar()->addMenu(m_menus["Help"]);

    m_actions["File.-"] = new QAction(this);
    m_actions["File.-"]->setSeparator(true);
    m_menus["File"]->addAction(m_actions["File.-"]);

    m_actions["File.Exit"] = new QAction(tr("E&xit"), this);
    m_menus["File"]->addAction(m_actions["File.Exit"]);
    connect(m_actions["File.Exit"], SIGNAL(triggered()), this, SLOT(onExit()));

    // ! test
    //m_actions["Tools.Test"] = new QAction(tr("&Test"), this);
    //m_menus["Tools"]->addAction(m_actions["Tools.Test"]);
    //connect(m_actions["Plugins.Test"], SIGNAL(triggered()), this, SLOT(test()));
    //

    m_actions["Window.TabbedView"] = new QAction(tr("Tabbed &View"), this);
    m_actions["Window.TabbedView"]->setCheckable(true);
    m_actions["Window.TabbedView"]->setChecked(false);
    m_menus["Window"]->addAction(m_actions["Window.TabbedView"]);
    connect(m_actions["Window.TabbedView"], SIGNAL(triggered(bool)), this, SLOT(actTabbedView(bool)));

    m_actions["Window.Cascade"] = new QAction(tr("&Cascade"), this);
    m_menus["Window"]->addAction(m_actions["Window.Cascade"]);
    connect(m_actions["Window.Cascade"], SIGNAL(triggered()), m_mdiArea, SLOT(cascadeSubWindows()));
    m_actions["Window.Tile"] = new QAction(tr("&Tile"), this);
    m_menus["Window"]->addAction(m_actions["Window.Tile"]);
    connect(m_actions["Window.Tile"], SIGNAL(triggered()), m_mdiArea, SLOT(tileSubWindows()));

    m_actions["Window.-[0]"] = new QAction(this);
    m_actions["Window.-[0]"]->setSeparator(true);
    m_menus["Window"]->addAction(m_actions["Window.-[0]"]);

    m_actions["Window.Close"] = new QAction(tr("C&lose"), this);
    m_menus["Window"]->addAction(m_actions["Window.Close"]);
    connect(m_actions["Window.Close"], SIGNAL(triggered()), this, SLOT(actCloseSubWindow()));
    m_actions["Window.CloseAll"] = new QAction(tr("Close &All"), this);
    m_menus["Window"]->addAction(m_actions["Window.CloseAll"]);
    connect(m_actions["Window.CloseAll"], SIGNAL(triggered()), this, SLOT(actCloseAllSubWindows()));
    if (m_subWindows.empty())
    {
        m_actions["Window.Close"]->setDisabled(true);
        m_actions["Window.CloseAll"]->setDisabled(true);
    }

    m_actions["Window.-[1]"] = new QAction(this);
    m_actions["Window.-[1]"]->setSeparator(true);
    m_menus["Window"]->addAction(m_actions["Window.-[1]"]);

    m_actions["Help.About"] = new QAction(tr("&About"), this);
    m_menus["Help"]->addAction(m_actions["Help.About"]);
    m_actions["Help.AboutPlugins"] = new QAction(tr("About &Plugins..."), this);
    m_menus["Help"]->addAction(m_actions["Help.AboutPlugins"]);
    connect(m_actions["Help.AboutPlugins"], SIGNAL(triggered()), this, SLOT(actAboutPlugins()));


    //tabifyDockWidget(m_dockWidget, dock);




}

void MainWindow::keyPressEvent(QKeyEvent *e)
{
    qDebug("MainWindow::keyPressEvent(%d)", e->key());

    ////
    if (e->key() == Qt::Key_F12)
    {
        PluginInterface *plugin = getActivePlugin();
        if (plugin == 0)
            qDebug("No active plugin");
        else
            qDebug("%s is active", plugin->getPluginName().toAscii().constData());
    }
    else if (e->key() == Qt::Key_U)
    {
        qDebug("Switch state");
        if (_state != 1)
        {
            _state = 1;
            _x = 320;
            _y = 240;
        }
        else
        {
            _state = 0;
        }
    }
    else if (e->key() == Qt::Key_I)
    {
        qDebug("Up");
        _y -= 10;
    }
    else if (e->key() == Qt::Key_K)
    {
        qDebug("Down");
        _y += 10;
    }
    else if (e->key() == Qt::Key_J)
    {
        qDebug("Left");
        _x -= 10;
    }
    else if (e->key() == Qt::Key_L)
    {
        qDebug("Right");
        _x += 10;
    }

    ////
    if (e->key() == Qt::Key_U || e->key() == Qt::Key_I || e->key() == Qt::Key_K ||
        e->key() == Qt::Key_J || e->key() == Qt::Key_L)
    {
        QList<QVariant> args;
        args.append(QVariant(_x));
        args.append(QVariant(_y));
        args.append(QVariant(_state));
        QString method = "cameraTrack";
        QVariant arg(args);
        getActivePlugin()->request(method, arg);
    }
    ////

    for (int i = 0; i < m_pluginManager->count(); i++)
        if (m_pluginManager->isLoaded(i))
            m_pluginManager->getPlugin(i)->keyPressEvent(e);
    /*if (e->key() == Qt::Key_F11)
    {
        qDebug("plugin: %s", m_pluginManager->getPlugin(3)->getPluginName().toAscii().constData());
        QImage img;
        QVariant var(img);
        bool ok = m_pluginManager->getPlugin(3)->request("getScreenshot()", var);
        if (ok)
        {
            qDebug("Save screenshot");
            img = qvariant_cast<QImage>(var);

            int ret = img.save("screenshot.png", "PNG");
            qDebug("Success: %s", ret ? "true" : "false");
        }
    }
    else*/
    //QMainWindow::keyPressEvent(e);
}

/*
void MainWindow::loadPlugins()
{
//#if defined(Q_WS_WIN) || defined(Q_WS_MAC)
//    QDir pluginsDir(QDir::currentPath());
//#else
//    QDir pluginsDir(qApp->applicationDirPath());
//#endif

    //QString dirname(tr("plugins"));
    //pluginsDir.cd(dirname);

    QStringList fileList;

    //pluginsDir.cd(getPluginsDir());
    QDir pluginsDir(getPluginsDir());

    // read "plugins.txt"
    //QFile pluginsFile(pluginsDir.absoluteFilePath("plugins.txt"));
    QFile pluginsFile(pluginsDir.filePath("plugins.txt"));
    if (pluginsFile.open(QIODevice::ReadOnly))
    {
        while (!pluginsFile.atEnd())
        {
            QString line = pluginsFile.readLine().trimmed();
            if (line[0] != '#')
                //fileList.append(pluginsDir.absoluteFilePath(line));
                fileList.append(pluginsDir.filePath(line));
        }
    }
    else    // no "plugins.txt", try to load all files
    {
        foreach (QString fileName, pluginsDir.entryList(QDir::Files))
            //fileList.append(pluginsDir.absoluteFilePath(fileName));
            fileList.append(pluginsDir.filePath(fileName));
    }

    // load plugins
    for (QStringList::iterator it = fileList.begin(); it != fileList.end(); it++)
    {

        QPluginLoader pluginLoader(*it);
        QObject *pluginInstance = pluginLoader.instance();
        if (!pluginInstance)
        {
            qDebug("Loading \"%s\" failed", (*it).toAscii().constData());
            continue;
        }
        PluginInterface *plugin = qobject_cast<PluginInterface *>(pluginInstance);
        if (!plugin)
        {
            qDebug("Loading \"%s\" failed", (*it).toAscii().constData());
            continue;
        }

        //
        plugin->setMainInterface(this);
        plugin->init();
        PluginData_a pluginData;
        pluginData.fileName = (*it);
        pluginData.isLoaded = true;
        pluginData.plugin = plugin;
        m_plugins.append(pluginData);
        //m_plugin = plugin;

        qDebug("Loading \"%s\" successfully", (*it).toAscii().constData());
    }
}
*/

void MainWindow::actTabbedView(bool checked)
{
    if (checked)
        m_mdiArea->setViewMode(QMdiArea::TabbedView);
    else
        m_mdiArea->setViewMode(QMdiArea::SubWindowView);
}

void MainWindow::actCloseSubWindow()
{
    m_mdiArea->closeActiveSubWindow();
}

void MainWindow::actCloseAllSubWindows()
{
    while (!m_subWindows.empty())
        m_subWindows.first()->close();
}

void MainWindow::actAboutPlugins()
{
    //AboutPluginsWindow *apwin = new AboutPluginsWindow(m_pluginManager);
    //apwin->show();
    m_aboutPluginsWindow->show();

}

void MainWindow::subWindowActivated(QMdiSubWindow *window)
{
    if (window == 0)
    {
        qDebug("subWindowActivated 0");
        return;
    }
    qDebug("subWindowActivated %s", window->windowTitle().toAscii().constData());
    SubWindow *win = qobject_cast<SubWindow *>(window);
    //win->getParentPlugin()->subWindowActivated(win->windowTitle(), win->widget());  ////
    win->getParentPlugin()->subWindowActivated(win->widget());                      ////
}

void MainWindow::subWindowClosed(QMdiSubWindow *window)
{
    //m_mdiArea->removeSubWindow(window);
    //QMdiSubWindow *win = qobject_cast<QMdiSubWindow *>(obj);
    SubWindow *win = qobject_cast<SubWindow *>(window);
    int index = m_subWindows.indexOf(win);
    //qDebug("index=%d name=%s", m_mdiSubWindows.indexOf(window), window->windowTitle().toAscii().constData());
    m_menus["Window"]->removeAction(m_windowActions[index]);
    m_windowActions.removeAt(index);
    m_subWindows.removeAt(index);
    if (m_subWindows.empty())
    {
        m_actions["Window.Close"]->setDisabled(true);
        m_actions["Window.CloseAll"]->setDisabled(true);
    }
    for (int i = index; i < m_windowActions.size(); i++)
        m_windowActions[i]->setText(QString("&%1 %2").arg(i + 1).arg(m_subWindows[i]->windowTitle()));
    //win->getParentPlugin()->subWindowClosed(win->windowTitle(), win->widget());
    win->getParentPlugin()->subWindowClosed(win->widget());
}


//
void MainWindow::actionView(int index)
{
    qDebug("actionView %d", index);
    if (m_dockWidgets[index]->isVisible())
        m_dockWidgets[index]->hide();
    else
        m_dockWidgets[index]->show();
}

/*void MainWindow::subWindowStateChanged(Qt::WindowStates oldState, Qt::WindowStates newState)
{
    qDebug("subWindowStateChanged");

}*/

//
void MainWindow::actionWindow(int index)
{

    m_subWindows[index]->setFocus();
}

void MainWindow::onExit()
{
    //qApp->exit();
    qApp->closeAllWindows();
}

void MainWindow::test(bool flag)
{
    qDebug("test: %s", flag ? "true" : "false");
    //m_plugin->sayHello();
}

void MainWindow::test2(int idx)
{
    qDebug("test2 %d", idx);
}

void MainWindow::test3()
{
    qDebug("Test 3");
}
