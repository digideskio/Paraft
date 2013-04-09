#include "DevRenderer.h"
#include "QTFEditor.h"

DevRenderer::~DevRenderer() {
    qDebug("~DevRenderer()");
    delete _mainUI;
}

void DevRenderer::init() {
    setPluginName(PLUGIN_NAME);
    setPluginVersion(PLUGIN_VERSION);

    _pluginPath = getPluginPath();
    _prefFileName = _pluginPath + "/preferences.txt";

    QFileInfo prefFI(_prefFileName);
    if (prefFI.exists()) {
        Json::Parser parser;
        parser.parseFile(prefFI.filePath().toStdString(), _pref);
        if (_pref.contains("recentFiles") && _pref["recentFiles"].isArray()) {
            for (int i = 0; i < _pref["recentFiles"].size(); i++) {
                _recentFiles.append(QString::fromStdString(_pref["recentFiles"][i].toString()));
            }
        }
    } else {
        _pref["recentFiles"].resize(0);
    }

    _mainUI = new MainUI(_pluginPath);
    getMainInterface()->addToolBar(_mainUI->getGeneralToolBar());
    connect(_mainUI->getGeneralToolBar()->getOpenAction(), SIGNAL(triggered()), this, SLOT(open()));
    addDockWidget(tr("Render Effect"), _mainUI->getRenderEffectPanel());
    addDockWidget(tr("Transfer Function"), _mainUI->getTFEditor());

    QMenu *fileMenu = getMenu("File");
    QAction *sepAct = getAction("File.-");
    QAction *openAct = new QAction(tr("Open..."), fileMenu);
    fileMenu->insertAction(sepAct, openAct);
    connect(openAct, SIGNAL(triggered()), this, SLOT(open()));

    _recentFilesMenu = new QMenu(tr("&Recent Files"), fileMenu);
    fileMenu->insertMenu(sepAct, _recentFilesMenu);
    for (int i = 0; i < _recentFiles.size(); i++) {
        IndexedAction *act = new IndexedAction(_recentFiles[i], i, _recentFilesMenu);
        _recentFilesMenu->addAction(act);
        connect(act, SIGNAL(triggered(int)), this, SLOT(openRecentFile(int)));
    }

    _activeWindow = 0;
    _udpClient = new UDPListener();
    _udpClient->initSocket();
    connect(_udpClient, SIGNAL(handEvent(int, int, int)), this, SLOT(cameraTrack(int, int, int)));
}

void DevRenderer::subWindowActivated(QWidget *widget) {
    VolumeRenderWindow *win = qobject_cast<VolumeRenderWindow *>(widget);
    if (win != 0) {
        _activeWindow = win;
        win->activated();
    }
}

void DevRenderer::subWindowClosed(QWidget *widget) {
    qDebug("%s closed.", widget->windowTitle().toAscii().constData());
    VolumeRenderWindow *win = qobject_cast<VolumeRenderWindow *>(widget);
    if (win != 0) {
        if (_activeWindow == win) {
            _activeWindow = 0;
        }
        delete win->model();
    }
    delete widget;
}

bool DevRenderer::request(const QString &method, QVariant &arg) {
    if (method == "cameraTrack") {
        QList<QVariant> args = arg.toList();
        if (args.size() < 3) {
            return false;
        }
        int x = args[0].toInt();
        int y = args[1].toInt();
        int state = args[2].toInt();
        qDebug("x = %d, y = %d, state = %d", x, y, state);
        _activeWindow->cameraTrack(x, y, state);
        return true;
    }
    return false;
}

void DevRenderer::updateRecentFileList(const QString &filePath) {
    qDebug("Recent file: %s", filePath.toAscii().constData());
    int index = _recentFiles.indexOf(filePath);
    if (index >= 0) {
        _recentFiles.removeAt(index);
        qDebug("Remove recent file %d: %s", index, filePath.toAscii().constData());
    } else if (_recentFiles.size() >= 10) { // at most 10 items in the list
        _recentFiles.removeLast();
    }
    _recentFiles.prepend(filePath);

    writePreferencesFile();

    // update recent files menu
    _recentFilesMenu->clear();
    for (int i = 0; i < _recentFiles.size(); i++) {
        IndexedAction *act = new IndexedAction(_recentFiles[i], i, _recentFilesMenu);
        _recentFilesMenu->addAction(act);
        connect(act, SIGNAL(triggered(int)), this, SLOT(openRecentFile(int)));
    }
}

void DevRenderer::writePreferencesFile() {
    _pref["recentFiles"].clear();
    for (int i = 0; i < _recentFiles.size(); i++) {
        _pref["recentFiles"].append(_recentFiles[i].toStdString());
    }
    Json::Parser parser;
    parser.writeFile(QString(_pluginPath + "/preferences.txt").toStdString(), _pref);
}

void DevRenderer::act() {
    qDebug("act");
}

void DevRenderer::open(const QString &path) {
    VolumeModel *model = new VolumeModel(path.toStdString());

    qDebug("Reading metadata file... done.");

    updateRecentFileList(path);

    VolumeRenderWindow *vrwin = new VolumeRenderWindow(model);
    vrwin->setWorkingPath(_pluginPath);
    vrwin->setMainUI(_mainUI);
    QMdiSubWindow *win = addSubWindow(vrwin);

    win->resize(782, 813);      // 768x768
    win->setMinimumSize(256, 256);
    win->show();
}

void DevRenderer::open()
{
    QString path = QFileDialog::getOpenFileName(0, tr("Open"), _pluginPath, tr("Volume Data Descriptor File (*.txt)"));
    if (path != "")
        open(path);
}

void DevRenderer::openRecentFile(int index) {
    open(QString(_recentFiles[index]));     // content of recentFiles would be modified implicitly
}

void DevRenderer::cameraTrack(int x, int y, int state) {
    QList<QVariant> args;
    args.append(QVariant(x));
    args.append(QVariant(y));
    args.append(QVariant(state));
    QString method = "cameraTrack";
    QVariant arg(args);
    request(method, arg);
}

void DevRenderer::test() {
    Json::Parser parser;
    Json::Value root;
    std::string doc = " /*adfdgfg*/  \n{\"aaa\": [123 ,[],  \"abcde\"  //fgfgd  \n,true], \"bbb\" :{}, \"ccc\":false";
    try {
        parser.parseFile("testjson.txt", root);
    } catch (Json::ParseException &e) {
        std::cout << "Error: " << e.what() << " (" << e.line() << ")" << std::endl;
    } catch (Json::Exception &e) {
        std::cout << "Error 2: " << e.what() << std::endl;
    }

    std::cout << ">>" << root << "<<" << std::endl;
    //std::cout << "line " << parser.getLineNumber() << std::endl;

    try {
        for (Json::Value::Iterator it = root.begin(); it != root.end(); it++) {
            std::cout << it.key() << std::endl;
        } for (Json::ValueIterator it = root["abc"].begin(); it != root["abc"].end(); ++it) {
            (*it) = 10.0;
            std::cout << it->toDouble() << std::endl;
        }
    } catch (Json::Exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    Json::Value root2;
    root2 = root;
    std::cout << "===== root2 ====" << std::endl << root2 << std::endl;
    if (root == root2)
        std::cout << "equal!" << std::endl;

    Json::Value a;
    a["aaa"] = 3.5;
    a["bbb"] = "xxxx";
    Json::Value b;
    b["aaa"] = 3.5;
    b["bbb"] = "xxxx";

    try {
        std::cout << (a == b) << std::endl;
    } catch (Json::ValueTypeException &e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    Json::Value c;
    c.resize(5);
    c[2] = 30;
    c[3] = "jkl";
    c[6] = 20;

    const Json::Value &c0 = c[0];

    std::cout << c0["abc"] << std::endl;
    std::cout << c << std::endl;
}

Q_EXPORT_PLUGIN2(DevRendererPlugin, DevRenderer);
