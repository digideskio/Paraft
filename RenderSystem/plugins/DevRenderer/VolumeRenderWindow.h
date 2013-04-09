#ifndef VOLUMERENDERWINDOW_H
#define VOLUMERENDERWINDOW_H

#include <GL/glew.h>
#include "QRenderWindow.h"
#include "CStructuredMeshData.h"
#include "VolumeParser.h"

#include <QtGui>

#include "GLShader.h"
#include "MSGLTexture.h"
#include "QParameterSet.h"
#include "MainUI.h"

#include "VolumeModel.h"
#include "PreIntegratorGL.h"
#include "RayCastingRenderer.h"
#include "PreIntegrationRenderer.h"
#include "SegmentedRayCastingRenderer.h"
#include "VolumeRenderer.h"
#include "SegmentedVolumeRenderer.h"

#include "UDPListener.h"


class VolumeRenderWindow : public QRenderWindow
{
    Q_OBJECT
public:
    VolumeRenderWindow(QWidget *parent = 0, const QGLWidget *shareWidget = 0);
    VolumeRenderWindow(VolumeModel *model, QWidget *parent = 0, const QGLWidget *shareWidget = 0);
    ~VolumeRenderWindow();

    void setWorkingPath(const QString &path) { _workingPath = path; }
    void setMainUI(MainUI *mainUI)           { _mainUI = mainUI; }
    VolumeModel *model() { return _model; }

    bool isActiveSubWindow() { QMdiSubWindow *win = qobject_cast<QMdiSubWindow *>(parentWidget()); return (win == win->mdiArea()->activeSubWindow()); }

protected:
    virtual void initializeGL();
    virtual void render();
    virtual void updateGL();
    virtual void resizeGL(int width, int height);

    virtual void mousePressEvent(QMouseEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void wheelEvent(QWheelEvent *e);
    virtual void keyPressEvent(QKeyEvent *e);
    virtual void keyReleaseEvent(QKeyEvent *e);
    virtual void focusInEvent(QFocusEvent *e);

    bool slicerEnabled() { return m_slicers.size() > 0; }
    void resizeBuffer(MSLib::GLFramebufferObject *&fbo, MSLib::GLTexture2D *&bufferTex, int width, int height);
    void reloadData();

    float sampleInterval() { return (_ps["sampleStep"].toFloat() * _model->scaledDim().length()); }
    int timeStep() { return (_ps["timestep"].toInt() - 1); }
    int varIndex() { return _ps["compIdx"].toInt(); }

    QString _workingPath;
    VolumeModel *_model;
    TVMVVolumeMetadata *_dataset;    //// determine who will delete this
    VolumeParser m_volumeParser;
    MainUI *_mainUI;
    QParameterSet _ps;

    VolumeData *m_volumeData;
    MSLib::GLTexture1D *_tfMapTex;

    // data for renderer
    CStructuredMeshData *m_data;
    float sampleSpacing;
    bool enableLight;
    LIGHTPARAM lightParam;

    Histogram *m_histogram;
    TF m_transferFunction;
    float m_vertex[8][3], m_texture[8][3];

    VolumeRenderer *_renderer;
    SegmentedVolumeRenderer *_segmentedRenderer;

    CameraOptions _defaultCamera;
    MSLib::GLFramebufferObject *_bufferFbo[2];
    MSLib::GLTexture2D *_bufferTex[2];
    GLShader *_copyColorShader;

    int _testMode;

    Vector2f _hand;
    int _state;



public slots:
    void activated();

    // QTFEditor
    void bgColorChanged(const QColor &);
    void tfChanged(float *colorMap, bool);
    void tfMappingChanged(float *, float *, bool);

    // QRenderEffEditor
    void sliceVectorChanged(Vector3);
    void timestepChanged(int);

    // ControlPanels
    void parameterChanged(const String &name);
    void actionTriggered(const QString &name);

    ////
    void cameraTrack(int x, int y, int state);

};

#endif // VOLUMERENDERWINDOW_H
