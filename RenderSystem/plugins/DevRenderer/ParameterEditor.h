#ifndef PARAMETEREDITOR_H
#define PARAMETEREDITOR_H

#include <QtGui>

//#include "QParameterSet.h"

/*class ParameterEditor
{
public:

protected:

};*/

class ScalarEditor : public QObject
{
    Q_OBJECT
public:
    ScalarEditor(const QString &text, QWidget *parent = 0)
        : QObject(parent)
    {
        _label = new QLabel(text, parent);
        _slider = new QSlider(Qt::Horizontal, parent);
        _spinBox = new QDoubleSpinBox(parent);
        connect(_slider, SIGNAL(valueChanged(int)), this, SLOT(_sliderValueChanged(int)));
        connect(_spinBox, SIGNAL(valueChanged(double)), this, SLOT(_spinBoxValueChanged(double)));
        _slider->setMinimum(0);
        //_spinBox->setMinimum(0.0);
        //_spinBox->setMaximum(1.0);
        setMinMax(0.0f, 1.0f);
        setSingleStep(0.01f);
    }
    QLabel *label() { return _label; }
    QSlider *slider() { return _slider; }
    QDoubleSpinBox *spinBox() { return _spinBox; }
    void setMinimum(float min) { _spinBox->setMinimum((double)min); _setSpinBoxSingleStep(); }
    void setMaximum(float max) { _spinBox->setMaximum((double)max); _setSpinBoxSingleStep(); }
    void setMinMax(float min, float max) { _spinBox->setMinimum((double)min); _spinBox->setMaximum((double)max); _setSpinBoxSingleStep(); }
    void setSingleStep(float val) { _slider->setMaximum((int)((_spinBox->maximum() - _spinBox->minimum()) / (double)val + 1.0e-3)); _setSpinBoxSingleStep(); }
    void setTotalSteps(int totalSteps) { _slider->setMaximum(totalSteps); _setSpinBoxSingleStep(); }
protected:
    void _setSpinBoxSingleStep() { _spinBox->setSingleStep((_spinBox->maximum() - _spinBox->minimum()) / (double)_slider->maximum()); _spinBoxValueChanged(_spinBox->value()); }
    QLabel *_label;
    QSlider *_slider;
    QDoubleSpinBox *_spinBox;
protected slots:
    void _sliderValueChanged(int val) { _spinBox->setValue(_spinBox->minimum() + _spinBox->singleStep() * val); emit valueChanged((float)_spinBox->value()); }
    void _spinBoxValueChanged(double val) { _slider->setValue((int)((val - _spinBox->minimum()) / _spinBox->singleStep() + 1.0e-3)); emit valueChanged((float)val); }
public slots:
    void setEnabled(bool enable) { _label->setEnabled(enable); _slider->setEnabled(enable); _spinBox->setEnabled(enable); }
    void setValue(float val) { _spinBox->setValue((double)val); }
signals:
    void valueChanged(float val);
};

class IntScalarEditor : public QObject
{
    Q_OBJECT
public:
    IntScalarEditor(const QString &text, QWidget *parent = 0)
        : QObject(parent)
    {
        _label = new QLabel(text, parent);
        _slider = new QSlider(Qt::Horizontal, parent);
        _spinBox = new QSpinBox(parent);
        connect(_slider, SIGNAL(valueChanged(int)), _spinBox, SLOT(setValue(int)));
        connect(_spinBox, SIGNAL(valueChanged(int)), _slider, SLOT(setValue(int)));
        connect(_spinBox, SIGNAL(valueChanged(int)), this, SIGNAL(valueChanged(int)));
        setMinMax(0, 100);
        setSingleStep(1);
    }
    QLabel *label() { return _label; }
    QSlider *slider() { return _slider; }
    QSpinBox *spinBox() { return _spinBox; }
    void setMinimum(int min) { _slider->setMinimum(min); _spinBox->setMinimum(min); }
    void setMaximum(int max) { _slider->setMaximum(max); _spinBox->setMaximum(max); }
    void setMinMax(int min, int max) { setMinimum(min); setMaximum(max); }
    void setSingleStep(int val) { _slider->setSingleStep(val); _spinBox->setSingleStep(val); }
protected:
    QLabel *_label;
    QSlider *_slider;
    QSpinBox *_spinBox;
public slots:
    void setEnabled(bool enable) { _label->setEnabled(enable); _slider->setEnabled(enable); _spinBox->setEnabled(enable); }
    void setValue(int val) { _spinBox->setValue(val); }
signals:
    void valueChanged(int val);
};

class ButtonGroup : public QButtonGroup
{
    Q_OBJECT
public:
    ButtonGroup(QWidget *parent = 0) : QButtonGroup(parent) {}
public slots:
    void toggle(int id) { if (id >= 0 && id < buttons().size()) button(id)->toggle(); }
};

#endif // PARAMETEREDITOR_H
