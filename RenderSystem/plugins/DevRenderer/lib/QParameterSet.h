#ifndef QPARAMETERSET_H
#define QPARAMETERSET_H

#include <QtCore>

#include "ParameterSet.h"

class QParameterSetConnecter;

// ParameterSet plus Qt features
class QParameterSet : public QObject, public ParameterSet
{
    Q_OBJECT
public:
    QParameterSet() : _psc(nullptr) {}
    virtual ~QParameterSet() { unbind(); }
    void bind(QParameterSetConnecter *psc);
    void unbind();

protected:
    virtual void valueChanged(const String &name);

protected:
    QParameterSetConnecter *_psc;

signals:
    void parameterChanged(const String &name);
};

// used to connect Qt widgets
class QParameterConnecter : public QObject
{
    Q_OBJECT
public:
    QParameterConnecter() {}
    QParameterConnecter(const QParameterConnecter &other) : _name(other._name), _type(other._type), _autoUpdateEnabled(other._autoUpdateEnabled), _parent(other._parent) {}
    void setName(const String &name) { _name = name; }
    void setType(Parameter::Type type) { _type = type; }
    void setAutoUpdate(bool enable) { _autoUpdateEnabled = enable; }
    void setParent(QParameterSetConnecter *p) { _parent = p; }
    const String &name() const { return _name; }
    Parameter::Type type() const { return _type; }

    Parameter *getParameter();

    /*void changeNotify(Parameter *param)
    {
        if (param == nullptr) return;
        switch (_type)
        {
        case Parameter::Bool : emit valueChanged(param->toBool());  break;
        case Parameter::Int  : emit valueChanged(param->toInt());   break;
        case Parameter::Float: emit valueChanged(param->toFloat()); break;
        default: break;
        }
    }*/

    void update()
    {
        if (!_autoUpdateEnabled)
        {
            qDebug("!autoUpdate: %s", _name.c_str());
            return;
        }
        Parameter *param = getParameter();
        if (param == nullptr) return;
        switch (_type)
        {
        case Parameter::BOOL_TYPE  : emit valueChanged(param->toBool());  break;
        case Parameter::INT_TYPE   : emit valueChanged(param->toInt());   break;
        case Parameter::FLOAT_TYPE : emit valueChanged(param->toFloat()); break;
        default: emit valueChanged(*getParameter()); emit valueChanged(); break;
        }
    }

protected:
    String _name;
    Parameter::Type _type;
    bool _autoUpdateEnabled;
    QParameterSetConnecter *_parent;

public slots:
    void setValue(bool  value);
    void setValue(int   value);
    void setValue(float value);

signals:
    void valueChanged();
    void valueChanged(bool  value);
    void valueChanged(int   value);
    void valueChanged(float value);
    void valueChanged(Parameter &param);
};

class QParameterSetConnecter : public QObject
{
    Q_OBJECT
public:
    QParameterSetConnecter() : _ps(nullptr) {}
    virtual ~QParameterSetConnecter() { unbind(); }
    void bind(QParameterSet *ps) { if (_ps != ps) { _ps = ps; if (_ps != nullptr) { _ps->bind(this); } } }
    void unbind() { if (_ps != nullptr) { QParameterSet *ps = _ps; _ps = nullptr; ps->unbind(); } }
    bool isBound() { return (_ps != nullptr); }

    void addParameterConnecter(const String &name)
    {
        if (_connecters.contains(name)) return;
        _connecters[name].setName(name);
        _connecters[name].setAutoUpdate(true);
        _connecters[name].setParent(this);
    }

    Parameter *getParameter(const String &name)
    {
        if (_ps == nullptr) return nullptr;
        //return &(*_ps)[name];
        return _ps->getParameter(name);
    }

    QParameterConnecter &operator [] (const String &name)
    {
        if (!_connecters.contains(name)) addParameterConnecter(name);
        return _connecters[name];
    }

protected:
    Hash<String, QParameterConnecter> _connecters;
    QParameterSet *_ps;

public slots:
    void update()
    {
        for (Hash<String, QParameterConnecter>::iterator it = _connecters.begin(); it != _connecters.end(); it++)
        {
            it->update();
        }
    }

    //void changeNotify(const String &name) { _connecters[name].changeNotify(getParameter(name)); }
    void update(const String &name) { if (_connecters.contains(name)) { _connecters[name].update(); } }
};

inline void QParameterSet::bind(QParameterSetConnecter *psc) { if (_psc != psc) { _psc = psc; if (_psc != nullptr) { _psc->bind(this); } } }
inline void QParameterSet::unbind() { if (_psc != nullptr) { QParameterSetConnecter *psc = _psc; _psc = nullptr; psc->unbind(); } }
inline void QParameterSet::valueChanged(const String &name) { if (_psc != nullptr) { _psc->update(name); } emit parameterChanged(name); }

inline Parameter *QParameterConnecter::getParameter() { if (_parent == nullptr) { return nullptr; } return _parent->getParameter(_name); }
inline void QParameterConnecter::setValue(bool  value) { if (_parent != nullptr && _parent->isBound()) { _parent->getParameter(_name)->setValue(value); } }
inline void QParameterConnecter::setValue(int   value) { if (_parent != nullptr && _parent->isBound()) { _parent->getParameter(_name)->setValue(value); } }
inline void QParameterConnecter::setValue(float value) { if (_parent != nullptr && _parent->isBound()) { _parent->getParameter(_name)->setValue(value); } }


#endif // QPARAMETERSET_H
