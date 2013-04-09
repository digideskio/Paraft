#ifndef PARAMETERSET_H
#define PARAMETERSET_H

#include <string>

////
#include <QtCore>

#include "MSVectors.h"
#include "Containers.h"

#define nullptr 0   // C++11

typedef std::string String;

class Parameter;
class ParameterDir;

class ParameterListener
{
public:
    virtual void valueChanged(Parameter *param) = 0;
};

class Parameter
{
public:
    enum Type
    {
        NULL_TYPE = 0,
        DIR,
        BOOL_TYPE,
        INT_TYPE,
        FLOAT_TYPE,
        DOUBLE_TYPE,
        VECTOR2F,
        VECTOR2D,
        VECTOR3F,
        VECTOR3D,
        VECTOR4F,
        VECTOR4D
    };

    Parameter() : _type(NULL_TYPE), _listener(nullptr) {}
    //Parameter(bool  value) : _type(BOOL_TYPE),  _listener(nullptr) { _value.b = value; }
    //Parameter(int   value) : _type(INT_TYPE),   _listener(nullptr) { _value.i = value; }
    //Parameter(float value) : _type(FLOAT_TYPE), _listener(nullptr) { _value.f = value; }
    virtual ~Parameter() {}
    explicit Parameter(const String &name) : _name(name), _type(NULL_TYPE), _listener(nullptr) {}

    Parameter(const Parameter &other) { *this = other; }
    void setName(const String &name) { _name = name; }
    void setListener(ParameterListener *listener) { _listener = listener; }
    const String &name() const { return _name; }
    Type type() const { return _type; }

    //template <typename T> void setValue(const T &) {}   // should not use this
    //template <> void setValue(const bool  &value) { if (_type != Bool  || _bvalue != value) { _type = Bool;  _bvalue = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }
    //template <> void setValue(const int   &value) { if (_type != Int   || _ivalue != value) { _type = Int;   _ivalue = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }
    //template <> void setValue(const float &value) { if (_type != Float || _fvalue != value) { _type = Float; _fvalue = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }
    void setValue(const bool  &value) { if (_type !=  BOOL_TYPE || _value.b != value) { _type =  BOOL_TYPE; _value.b = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }
    void setValue(const int   &value) { if (_type !=   INT_TYPE || _value.i != value) { _type =   INT_TYPE; _value.i = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }
    void setValue(const float &value) { if (_type != FLOAT_TYPE || _value.f != value) { _type = FLOAT_TYPE; _value.f = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }
    const bool  &toBool()  const { return _value.b; }
    const int   &toInt()   const { return _value.i; }
    const float &toFloat() const { return _value.f; }
    //template <typename T> const T &value() const { /* Error */ return *reinterpret_cast<const T *>(&_ivalue); }
    //template <> const bool  &value<bool>()  const { return _bvalue; }
    //template <> const int   &value<int>()   const { return _ivalue; }
    //template <> const float &value<float>() const { return _fvalue; }

    void clear()
    {
        switch (_type)
        {
        }
        _type = NULL_TYPE;
    }

    /* // legacy
    Parameter &operator = (const Parameter &other)
    {
        _name = other._name;
        _type = other._type;
        _listener = other._listener;
        switch (_type)
        {
        case VOID_TYPE: break;
        case DIR: break;    ////
        case BOOL:  _bvalue = other._bvalue; break;
        case INT:   _ivalue = other._ivalue; break;
        case FLOAT: _fvalue = other._fvalue; break;
        default: break;
        }
        return *this;
    }*/

    Parameter &operator = (const Parameter &other)
    {
        clear();
        _name = other._name;
        _type = other._type;
        _listener = other._listener;
        switch (_type)
        {
        case NULL_TYPE: break;
        case DIR: break;    ////
        case BOOL_TYPE:  _value.b = other._value.b; break;
        case INT_TYPE:   _value.i = other._value.i; break;
        case FLOAT_TYPE: _value.f = other._value.f; break;
        default: break;
        }
        return *this;
    }



protected:
    String _name;
    Type _type;
    /*union
    {
        bool _bvalue;
        int _ivalue;
        float _fvalue;
        double _dvalue;
    };*/
    ParameterListener *_listener;
    union _Value
    {
        bool b;
        int i;
        float f;
        double d;
        Vector2i *v2i;
        Vector3i *v3i;
        Vector4i *v4i;
        Vector2f *v2f;
        Vector3f *v3f;
        Vector4f *v4f;
        String *str;
        Vector<Parameter *> *arr;
        Hash<String, Parameter *> *obj;
    } _value;
};

////
class ParameterDirListener
{
public:
    virtual void parameterAdded(ParameterDir *dir, Parameter *param) = 0;
};

class ParameterDir : protected Parameter
{
public:
    ParameterDir() : _listener(nullptr) { Parameter::_type = DIR; }
    virtual ~ParameterDir()
    {
        for (Hash<String, Parameter *>::iterator it = _params.begin(); it != _params.end(); ++it)
        {
            delete *it;
        }
    }

    ParameterDir(const ParameterDir &other) { *this = other; }
    void setName(const String &name) { Parameter::setName(name); }
    const String &name() const { return Parameter::name(); }
    Type type() const { return Parameter::type(); }
    void setListener(ParameterDirListener *listener) { _listener = listener; }
    /*Parameter *addParameter(const String &name)
    {
        if (_params.contains(name)) return nullptr;
        Parameter *param = new Parameter(name);
        _params[param->name()] = param;
        if (_listener != nullptr) _listener->parameterAdded(this, param);
        return param;
    }*/
    void addParameter(const String &name)
    {
        if (_params.contains(name)) return;
        Parameter *param = new Parameter(name);
        //_params[name].setName(name);
        param->setName(name);
        _params[name] = param;
        if (_listener != nullptr) _listener->parameterAdded(this, _params[name]);
        //return param;
    }

    ParameterDir &operator = (const ParameterDir &other)
    {
        Parameter::operator = (other);
        _params = other._params;
        _listener = other._listener;
    }

    Parameter &operator [] (const String &name)
    {
        if (!_params.contains(name)) addParameter(name);
        return *_params[name];
    }

    //const Parameter &operator [] (const String &name) const { return _items[name]; }

protected:
    Hash<String, Parameter *> _params;
    //Hash<String, Parameter> _params;
    ParameterDirListener *_listener;
};

class ParameterSet : public ParameterListener, public ParameterDirListener
{
public:
    ParameterSet() { _root.setName("root"), _root.setListener(this); }
    ~ParameterSet()
    {
        //for (int i = 0; i < _params.size(); i++)
        //    delete _params[i];
    }

    Parameter &operator [] (const String &name) { return _root[name]; }

public:
    void valueChanged(Parameter *param)
    {
        //qDebug("ParameterSet::valueChanged(%s)", param->name().c_str());
        valueChanged(param->name());
    }

    void parameterAdded(ParameterDir *dir, Parameter *param)
    {
        //qDebug("ParameterSet::parameterAdded(%s, %s)", dir->name().c_str(), param->name().c_str());
        param->setListener(this);
        _params.append(param);
    }

protected:
    virtual void valueChanged(const String &) {}

protected:
    ParameterDir _root;
    Vector<Parameter *> _params;
};

#endif // PARAMETERSET_H
