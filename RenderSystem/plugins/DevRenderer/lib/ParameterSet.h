#ifndef PARAMETERSET_H
#define PARAMETERSET_H

#include <string>
#include <sstream>

////
#include <QtCore>

#include "MSVectors.h"
#include "Containers.h"

#define nullptr 0   // C++11

typedef std::string String;

class Parameter;
//class ParameterDir;

class ParameterListener
{
public:
    virtual void valueChanged(Parameter *param) = 0;
    virtual void parameterAdded(Parameter *param) = 0;
};

class Parameter
{
public:
    enum Type
    {
        NULL_TYPE = 0,
        OBJECT_TYPE,
        ARRAY_TYPE,
        STRING_TYPE,
        BOOL_TYPE,
        INT_TYPE,
        FLOAT_TYPE,
        DOUBLE_TYPE,
        VECTOR2I_TYPE, VECTOR3I_TYPE, VECTOR4I_TYPE,
        VECTOR2F_TYPE, VECTOR3F_TYPE, VECTOR4F_TYPE,
        VECTOR2D_TYPE, VECTOR3D_TYPE, VECTOR4D_TYPE
    };

    typedef std::map<String, Parameter *>::iterator ObjectIterator;

public:
    Parameter() : _type(NULL_TYPE), _listener(nullptr), _parent(nullptr) {}
    Parameter(bool  value) : _type(BOOL_TYPE),  _listener(nullptr), _parent(nullptr) { _value.b = value; }
    Parameter(int   value) : _type(INT_TYPE),   _listener(nullptr), _parent(nullptr) { _value.i = value; }
    Parameter(float value) : _type(FLOAT_TYPE), _listener(nullptr), _parent(nullptr) { _value.f = value; }
    //explicit Parameter(const String &name) : _name(name), _type(NULL_TYPE), _listener(nullptr), _parent(nullptr) {}
    Parameter(const String &name, ParameterListener *listener = nullptr, Parameter *parent = nullptr) : _name(name), _type(NULL_TYPE), _listener(listener), _parent(parent) {}
    //Parameter(const Parameter &other) { *this = other; }
    virtual ~Parameter() { clear(); }

    Type type() const { return _type; }
    bool isNull()   const { return (_type == NULL_TYPE); }
    bool isObject() const { return (_type == OBJECT_TYPE); }
    bool isArray()  const { return (_type == ARRAY_TYPE); }
    bool isString() const { return (_type == STRING_TYPE); }

    void setName(const String &name) { _name = name; }
    void setListener(ParameterListener *listener) { _listener = listener; }
    void setParent(Parameter *parent) { _parent = parent; }
    const String &name() const { return _name; }

    //template <typename T> void setValue(const T &) {}   // should not use this
    //template <> void setValue(const bool  &value) { if (_type != Bool  || _bvalue != value) { _type = Bool;  _bvalue = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }
    //template <> void setValue(const int   &value) { if (_type != Int   || _ivalue != value) { _type = Int;   _ivalue = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }
    //template <> void setValue(const float &value) { if (_type != Float || _fvalue != value) { _type = Float; _fvalue = value; if (_listener != nullptr) { _listener->valueChanged(this); } } }

    void setValue(const bool  &value) { if (_type != BOOL_TYPE  || _value.b != value) { _type = BOOL_TYPE;  _value.b = value; _emitValueChanged(); } }
    void setValue(const int   &value) { if (_type != INT_TYPE   || _value.i != value) { _type = INT_TYPE;   _value.i = value; _emitValueChanged(); } }
    void setValue(const float &value) { if (_type != FLOAT_TYPE || _value.f != value) { _type = FLOAT_TYPE; _value.f = value; _emitValueChanged(); } }

    void setValue(const String &value)
    {
        qDebug("setValue(%s)", value.c_str());
        if (isNull())
        {
            qDebug("new String");
            _type = STRING_TYPE;
            _value.str = new String(value);
            _emitValueChanged();
            return;
        }
        _checkType(STRING_TYPE);
        if (*_value.str != value)
        {
            *_value.str = value;
            _emitValueChanged();
        }
    }
    void setValue(const char *value) { setValue(String(value)); }

    void setValue(const Vector2i &value)
    {
        if (isNull()) { qDebug("new Vector2i"); _type = VECTOR2I_TYPE; _value.v2i = new Vector2i(value); _emitValueChanged(); }
        else { _checkType(VECTOR2I_TYPE); if (*_value.v2i != value) { *_value.v2i = value; _emitValueChanged(); } }
    }

    void setValue(const Vector3f &value)
    {
        if (isNull()) { qDebug("new Vector3f"); _type = VECTOR3F_TYPE; _value.v3f = new Vector3f(value); _emitValueChanged(); }
        else { _checkType(VECTOR3F_TYPE); if (*_value.v3f != value) { *_value.v3f = value; _emitValueChanged(); } }
    }

    const bool     &toBool()     const { return _value.b; }
    const int      &toInt()      const { return _value.i; }
    const float    &toFloat()    const { return _value.f; }
    const double   &toDouble()   const { return _value.d; }
    const String   &toString()   const { return *_value.str; }
    const char     *toCString()  const { return _value.str->c_str(); }
    const Vector2i &toVector2i() const { return *_value.v2i; }
    const Vector3f &toVector3f() const { return *_value.v3f; }

    //template <typename T> const T &value() const { /* Error */ return *reinterpret_cast<const T *>(&_ivalue); }
    //template <> const bool  &value<bool>()  const { return _bvalue; }
    //template <> const int   &value<int>()   const { return _ivalue; }
    //template <> const float &value<float>() const { return _fvalue; }

    void clear()
    {
        switch (_type)
        {
        case OBJECT_TYPE:
            for (ObjectIterator it = _value.obj->begin(); it != _value.obj->end(); it++)
                delete it->second;
            delete _value.obj;
            break;
        case ARRAY_TYPE:
            for (size_t i = 0; i < _value.arr->size(); i++)
                delete (*_value.arr)[i];
            delete _value.arr;
            break;
        case STRING_TYPE:   delete _value.str; break;
        case VECTOR2I_TYPE: delete _value.v2i; break;
        case VECTOR3I_TYPE: delete _value.v3i; break;
        case VECTOR4I_TYPE: delete _value.v4i; break;
        case VECTOR2F_TYPE: delete _value.v2f; break;
        case VECTOR3F_TYPE: delete _value.v3f; break;
        case VECTOR4F_TYPE: delete _value.v4f; break;
        }
        _type = NULL_TYPE;
    }

    /*Parameter &operator = (const Parameter &other)
    {
        clear();
        _name = other._name;
        _type = other._type;
        _listener = other._listener;
        _parent = other._parent;
        switch (_type)
        {
        case NULL_TYPE: break;
        case BOOL_TYPE:  _value.b = other._value.b; break;
        case INT_TYPE:   _value.i = other._value.i; break;
        case FLOAT_TYPE: _value.f = other._value.f; break;
        case OBJECT_TYPE:
            _value.obj = new std::map<String, Parameter *>();
            for (ObjectIterator it = other._value.obj->begin(); it != other._value.obj->end(); it++)
                (*_value.obj)[it->first] = new Parameter(*it->second);
            break;
        default: break;
        }
        return *this;
    }*/

    bool contains(const String &key) const
    {
        return (_value.obj->find(key) != _value.obj->end());
    }

    int size() const
    {
        _checkType(ARRAY_TYPE);
        return (int)_value.arr->size();
    }

    Parameter &operator [] (int index)
    {
        if (isNull())
        {
            qDebug("new Vector");
            _type = ARRAY_TYPE;
            _value.arr = new Vector<Parameter *>();
        }
        //assert(index >= 0)
        if (index >= _value.arr->size())
        {
            for (size_t i = _value.arr->size(); i <= (size_t)index; i++)
            {
                std::stringstream name;
                name << _name << '[' << _value.arr->size() << ']';
                _value.arr->append(new Parameter(name.str(), _listener, this));
                _emitParameterAdded(_value.arr->last());
            }
        }
        return *(*_value.arr)[index];
    }

    Parameter &operator [] (const String &key)
    {
        //qDebug("Parameter::operator[%s]", key.c_str());
        if (isNull())
        {
            qDebug("new std::map");
            _type = OBJECT_TYPE;
            _value.obj = new std::map<String, Parameter *>();
        }
        if (!contains(key))
        {
            String name = (_parent == nullptr) ? key : _name + '.' + key;
            (*_value.obj)[key] = new Parameter(name, _listener, this);
            _emitParameterAdded((*_value.obj)[key]);
        }
        return *(*_value.obj)[key];
    }

    /*const Parameter &operator [] (const String &key) const
    {
        return *(*_value.obj)[key];
    }*/

protected:
    void _checkType(Type type) const
    {
        if (_type != type)
            qDebug("Incorrect type!");
    }

    void _emitValueChanged()
    {
        if (_listener != nullptr)
            _listener->valueChanged(this);
        if (_parent != nullptr)
            _parent->_emitValueChanged();
    }

    void _emitParameterAdded(Parameter *param)
    {
        if (_listener != nullptr)
            _listener->parameterAdded(param);
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
    Parameter *_parent;
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
        std::map<String, Parameter *> *obj;
    } _value;
};

////
/*class ParameterDirListener
{
public:
    virtual void parameterAdded(ParameterDir *dir, Parameter *param) = 0;
};*/

/*class ParameterDir : protected Parameter
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
    //Parameter *addParameter(const String &name)
    //{
    //    if (_params.contains(name)) return nullptr;
    //    Parameter *param = new Parameter(name);
    //    _params[param->name()] = param;
    //    if (_listener != nullptr) _listener->parameterAdded(this, param);
    //    return param;
    //}
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
        qDebug("ParameterDir::operator[%s]", name.c_str());
        if (!_params.contains(name)) addParameter(name);
        return *_params[name];
    }

    //const Parameter &operator [] (const String &name) const { return _items[name]; }

protected:
    Hash<String, Parameter *> _params;
    //Hash<String, Parameter> _params;
    ParameterDirListener *_listener;
};*/

class ParameterSet : public ParameterListener //, public ParameterDirListener
{
public:
    ParameterSet() { _root.setName("/"), _root.setListener(this); parameterAdded(&_root); }
    virtual ~ParameterSet()
    {
        //for (int i = 0; i < _params.size(); i++)
        //    delete _params[i];
    }

    bool contains(const String &name) const         // full name
    {
        return (_params.find(name) != _params.end());
    }

    Parameter *getParameter(const String &name)     // full name
    {
        if (!contains(name))
            return nullptr;
        else
            return _params[name];
    }

    //Parameter &operator [] (const String &name) { qDebug("root[%s]", name.c_str()); return *_params[name]; }
    Parameter &operator [] (const String &key) { return _root[key]; }

protected:     //!! protected
    void valueChanged(Parameter *param)
    {
        //qDebug("ParameterSet::valueChanged(%s)", param->name().c_str());
        valueChanged(param->name());
    }

    /*void parameterAdded(ParameterDir *dir, Parameter *param)
    {
        //qDebug("ParameterSet::parameterAdded(%s, %s)", dir->name().c_str(), param->name().c_str());
        param->setListener(this);
        _params.append(param);
    }*/

    void parameterAdded(Parameter *param)
    {
        //qDebug("parameterAdded(%s)", param->name().c_str());
        _params[param->name()] = param;
    }

protected:
    virtual void valueChanged(const String &) {}

protected:
    //ParameterDir _root;
    Parameter _root;
    //Vector<Parameter *> _params;
    std::map<String, Parameter *> _params;
};

#endif // PARAMETERSET_H
