//
// JsonParser.h
//
// Copyright (C) 2012 Min Shih
//

#ifndef JSONPARSER_H
#define JSONPARSER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

namespace Json
{

enum ValueType
{
    VoidType = -1,
    NullType = 0,
    BoolType,
    NumberType,
    StringType,
    ArrayType,
    ObjectType
};

class Value;
class ValueIterator;
class ValueConstIterator;
class Parser;

class Exception : public std::exception
{
public:
    Exception() {}
    Exception(const std::string &message) : _message(message) {}
    ~Exception() throw() {}
    virtual const char *what() const throw() { return _message.c_str(); }

protected:
    std::string _message;
};

class ValueTypeException : public Exception
{
public:
    ValueTypeException(ValueType type, ValueType expectedType, ValueType expectedType2 = VoidType);
    ValueType type() const { return _type; }
    ValueType expectedType() const { return _expectedType; }
    ValueType expectedType2() const { return _expectedType2; }

private:
    ValueType _type;
    ValueType _expectedType;
    ValueType _expectedType2;
};

class IteratorException : public Exception
{
public:
    IteratorException(const std::string &message) : Exception(message) {}
};

class ParseException : public Exception
{
public:
    ParseException(const std::string &message, int line) : Exception(message), _line(line) {}
    int line() const { return _line; }

private:
    int _line;
};

class Value
{
    friend class Parser;

public:
    typedef ValueIterator      iterator;
    typedef ValueIterator      Iterator;
    typedef ValueConstIterator const_iterator;
    typedef ValueConstIterator ConstIterator;

public:
    Value();
    Value(ValueType type);
    Value(bool value);
    Value(int value);
    Value(double value);
    Value(const char *value);
    Value(const std::string &value);
    Value(const Value &other);
    ~Value();

    ValueType type() const { return _type; }
    bool isNull()    const { return (_type == NullType); }
    bool isBool()    const { return (_type == BoolType); }
    bool isInt()     const { return (_type == NumberType && floor(_value.num) == _value.num); }
    bool isNumber()  const { return (_type == NumberType); }
    bool isString()  const { return (_type == StringType); }
    bool isArray()   const { return (_type == ArrayType); }
    bool isObject()  const { return (_type == ObjectType); }

    bool               toBool()    const { _assertType(BoolType);   return _value.bln; }
    int                toInt()     const { _assertType(NumberType); return (int)_value.num; }
    float              toFloat()   const { _assertType(NumberType); return (float)_value.num; }
    double             toDouble()  const { _assertType(NumberType); return _value.num; }
    const char        *toCString() const { _assertType(StringType); return _value.str->c_str(); }
    const std::string &toString()  const { _assertType(StringType); return *_value.str; }

    int  size() const;                                          // 0 if not array nor object
    void resize(int newSize);                                   // array or null -> array
    void append(const Value &value);                            // array or null -> array
    bool contains(const std::string &key) const;                // object or null

    iterator       begin();
    const_iterator begin() const;
    iterator       end();
    const_iterator end() const;
    const_iterator constBegin() const;
    const_iterator constEnd() const;

    Value get(int index, const Value &defaultValue) const;
    Value get(const std::string &key, const Value &defaultValue) const;

    void clear();                                               // clear everything and set to null

    bool   operator != (const Value &other) const { return !(*this == other); }
    Value &operator =  (const Value &other);
    bool   operator == (const Value &other) const;

    Value       &operator [] (int index);                       // array
    const Value &operator [] (int index) const;                 // array
    Value       &operator [] (const std::string &key);          // object
    const Value &operator [] (const std::string &key) const;    // object

    static std::string typeName(ValueType type);

public:
    static const Value Null;

private:
    void _assertType(ValueType expectedType) const;
    void _assertType(ValueType expectedType, ValueType expectedType2) const;

private:
    ValueType _type;
    union _Value
    {
        bool bln;
        double num;
        std::string *str;
        std::vector<Value *> *arr;
        std::map<std::string, Value *> *obj;
    } _value;
};

std::istream &operator >> (std::istream &is, Value &value);
std::ostream &operator << (std::ostream &os, const Value &value);

class ValueIterator
{
    friend class Value;
    friend class ValueConstIterator;

public:
    ValueIterator()                           : _type(NullType) {}
    ValueIterator(const ValueIterator &other) { *this = other; }

    ValueType          type()   const { return _type; }
    bool               isNull() const { return (_type == NullType); }
    const std::string &key()    const;
    Value             &value()  const { return **this; }

    bool           operator != (const ValueIterator &other)      const { return !(*this == other); }
    bool           operator != (const ValueConstIterator &other) const { return !(*this == other); }
    Value         &operator *  ()                                const;
    ValueIterator &operator ++ ();
    ValueIterator  operator ++ (int)                                   { ValueIterator it(*this); ++(*this); return it; }
    ValueIterator &operator -- ();
    ValueIterator  operator -- (int)                                   { ValueIterator it(*this); --(*this); return it; }
    Value         *operator -> ()                                const { return &(**this); }
    ValueIterator &operator =  (const ValueIterator &other);
    bool           operator == (const ValueIterator &other)      const;
    bool           operator == (const ValueConstIterator &other) const;

private:
    explicit ValueIterator(const std::vector<Value *>::iterator it)           : _type(ArrayType),  _arrit(it) {}
    explicit ValueIterator(const std::map<std::string, Value *>::iterator it) : _type(ObjectType), _objit(it) {}

private:
    ValueType _type;
    std::vector<Value *>::iterator _arrit;
    std::map<std::string, Value *>::iterator _objit;
};

class ValueConstIterator
{
    friend class Value;

public:
    ValueConstIterator()                                : _type(NullType) {}
    ValueConstIterator(const ValueConstIterator &other) { *this = other; }
    ValueConstIterator(const ValueIterator &other);

    ValueType          type()   const { return _type; }
    bool               isNull() const { return (_type == NullType); }
    const std::string &key()    const;
    const Value       &value()  const { return **this; }

    bool                operator != (const ValueConstIterator &other) const { return !(*this == other); }
    const Value        &operator *  ()                                const;
    ValueConstIterator &operator ++ ();
    ValueConstIterator  operator ++ (int)                                   { ValueConstIterator it(*this); ++(*this); return it; }
    ValueConstIterator &operator -- ();
    ValueConstIterator  operator -- (int)                                   { ValueConstIterator it(*this); --(*this); return it; }
    const Value        *operator -> ()                                const { return &(**this); }
    ValueConstIterator &operator =  (const ValueConstIterator &other);
    bool                operator == (const ValueConstIterator &other) const;

private:
    explicit ValueConstIterator(const std::vector<Value *>::const_iterator it)           : _type(ArrayType),  _arrit(it) {}
    explicit ValueConstIterator(const std::map<std::string, Value *>::const_iterator it) : _type(ObjectType), _objit(it) {}

private:
    ValueType _type;
    std::vector<Value *>::const_iterator _arrit;
    std::map<std::string, Value *>::const_iterator _objit;
};

class Parser
{
public:
    Parser();
    ~Parser();

    void setIndent(int indent) { _indent = indent; }

    void parse(std::istream &is, Value &root);
    void parse(const std::string &document, Value &root);
    bool parseFile(const std::string &fileName, Value &root);
    void write(std::ostream &os, const Value &root);
    void write(std::string &document, const Value &root);
    bool writeFile(const std::string &fileName, Value &root);

private:
    char _getChar(std::istream &is);
    double _getNumber(std::istream &is);
    std::string _getString(std::istream &is);
    bool _skipComment(std::istream &is);
    char _getNonSpaceChar(std::istream &is);
    bool _matchToken(std::istream &is, char token);
    bool _match(std::istream &is, const char *pattern, char *gots);
    void _readArray(std::istream &is, Value &value);
    void _readObject(std::istream &is, Value &value);
    void _readValue(std::istream &is, Value &value);

    void _writeArray(std::ostream &os, const Value &value);
    void _writeObject(std::ostream &os, const Value &value);
    void _writeValue(std::ostream &os, const Value &value);

protected:
    int _line;
    int _indent;
    int _level;
};

////////////////////////////////////////////////////////////////////////////////
//  inline functions                                                          //
////////////////////////////////////////////////////////////////////////////////

inline ValueTypeException::ValueTypeException(ValueType type, ValueType expectedType, ValueType expectedType2)
    : _type(type), _expectedType(expectedType), _expectedType2(expectedType2)
{
    _message = "Expected \"" + Value::typeName(_expectedType) + "\"";
    if (_expectedType2 != VoidType)
        _message += " or \"" + Value::typeName(_expectedType2) + "\"";
    _message += " type; got \"" + Value::typeName(_type) + "\" type";
}

inline int Value::size() const
{
    switch (_type)
    {
    case NullType:
    case BoolType:
    case NumberType:
    case StringType: return 0;
    case ArrayType:  return (int)_value.arr->size();
    case ObjectType: return (int)_value.obj->size();
    }
    return 0;   // unreachable
}

inline Value::iterator Value::begin()
{
    if      (_type == ArrayType)  return Value::iterator(_value.arr->begin());
    else if (_type == ObjectType) return Value::iterator(_value.obj->begin());
    else                          return Value::iterator();
}

inline Value::const_iterator Value::begin() const
{
    if      (_type == ArrayType)  return Value::const_iterator(_value.arr->begin());
    else if (_type == ObjectType) return Value::const_iterator(_value.obj->begin());
    else                          return Value::const_iterator();
}

inline Value::iterator Value::end()
{
    if      (_type == ArrayType)  return Value::iterator(_value.arr->end());
    else if (_type == ObjectType) return Value::iterator(_value.obj->end());
    else                          return Value::iterator();
}

inline Value::const_iterator Value::end() const
{
    if      (_type == ArrayType)  return Value::const_iterator(_value.arr->end());
    else if (_type == ObjectType) return Value::const_iterator(_value.obj->end());
    else                          return Value::const_iterator();
}

inline Value::const_iterator Value::constBegin() const
{
    return begin();
}

inline Value::const_iterator Value::constEnd() const
{
    return end();
}

inline Value Value::get(int index, const Value &defaultValue) const
{
    _assertType(ArrayType, NullType);
    if (index < 0)
        throw Exception("Negative array index");
    if (isNull() || index >= (int)_value.arr->size())
        return defaultValue;
    return *(*_value.arr)[index];
}

inline Value Value::get(const std::string &key, const Value &defaultValue) const
{
    _assertType(ObjectType, NullType);
    if (isNull() || !contains(key))
        return defaultValue;
    return *(*_value.obj)[key];

}

inline Value &Value::operator [] (int index)
{
    _assertType(ArrayType, NullType);
    if (index < 0)
        throw Exception("Negative array index");
    if (isNull())
        *this = Value(ArrayType);
    if (index >= (int)_value.arr->size())
        resize(index + 1);
    return *(*_value.arr)[index];
}

inline const Value &Value::operator [] (int index) const
{
    _assertType(ArrayType, NullType);
    if (index < 0)
        throw Exception("Negative array index");
    if (isNull() || index >= (int)_value.arr->size())
        return Null;
    return *(*_value.arr)[index];
}

inline Value &Value::operator [] (const std::string &key)
{
    _assertType(ObjectType, NullType);
    if (isNull())
        *this = Value(ObjectType);
    if (!contains(key))
        (*_value.obj)[key] = new Value();
    return *(*_value.obj)[key];
}

inline const Value &Value::operator [] (const std::string &key) const
{
    _assertType(ObjectType, NullType);
    if (isNull() || !contains(key))
        return Null;
    return *(*_value.obj)[key];
}

inline std::string Value::typeName(ValueType type)
{
    switch (type)
    {
    case NullType:   return "Null";
    case BoolType:   return "Bool";
    case NumberType: return "Number";
    case StringType: return "String";
    case ArrayType:  return "Array";
    case ObjectType: return "Object";
    default:         return "";
    }
}

inline void Value::_assertType(ValueType expectedType) const
{
    if (_type != expectedType)
        throw ValueTypeException(_type, expectedType);
}

inline void Value::_assertType(ValueType expectedType, ValueType expectedType2) const
{
    if (_type != expectedType && _type != expectedType2)
        throw ValueTypeException(_type, expectedType, expectedType2);
}

inline const std::string &ValueIterator::key() const
{
    if (_type != ObjectType) throw IteratorException("Getting key from non-object value iterator");
    return _objit->first;
}

inline Value &ValueIterator::operator * () const
{
    if      (_type == ArrayType)  return **_arrit;
    else if (_type == ObjectType) return *_objit->second;
    throw IteratorException("Dereferencing uninitialized iterator");
}

inline ValueIterator &ValueIterator::operator ++ ()
{
    if      (_type == ArrayType)  ++_arrit;
    else if (_type == ObjectType) ++_objit;
    else throw IteratorException("Incrementing uninitialized iterator");
    return *this;
}

inline ValueIterator &ValueIterator::operator -- ()
{
    if      (_type == ArrayType)  --_arrit;
    else if (_type == ObjectType) --_objit;
    else throw IteratorException("Decrementing uninitialized iterator");
    return *this;
}

inline ValueIterator &ValueIterator::operator = (const ValueIterator &other)
{
    _type = other._type;
    if      (_type == ArrayType)  _arrit = other._arrit;
    else if (_type == ObjectType) _objit = other._objit;
    return *this;
}

inline bool ValueIterator::operator == (const ValueIterator &other) const
{
    if      (_type != other._type) return false;
    else if (_type == ArrayType)   return (_arrit == other._arrit);
    else if (_type == ObjectType)  return (_objit == other._objit);
    else                           return true;
}

inline bool ValueIterator::operator == (const ValueConstIterator &other) const
{
    return (ValueConstIterator(*this) == other);
}

inline ValueConstIterator::ValueConstIterator(const ValueIterator &other)
{
    _type = other._type;
    if      (_type == ArrayType)  _arrit = other._arrit;
    else if (_type == ObjectType) _objit = other._objit;
}

inline const std::string &ValueConstIterator::key() const
{
    if (_type != ObjectType) throw IteratorException("Getting key from non-object value iterator");
    return _objit->first;
}

inline const Value &ValueConstIterator::operator * () const
{
    if      (_type == ArrayType)  return **_arrit;
    else if (_type == ObjectType) return *_objit->second;
    throw IteratorException("Dereferencing uninitialized iterator");
}

inline ValueConstIterator &ValueConstIterator::operator ++ ()
{
    if      (_type == ArrayType)  ++_arrit;
    else if (_type == ObjectType) ++_objit;
    else throw IteratorException("Incrementing uninitialized iterator");
    return *this;
}

inline ValueConstIterator &ValueConstIterator::operator -- ()
{
    if      (_type == ArrayType)  --_arrit;
    else if (_type == ObjectType) --_objit;
    else throw IteratorException("Decrementing uninitialized iterator");
    return *this;
}

inline ValueConstIterator &ValueConstIterator::operator = (const ValueConstIterator &other)
{
    _type = other._type;
    if      (_type == ArrayType)  _arrit = other._arrit;
    else if (_type == ObjectType) _objit = other._objit;
    return *this;
}

inline bool ValueConstIterator::operator == (const ValueConstIterator &other) const
{
    if      (_type != other._type) return false;
    else if (_type == ArrayType)   return (_arrit == other._arrit);
    else if (_type == ObjectType)  return (_objit == other._objit);
    else                           return true;
}

} // namespace Json

#endif // JSONPARSER_H
