//
// JsonParser.cpp
//
// Copyright (C) 2012 Min Shih
//

#include <cctype>
#include <sstream>

#include "JsonParser.h"

namespace Json
{

const Value Value::Null = Value();

Value::Value()
    : _type(NullType)
{
}

Value::Value(ValueType type)
    : _type(type)
{
    switch (_type)
    {
    case NullType:   break;
    case BoolType:   _value.bln = false; break;
    case NumberType: _value.num = 0.0; break;
    case StringType: _value.str = new std::string(""); break;
    case ArrayType:  _value.arr = new std::vector<Value *>(); break;
    case ObjectType: _value.obj = new std::map<std::string, Value *>(); break;
    default:         _type = NullType; break;
    }
}

Value::Value(bool value)
    : _type(BoolType)
{
    _value.bln = value;
}

Value::Value(int value)
    : _type(NumberType)
{
    _value.num = (double)value;
}

Value::Value(double value)
    : _type(NumberType)
{
    _value.num = value;
}

Value::Value(const char *value)
    : _type(StringType)
{
    _value.str = new std::string(value);
}

Value::Value(const std::string &value)
    : _type(StringType)
{
    _value.str = new std::string(value);
}

Value::Value(const Value &other)
    : _type(NullType)
{
    *this = other;
}

Value::~Value()
{
    clear();
}

void Value::resize(int newSize)
{
    _assertType(ArrayType, NullType);
    if (isNull())
        *this = Value(ArrayType);
    int oldSize = this->size();
    if (newSize < oldSize)
    {
        for (int i = newSize; i < oldSize; i++)
            delete (*_value.arr)[i];
    }
    _value.arr->resize(newSize, NULL);
    if (newSize > oldSize)
    {
        for (int i = oldSize; i < newSize; i++)
            (*_value.arr)[i] = new Value();
    }
}

void Value::append(const Value &value)
{
    _assertType(ArrayType, NullType);
    if (isNull())
        *this = Value(ArrayType);
    _value.arr->push_back(new Value(value));
}

bool Value::contains(const std::string &key) const
{
    _assertType(ObjectType, NullType);
    if (isNull())
        return false;
    return (_value.obj->find(key) != _value.obj->end());
}

void Value::clear()
{
    switch (_type)
    {
    case StringType:
        delete _value.str;
        break;
    case ArrayType:
        for (size_t i = 0; i < _value.arr->size(); i++)
            delete (*_value.arr)[i];
        delete _value.arr;
        break;
    case ObjectType:
        for (iterator it = begin(); it != end(); ++it)
            delete &it.value();
        delete _value.obj;
        break;
    }
    _type = NullType;
}

Value &Value::operator = (const Value &other)
{
    clear();
    _type = other._type;
    switch (_type)
    {
    case BoolType:   _value.bln = other._value.bln; break;
    case NumberType: _value.num = other._value.num; break;
    case StringType: _value.str = new std::string(*other._value.str); break;
    case ArrayType:
        _value.arr = new std::vector<Value *>();
        for (int i = 0; i < other.size(); i++)
            append(other[i]);
        break;
    case ObjectType:
        _value.obj = new std::map<std::string, Value *>();
        for (const_iterator it = other.constBegin(); it != other.constEnd(); ++it)
            (*this)[it.key()] = it.value();
        break;
    }
    return *this;
}

bool Value::operator == (const Value &other) const
{
    if (_type != other._type)
        return false;
    switch (_type)
    {
    case NullType:   return true;
    case BoolType:   return (_value.bln == other._value.bln);
    case NumberType: return (_value.num == other._value.num);
    case StringType: return (*_value.str == *other._value.str);
    case ArrayType:
        if (size() != other.size())
            return false;
        for (int i = 0; i < size(); i++)
            if ((*this)[i] != other[i])
                return false;
        return true;
    case ObjectType:
        if (size() != other.size())
            return false;
        for (const_iterator it = constBegin(); it != constEnd(); ++it)
        {
            if (!other.contains(it.key()))
                return false;
            if (it.value() != other[it.key()])
                return false;
        }
        return true;
    }
    return false;   // unreachable
}

std::istream &operator >> (std::istream &is, Json::Value &value)
{
    Parser().parse(is, value);
    return is;
}

std::ostream &operator << (std::ostream &os, const Value &value)
{
    Parser().write(os, value);
    return os;
}

Parser::Parser()
    : _indent(2)
{
}

Parser::~Parser()
{
}

void Parser::parse(std::istream &is, Value &root)
{
    _line = 1;
    _readValue(is, root);
}

void Parser::parse(const std::string &document, Value &root)
{
    std::stringstream ss(document);
    parse(ss, root);
}

bool Parser::parseFile(const std::string &fileName, Value &root)
{
    std::ifstream ifs;
    ifs.open(fileName.c_str(), std::ios::in);
    if (ifs.fail()) return false;
    parse(ifs, root);
    ifs.close();
    return true;
}

void Parser::write(std::ostream &os, const Value &root)
{
    _level = 0;
    _writeValue(os, root);
}

void Parser::write(std::string &document, const Value &root)
{
    std::stringstream ss;
    write(ss, root);
    document = ss.str();
}

bool Parser::writeFile(const std::string &fileName, Value &root)
{
    std::ofstream ofs;
    ofs.open(fileName.c_str(), std::ios::out);
    if (ofs.fail()) return false;
    write(ofs, root);
    ofs.close();
    return true;
}

inline char Parser::_getChar(std::istream &is)
{
    char c;
    is.get(c);
    if (is.fail())
    {
        if (is.eof())
            throw ParseException("Unexpected end-of-file found", _line);
        else
            throw ParseException("Fetch error", _line);
    }
    return c;
}

double Parser::_getNumber(std::istream &is)
{
    double num;
    is >> num;
    if (is.fail())
    {
        if (is.eof())
            throw ParseException("Unexpected end-of-file found", _line);
        else
            throw ParseException("Fetch error", _line);
    }
    return num;
}

std::string Parser::_getString(std::istream &is)
{
    std::string str;
    char c = _getChar(is);
    while (c != '"')
    {
        if (c == '\\')
        {
            c = _getChar(is);
            switch (c)
            {
            case '"':  str += '"';  break;
            case '\\': str += '\\'; break;
            case '/':  str += '/';  break;
            case 'b':  str += '\b'; break;
            case 'f':  str += '\f'; break;
            case 'n':  str += '\n'; break;
            case 'r':  str += '\r'; break;
            case 't':  str += '\t'; break;
            default: throw ParseException(std::string("Bad escape sequence in string: '\\") + c + "'", _line);
            }
        }
        else
        {
            str += c;
        }
        c = _getChar(is);
    }
    return str;
}

bool Parser::_skipComment(std::istream &is)
{
    char c = _getChar(is);  // character next to '/'
    if (c == '*')           // C-style comment
    {
        c = _getChar(is);
        char cc = '\0';     // last character
        while (!(c == '/' && cc == '*'))    // not '*/'
        {
            if (c == '\r' || (c == '\n' && cc != '\r'))     // '\r', '\n', '\r\n'
                _line++;
            cc = c;
            c = _getChar(is);
        }
    }
    else if (c == '/')      // Cpp-style comment
    {
        do
        {
            c = _getChar(is);
        }
        while (c != '\r' && c != '\n');
        is.putback(c);      // put '\r' or '\n' back
    }
    else                    // not valid comment
    {
        is.putback(c);
        return false;
    }
    return true;
}

char Parser::_getNonSpaceChar(std::istream &is)
{
    char c = _getChar(is), cc = '\0';
    while (isspace(c) || c == '/')      // skip white spaces and comments
    {
        if (c == '\r' || (c == '\n' && cc != '\r'))     // '\r', '\n', '\r\n'
            _line++;
        if (c == '/')                   // comment
            if (!_skipComment(is))      // not valid comment
                break;
        cc = c;
        c = _getChar(is);
    }
    return c;
}

bool Parser::_matchToken(std::istream &is, char token)
{
    char c = _getNonSpaceChar(is);
    if (c != token)
    {
        is.putback(c);
        return false;
    }
    return true;
}

bool Parser::_match(std::istream &is, const char *pattern, char *gots)
{
    for (int i = 0; pattern[i] != '\0'; i++)
    {
        gots[i] = _getChar(is);
        if (gots[i] != pattern[i])
        {
            for (int j = i; j >= 0; j--)
                is.putback(gots[j]);
            gots[i + 1] = '\0';
            return false;
        }
    }
    return true;
}

void Parser::_readArray(std::istream &is, Value &value)
{
    if (_matchToken(is, ']'))
        return;
    while (true)
    {
        Value *item = new Value();
        value._value.arr->push_back(item);
        _readValue(is, *item);
        char c = _getNonSpaceChar(is);
        if (c == ']')
            break;
        if (c != ',')
            throw ParseException("Missing ',' in array declaration", _line);
    }
}

void Parser::_readObject(std::istream &is, Value &value)
{
    if (_matchToken(is, '}'))
        return;
    while (true)
    {
        if (!_matchToken(is, '"'))
            throw ParseException("Missing object member name", _line);
        std::string key = _getString(is);
        if (!_matchToken(is, ':'))
            throw ParseException("Missing ':' after object member name", _line);
        Value *val = new Value();
        (*value._value.obj)[key] = val;
        _readValue(is, *val);
        char c = _getNonSpaceChar(is);
        if (c == '}')
            break;
        if (c != ',')
            throw ParseException("Missing ',' in object declaration", _line);
    }
}

void Parser::_readValue(std::istream &is, Value &value)
{
    char s[10];
    char c = _getNonSpaceChar(is);
    if (c == 'n')
    {
        if (!_match(is, "ull", s))
            throw ParseException(std::string("Expected 'null', got 'n") + std::string(s) + "'", _line);
        value = Value();
    }
    else if (c == 't')
    {
        if (!_match(is, "rue", s))
            throw ParseException(std::string("Expected 'true', got 't") + std::string(s) + "'", _line);
        value = Value(true);
    }
    else if (c == 'f')
    {
        if (!_match(is, "alse", s))
            throw ParseException(std::string("Expected 'false', got 'f") + std::string(s) + "'", _line);
        value = Value(false);
    }
    else if (c == '-' || isdigit(c))
    {
        is.putback(c);
        value = Value(_getNumber(is));
    }
    else if (c == '"')
    {
        value = Value(_getString(is));
    }
    else if (c == '[')
    {
        value = Value(ArrayType);
        _readArray(is, value);
    }
    else if (c == '{')
    {
        value = Value(ObjectType);
        _readObject(is, value);
    }
    else
    {
        throw ParseException(std::string("Expected value, object, or array, got '") + c + "'", _line);
    }
}

void Parser::_writeArray(std::ostream &os, const Value &value)
{
    _level++;
    os << '[' << std::endl;
    for (int i = 0; i < value.size(); i++)
    {
        os << std::string(_indent * _level, ' ');
        _writeValue(os, value[i]);
        if (i + 1 < value.size())
            os << ',';
        os << std::endl;
    }
    _level--;
    os << std::string(_indent * _level, ' ') << ']';
}

void Parser::_writeObject(std::ostream &os, const Value &value)
{
    _level++;
    os << '{' << std::endl;
    for (Value::const_iterator it = value.constBegin(); it != value.constEnd(); it++)
    {
        os << std::string(_indent * _level, ' ');
        os << '"' << it.key() << "\" : ";
        _writeValue(os, it.value());
        Value::const_iterator next = it;
        if (++next != value.constEnd())
            os << ',';
        os << std::endl;
    }
    _level--;
    os << std::string(_indent * _level, ' ') << "}";
}

void Parser::_writeValue(std::ostream &os, const Value &value)
{
    switch (value.type())
    {
    case NullType:   os << "null"; break;
    case BoolType:   os << (value.toBool() ? "true" : "false"); break;
    case NumberType: os << value.toDouble(); break;
    case StringType: os << '"' << value.toString() << '"'; break;
    case ArrayType:  _writeArray(os, value); break;
    case ObjectType: _writeObject(os, value); break;
    default:         os << "null"; break;
    }
}

} // namespace Json
