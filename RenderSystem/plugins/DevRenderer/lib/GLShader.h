#ifndef GLSHADER_H
#define GLSHADER_H

#include <GL/glew.h>

////
#include <iostream>

#include "MSVectors.h"
#include "Containers.h"
#include "ParameterSet.h"
#include "MSGLTexture.h"

namespace MSLib
{

template <typename T> inline void setParameterValue(Parameter &, const T &) {}
template <> inline void setParameterValue(Parameter &param, const bool  &value) { param.setValue(value); }
template <> inline void setParameterValue(Parameter &param, const int   &value) { param.setValue(value); }
template <> inline void setParameterValue(Parameter &param, const float &value) { param.setValue(value); }

template <typename T> inline const T &getParameterValue(const Parameter &) { static T t; return t; }
template <> inline const bool  &getParameterValue(const Parameter &param) { return param.toBool(); }
template <> inline const int   &getParameterValue(const Parameter &param) { return param.toInt(); }
template <> inline const float &getParameterValue(const Parameter &param) { return param.toFloat(); }

class GLAbstractUniform
{
public:
    enum Type
    {
        Void = 0,
        Bool = 2,
        Int,
        Float,
        Double,
        Vec2,
        Vec3,
        Vec4,
        DVec2,
        DVec3,
        DVec4
    };

    //GLUniform() {}
    GLAbstractUniform(const String &name, GLuint location, Parameter *param = nullptr) : _name(name), _location(location), _param(param) {}
    virtual ~GLAbstractUniform() {}
    String name() const { return _name; }
    //Type type() const { return _type; }
    GLuint location() const { return _location; }
    void setLocation(GLuint location) { _location = location; }
    void bindParameter(Parameter *param) { _param = param; }
    //virtual void update() = 0;
    virtual Type type() const { return Void; }  ////
    virtual void set() = 0;

protected:
    String _name;
    //Type _type;
    GLuint _location;
    Parameter *_param;
};

template <typename T>
class GLUniform : public GLAbstractUniform
{
public:
    typedef T ValueType;

    GLUniform(const String &name, GLuint location, T value)
        : GLAbstractUniform(name, location), _count(1)
    {
        _value = new T(value);
    }
    GLUniform(const String &name, GLuint location, int count, const T *values)
        : GLAbstractUniform(name, location), _count(count)
    {
        _value = new T[_count];
        for (int i = 0; i < _count; i++)
            _value[i] = values[i];
    }
    GLUniform(const String &name, GLuint location, Parameter *param)
        : GLAbstractUniform(name, location, param)
    {
        _count = 1;     ////
    }
    ~GLUniform()
    {
        if (_param == nullptr)
        {
            if (_count == 1)
                delete _value;
            else
                delete [] _value;
        }
    }
    void setValue(const T &value)
    {
        if (_param != nullptr)
            MSLib::setParameterValue(*_param, value);
        else
            *_value = value;
    }
    void setValues(const T *values)
    {
        if (_param != nullptr)
        {}                      ////
        else
        {
            for (int i = 0; i < _count; i++)
                _value[i] = values[i];
        }
    }
    T value() const
    {
        //return (_param != nullptr) ? _param->value<T>() : *_value;
        return (_param != nullptr) ? MSLib::getParameterValue<T>(*_param) : *_value;
    }
    const T *values() const
    {
        //return (_param != nullptr) ? &_param->value<T>() : _value;
        return (_param != nullptr) ? &MSLib::getParameterValue<T>(*_param) : _value;
    }
    Type type() const
    {
        String typeId = typeid(T).name();
        if (typeId == "bool") return GLAbstractUniform::Bool;
        else if (typeId == "int") return GLAbstractUniform::Int;
        else if (typeId == "float") return GLAbstractUniform::Float;
        else if (typeId == "double") return GLAbstractUniform::Double;
        else if (typeId == "class Vector2f") return GLAbstractUniform::Vec2;
        else if (typeId == "class Vector3f") return GLAbstractUniform::Vec3;
        else if (typeId == "class Vector4f") return GLAbstractUniform::Vec4;
        //return _type<T>();
        else return GLAbstractUniform::Void;
    }
    void set()
    {
        //const T *val = values(); _print(val[0]);
        _set(values());
    }

protected:
    //template <typename Typ> GLUniformBase::Type _type() const { return GLUniformBase::Void; }
    //template <> GLUniformBase::Type _type<bool>() const { return GLUniformBase::Bool; }
    //template <> GLUniformBase::Type _type<int>() const { return GLUniformBase::Int; }
    //template <> GLUniformBase::Type _type<float>() const { return GLUniformBase::Float; }
    //template <> GLUniformBase::Type _type<double>() const { return GLUniformBase::Double; }
    //template <> GLUniformBase::Type _type<Vector2f>() const { return GLUniformBase::Vec2; }
    //template <> GLUniformBase::Type _type<Vector3f>() const { return GLUniformBase::Vec3; }
    //template <> GLUniformBase::Type _type<Vector4f>() const { return GLUniformBase::Vec4; }

    void _set(const bool *values)
    {
        if (_count == 1)
            glUniform1i(_location, *values ? 1 : 0);
        else                                            // need to be tested
        {
            // since the size of bool is 1
            int *ivalues = new int[_count];
            for (int i = 0; i < _count; i++)
                ivalues[i] = values[i] ? 1 : 0;
            glUniform1iv(_location, _count, ivalues);
            delete [] ivalues;
        }
    }
    void _set(const int *values) { glUniform1iv(_location, _count, values); }
    void _set(const float *values) { glUniform1fv(_location, _count, values); }
    void _set(const Vector2f *values) { glUniform2fv(_location, _count, (float *)values); }
    void _set(const Vector3f *values) { glUniform3fv(_location, _count, (float *)values); }
    void _set(const Vector4f *values) { glUniform4fv(_location, _count, (float *)values); }

    // debug
    void _print(bool val) { std::cout << _name << '=' << (val ? "true" : "false") << std::endl; }
    void _print(int val) { std::cout << _name << '=' << val << std::endl; }
    void _print(float val) { std::cout << _name << '=' << val << std::endl; }
    void _print(const Vector2f &val) { std::cout << _name << '=' << val.x << ',' << val.y << std::endl; }
    void _print(const Vector3f &val) { std::cout << _name << '=' << val.x << ',' << val.y << ',' << val.z << std::endl; }
    void _print(const Vector4f &val) { std::cout << _name << '=' << val.x << ',' << val.y << ',' << val.z << ',' << val.w << std::endl; }

protected:
    int _count;
    T *_value;
};

class GLShader
{
public:
    GLShader();
    ~GLShader();
    void setVertexShader(const GLchar *string, GLint length);
    void setFragmentShader(const GLchar *string, GLint length);
    void link();
    void use();
    static void useFixed();
    GLint getAttribLocation(const GLchar *name);                            // after link()
    GLint getUniformLocation(const GLchar *name);                           // after link()
    void setUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);  // after use()
    void setUniform1i(GLint location, GLint v0);
    void printShaderInfoLog(GLuint obj);
    void printVertexShaderInfoLog();
    void printFragmentShaderInfoLog();
    void printProgramInfoLog();

    // high-level
    bool loadVertexShader(const char *fileName);
    bool loadFragmentShader(const char *fileName);
    bool reload();

    template <typename T> void addUniform(const String &name, const T &value);
    template <typename T> void addUniform(const String &name, int count, const T *values);
    template <typename T> void addUniform(const String &name, Parameter *param);
    template <typename T> void setUniform(const String &name, const T &value);
    template <typename T> void setUniform(const String &name, const T *values);

    void addUniform1b(const String &name, bool value = false) { addUniform<bool>(name, value); }    ////
    void addUniform1i(const String &name, int value = 0)      { addUniform<int>(name, value); }
    void addUniform1f(const String &name, float value = 0.0f) { addUniform<float>(name, value); }
    void addUniform2f(const String &name, float v0 = 0.0f, float v1 = 0.0f)                                   { addUniform<Vector2f>(name, Vector2f(v0, v1)); }         ////
    void addUniform3f(const String &name, float v0 = 0.0f, float v1 = 0.0f, float v2 = 0.0f)                  { addUniform<Vector3f>(name, Vector3f(v0, v1, v2)); }
    void addUniform4f(const String &name, float v0 = 0.0f, float v1 = 0.0f, float v2 = 0.0f, float v3 = 0.0f) { addUniform<Vector4f>(name, Vector4f(v0, v1, v2, v3)); }
    void addUniform3f(const String &name, const Vector3f &value = Vector3f()) { addUniform<Vector3f>(name, value); }
    void addUniform3fv(const String &name, int count, const float *values) { addUniform<Vector3f>(name, count, reinterpret_cast<const Vector3f *>(values)); }

    void addUniform1b(const String &name, Parameter *param) { addUniform<bool> (name, param); }
    void addUniform1i(const String &name, Parameter *param) { addUniform<int>  (name, param); }
    void addUniform1f(const String &name, Parameter *param) { addUniform<float>(name, param); }

    void addUniformSampler(const String &name, int textureUnit = 0) { addUniform<int>(name, textureUnit); }
    void addUniformSampler(const String &name, const MSLib::GLTexture &texture) { addUniform<int>(name, texture.boundUnit()); }

    void setUniform1b(const String &name, bool value)  { setUniform<bool> (name, value); }
    void setUniform1i(const String &name, int value)   { setUniform<int>  (name, value); }
    void setUniform1f(const String &name, float value) { setUniform<float>(name, value); }
    void setUniform2f(const String &name, float v0, float v1)                     { setUniform<Vector2f>(name, Vector2f(v0, v1)); }         ////
    void setUniform3f(const String &name, float v0, float v1, float v2)           { setUniform<Vector3f>(name, Vector3f(v0, v1, v2)); }
    void setUniform4f(const String &name, float v0, float v1, float v2, float v3) { setUniform<Vector4f>(name, Vector4f(v0, v1, v2, v3)); }
    void setUniform3f(const String &name, const Vector3f &value) { setUniform<Vector3f>(name, value); }
    void setUniform3fv(const String &name, const float *values) { setUniform<Vector3f>(name, reinterpret_cast<const Vector3f *>(values)); }

    void setUniformSampler(const String &name, int textureUnit) { setUniform<int>(name, textureUnit); }
    void setUniformSampler(const String &name, const MSLib::GLTexture &texture) { setUniform<int>(name, texture.boundUnit()); }

    void setUniforms();

protected:
    bool _loadVertexShader();
    bool _loadFragmentShader();

protected:
    GLuint _vertShaderId;
    GLuint _fragShaderId;
    GLuint _programId;
    bool _vertShaderAttached;
    bool _fragShaderAttached;

    String _vertShaderFileName;
    String _fragShaderFileName;
    Hash<String, GLAbstractUniform *> _uniforms;
};

template <typename T>
inline void GLShader::addUniform(const String &name, const T &value)
{
    if (_uniforms.contains(name))
    {
        // warning
        return;
    }
    _uniforms[name] = new GLUniform<T>(name, getUniformLocation(name.c_str()), value);
}

template <typename T>
inline void GLShader::addUniform(const String &name, int count, const T *values)
{
    if (_uniforms.contains(name))
    {
        // warning
        return;
    }
    _uniforms[name] = new GLUniform<T>(name, getUniformLocation(name.c_str()), count, values);
}

template <typename T>
inline void GLShader::addUniform(const String &name, Parameter *param)
{
    if (_uniforms.contains(name))
    {
        // warning
        return;
    }
    _uniforms[name] = new GLUniform<T>(name, getUniformLocation(name.c_str()), param);
}

template <typename T>
inline void GLShader::setUniform(const String &name, const T &value)
{
    if (!_uniforms.contains(name))
    {
        // warning
        return;
    }
    // test type
    static_cast<GLUniform<T> *>(_uniforms[name])->setValue(value);
}

template <typename T>
inline void GLShader::setUniform(const String &name, const T *values)
{
    if (!_uniforms.contains(name))
    {
        // warning
        return;
    }
    // test type
    static_cast<GLUniform<T> *>(_uniforms[name])->setValues(values);
}

} // namespace MSLib

typedef MSLib::GLShader GLShader;

#endif // GLSHADER_H
