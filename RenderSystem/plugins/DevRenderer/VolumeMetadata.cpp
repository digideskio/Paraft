#include <iostream>
#include <fstream>
#include <sstream>

#include "VolumeMetadata.h"

VolumeMetadata::VolumeMetadata() : _offset(0), _rangeDefined(false) { }

VolumeMetadata::VolumeMetadata(ByteOrder byteOrder, Type type, const Vector3i &dim)
    : _offset(0),
      _byteOrder(byteOrder),
      _type(type),
      _dim(dim),
      _rangeDefined(false) {
}

void VolumeMetadata::read(const Json::Value &val, const Json::Value &globalVal) {
    _fileName = val.isObject() ? val["fileName"].toString() : val.toString();
    _offset = (val.isObject() && val.contains("offset")) ? val["offset"].toInt() : 0;

    const String &byteOrder = (val.isObject() && val.contains("byteOrder")) ?
                                  val["byteOrder"].toString() : globalVal["byteOrder"].toString();
    if (byteOrder == "LITTLE_ENDIAN")   _byteOrder = LITTLE_ENDIAN_;
    else if (byteOrder == "BIG_ENDIAN") _byteOrder = BIG_ENDIAN_;
    else                                _byteOrder = UNKNOWN_ORDER;

    const String &type = (val.isObject() && val.contains("type")) ? val["type"].toString() : globalVal["type"].toString();
    if (type == "UNSIGNED_8BIT")       _type = UNSIGNED_8BIT;
    else if (type == "SIGNED_8BIT")    _type =   SIGNED_8BIT;
    else if (type == "UNSIGNED_16BIT") _type = UNSIGNED_16BIT;
    else if (type == "SIGNED_16BIT")   _type =   SIGNED_16BIT;
    else if (type == "UNSIGNED_32BIT") _type = UNSIGNED_32BIT;
    else if (type == "SIGNED_32BIT")   _type =   SIGNED_32BIT;
    else if (type == "FLOAT")          _type = FLOAT;
    else if (type == "DOUBLE")         _type = DOUBLE;
    else                               _type = UNKNOWN_TYPE;

    if (val.isObject() && val.contains("dim")) {
        _dim.x = val["dim"][0].toInt();
        _dim.y = val["dim"][1].toInt();
        _dim.z = val["dim"][2].toInt();
    } else {
        _dim.x = globalVal["dim"][0].toInt();
        _dim.y = globalVal["dim"][1].toInt();
        _dim.z = globalVal["dim"][2].toInt();
    }

    _range.x = (val.isObject() && val.contains("min")) ? val["min"].toDouble() : (globalVal.contains("min") ? globalVal["min"].toDouble() : 0.0);
    _range.y = (val.isObject() && val.contains("max")) ? val["max"].toDouble() : (globalVal.contains("max") ? globalVal["max"].toDouble() : 0.0);
    _rangeDefined = (val.isObject() && val.contains("min") && val.contains("max") || globalVal.contains("min") && globalVal.contains("max"));

    std::cout << _byteOrder << std::endl;
    std::cout << _type << std::endl;
    std::cout << _dim.x << ", " << _dim.y << ", " << _dim.z << std::endl;
    std::cout << _fileName << std::endl;
    std::cout << _range.x << ' ' << _range.y << std::endl;
}

void VolumeMetadata::write(Json::Value &val) const {
    val["fileName"] = _fileName;
    String byteOrder = "UNKNOWN_ORDER";
    switch (_byteOrder) {
        case LITTLE_ENDIAN_: byteOrder = "LITTLE_ENDIAN"; break;
        case BIG_ENDIAN_:    byteOrder = "BIG_ENDIAN";    break;
    }
    val["byteOrder"] = byteOrder;
    String type = "UNKNOWN_TYPE";
    switch (_type) {
        case UNSIGNED_8BIT:  type = "UNSIGNED_8BIT";  break;
        case SIGNED_8BIT:    type = "SIGNED_8BIT";    break;
        case UNSIGNED_16BIT: type = "UNSIGNED_16BIT"; break;
        case SIGNED_16BIT:   type = "SIGNED_16BIT";   break;
        case UNSIGNED_32BIT: type = "UNSIGNED_32BIT"; break;
        case SIGNED_32BIT:   type = "SIGNED_32BIT";   break;
        case FLOAT:          type = "FLOAT";          break;
        case DOUBLE:         type = "DOUBLE";         break;
    }
    val["type"] = type;
    val["dim"].append(_dim.x);
    val["dim"].append(_dim.y);
    val["dim"].append(_dim.z);
    val["min"] = _range.x;
    val["max"] = _range.y;
}

VolumeMetadata::ByteOrder VolumeMetadata::nativeByteOrder() {
    short test = 1;
    if (*(char *)(&test) == 1)
        return LITTLE_ENDIAN_;
    else
        return BIG_ENDIAN_;
}

TVMVVolumeMetadata::TVMVVolumeMetadata() { }
TVMVVolumeMetadata::TVMVVolumeMetadata(const String &name) : _name(name) { }

int TVMVVolumeMetadata::getVarIndex(const String &varName) const {
    for (unsigned int i = 0; i < _varNames.size(); i++) {
        if (_varNames[i] == varName) {
            return i;
        }
    }
    return -1;
}

void TVMVVolumeMetadata::readFile(const String &fileName) {
    _varNames.clear();
    _volumes.clear();

    Json::Value root;
    Json::Parser().parseFile(fileName, root);

    _name = root["name"].toString();

    int stepCount = root.contains("stepCount") ? root["stepCount"].toInt() : 1;
    int varCount = root.contains("varCount") ? root["varCount"].toInt() : 1;

    if (root.contains("varNames")) {
        for (int i = 0; i < varCount; i++) {
            _varNames.append(root["varNames"][i].toString());
            std::cout << _varNames[i] << std::endl;
        }
    } else {
        for (int i = 0; i < varCount; i++) {
            std::stringstream ss;
            ss << "Variable " << i + 1;
            _varNames.append(ss.str());
        }
    }

    if (!root.contains("volumes")) {
        _volumes.append(Vector<VolumeMetadata>());
        _volumes[0].append(VolumeMetadata());
        _volumes[0][0].read(root, root);
    } else {
        for (int i = 0; i < stepCount; i++) {
            _volumes.append(Vector<VolumeMetadata>());
            for (int j = 0; j < varCount; j++) {
                _volumes[i].append(VolumeMetadata());
                if (varCount == 1 && !root["volumes"][i].isArray())
                    _volumes[i][j].read(root["volumes"][i], root);
                else if (stepCount == 1 && !root["volumes"][0].isArray())
                    _volumes[i][j].read(root["volumes"][j], root);
                else
                    _volumes[i][j].read(root["volumes"][i][j], root);
            }
        }
    }
}

void TVMVVolumeMetadata::writeFile(const String &fileName) const {
    Json::Value root;
    root["name"] = _name;
    root["stepCount"] = stepCount();
    root["varCount"] = varCount();
    for (unsigned int i = 0; i < _varNames.size(); i++) {
        root["varNames"].append(_varNames[i]);
    }
    root["volumes"].resize(stepCount());
    for (int i = 0; i < stepCount(); i++) {
        root["volumes"][i].resize(varCount());
        for (int j = 0; j < varCount(); j++) {
            _volumes[i][j].write(root["volumes"][i][j]);
        }
    }
    Json::Parser().writeFile(fileName, root);
}
