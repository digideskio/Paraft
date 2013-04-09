#ifndef VOLUMEMETADATA_H
#define VOLUMEMETADATA_H

#include <iostream>
#include <string>

#include "MSVectors.h"
#include "Containers.h"

#include "JsonParser.h"

typedef std::string String;

class VolumeMetadata
{
public:
    enum ByteOrder { UNKNOWN_ORDER, LITTLE_ENDIAN_, BIG_ENDIAN_ };  // postfix '_': avoid conflict
    enum Type      { UNKNOWN_TYPE,
                     UNSIGNED_8BIT, SIGNED_8BIT,
                     UNSIGNED_16BIT, SIGNED_16BIT,
                     UNSIGNED_32BIT, SIGNED_32BIT,
                     FLOAT, DOUBLE };

    VolumeMetadata();
    VolumeMetadata(ByteOrder byteOrder, Type type, const Vector3i &dim);

    const String   &fileName()     const { return _fileName; }
    int             offset()       const { return _offset; }
    ByteOrder       byteOrder()    const { return _byteOrder; }
    Type            type()         const { return _type; }
    const Vector3i &dim()          const { return _dim; }
    const Vector2d &range()        const { return _range; }
    double          min()          const { return _range.x; }
    double          max()          const { return _range.y; }
    bool            rangeDefined() const { return _rangeDefined; }

    void setFileName(const String &fileName) { _fileName = fileName; }
    void setRange(double min, double max)    { _range = Vector2d(min, max); }

    void read(const Json::Value &val, const Json::Value &globalVal);
    void write(Json::Value &val) const;

    static ByteOrder nativeByteOrder();

protected:
    String    _fileName;
    int       _offset;
    ByteOrder _byteOrder;
    Type      _type;
    Vector3i  _dim;
    //double    _min, _max;
    Vector2d  _range;
    bool      _rangeDefined;
};

//
// time-varying multivariate volume metadata
//
class TVMVVolumeMetadata
{
public:
    TVMVVolumeMetadata();
    TVMVVolumeMetadata(const String &name);

    const String &name()      const { return _name; }
    int           stepCount() const { return (int)_volumes.size(); }
    int           varCount()  const { return (_volumes.size() > 0) ? (int)_volumes[0].size() : 0; }
    const String &getVarName(int varIdx) const { return _varNames[varIdx]; }
    int           getVarIndex(const String &varName) const;

    VolumeMetadata       &getVolumeMetadata(int timeStep = 0, int varIdx = 0)       { return _volumes[timeStep][varIdx]; }
    const VolumeMetadata &getVolumeMetadata(int timeStep = 0, int varIdx = 0) const { return _volumes[timeStep][varIdx]; }
    //void addVolume(const VolumeDescriptor &volume) { _volumesOld.append(volume); }
    void readFile(const String &fileName);
    void writeFile(const String &fileName) const;

protected:
    String _name;
    Vector<String> _varNames;
    Vector< Vector<VolumeMetadata> > _volumes;    // [timeStep][varIdx]
};

#endif // VOLUMEMETADATA_H
