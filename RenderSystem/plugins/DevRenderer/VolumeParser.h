#ifndef VOLUMEPARSER_H
#define VOLUMEPARSER_H

#include <QtCore>
#include "vectors.h"
#include "CData.h"

struct VolumeData {
    VolumeData() {
        name = NULL;
        filename = NULL;
        tfefile = NULL;
        totalstep = 1;
        startstep = 1;
        numdigits = -1;
        dim = Vector3(0.0,0.0,0.0);
        format = UNSIGNED_8BIT;
        byteorder = BIGENDIAN;
        ifRange = false;
        this->max = 1E+20;
        this->min = -1E+20;
    }
    ~VolumeData() {
        if (name) delete [] name;
        if (filename) delete [] filename;
        if (tfefile) delete [] tfefile;
    }
    char * name;
    char * filename;
    char * tfefile;
    int totalstep;
    int startstep;
    int numdigits;
    Vector3 dim;
    MESHATT format;
    BYTEORDER byteorder;
    bool ifRange;
    double max;
    double min;
};

class VolumeParser {
public:
    VolumeParser();
    VolumeParser(const char *);
    ~VolumeParser();

    void setVolumeData(const QString &);
    void setVolumeData(const char *);
    int size() const { return m_volumedata.size(); }

    VolumeData & operator[](const int & idx);
    VolumeData & operator()(const QString & idx);

protected:
    void parseVolumeData(const char *);
    bool parseVolumeAttribute(VolumeData *, QByteArray, QByteArray);
    QList<VolumeData*>		m_volumedata;
    QHash<QString, VolumeData*>	m_volumedatahash;
};

#endif // VOLUMEPARSER_H
