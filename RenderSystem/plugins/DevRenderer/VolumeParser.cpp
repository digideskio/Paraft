#include "VolumeParser.h"

VolumeParser::VolumeParser() { }
VolumeParser::VolumeParser(const char * datafile) {
    parseVolumeData(datafile);
}
VolumeParser::~VolumeParser() {
    for (int i = 0; i < m_volumedata.size(); ++i) delete m_volumedata[i];
}
void VolumeParser::setVolumeData(const QString & datafile) {
    setVolumeData(datafile.toAscii().constData());
}
void VolumeParser::setVolumeData(const char * datafile) {
    for (int i = 0; i < m_volumedata.size(); ++i) delete m_volumedata[i];
    m_volumedata.clear();
    m_volumedatahash.clear();
    parseVolumeData(datafile);
}
void VolumeParser::parseVolumeData(const char * datafile) {
    QFile inpfile(datafile);
    inpfile.open(QIODevice::ReadOnly);
    QByteArray buffer = inpfile.readAll();
    QByteArray name, attr, value;
    enum State {OUT, ITEM, VALUE};
    State state = OUT;

    VolumeData * temp = NULL;

    for (int i = 0; i < buffer.size(); ++i) {
        switch(state) {
        case OUT:
            if (buffer[i] == '{') {
                name = name.trimmed();
                name += '\0';
                temp = new VolumeData;
                temp->name = new char[name.size()+1];
                qstrcpy(temp->name, name.constData());
                state = ITEM;
            }
            else name += buffer[i];
            break;
        case ITEM:
            if (buffer[i] == '=') {
                attr = attr.trimmed();
                state = VALUE;
            }
            else if (buffer[i] == '}') {
                m_volumedata.append(temp);
                m_volumedatahash[QString(name)] = temp;
                name.clear(); attr.clear(); value.clear(); state = OUT;
            }
            else attr += buffer[i];
            break;
        case VALUE:
            if (buffer[i] == ',') {
                value = value.trimmed();
                if (!parseVolumeAttribute(temp, attr, value)) fprintf(stderr, "Volume Data Format Error!\n");
                attr.clear(); value.clear(); state = ITEM;
            }
            else if (buffer[i] == '}') {
                value = value.trimmed();
                if (!parseVolumeAttribute(temp, attr, value)) fprintf(stderr, "Volume Data Format Error!\n");
                m_volumedata.append(temp);
                m_volumedatahash[QString(name)] = temp;
                name.clear(); attr.clear(); value.clear(); state = OUT;
            }
            else value += buffer[i];
            break;
        }
    }
}

bool VolumeParser::parseVolumeAttribute(VolumeData * vd, QByteArray attr, QByteArray value) {
    if (attr == "filename") {
        value += '\0';
        vd->filename = new char[value.size() + 1];
        qstrcpy(vd->filename, value.constData());
        return true;
    }
    else if (attr == "tfefile") {
        value += '\0';
        vd->tfefile = new char[value.size() + 1];
        qstrcpy(vd->tfefile, value.constData());
        return true;
    }
    else if (attr == "totalstep") {
        vd->totalstep = value.toInt();
        return true;
    }
    else if (attr == "startstep") {
        vd->startstep = value.toInt();
        return true;
    }
    else if (attr == "numdigits") {
        vd->numdigits = value.toInt();
        return true;
    }
    else if (attr == "dim") {
        QList<QByteArray> dim = value.split('x');
        if (dim.size() != 3) return false;
        vd->dim = Vector3(dim[0].toInt(), dim[1].toInt(), dim[2].toInt());
        return true;
    }
    else if (attr == "format") {
        if (value == "UNSIGNED_8BIT") { vd->format = UNSIGNED_8BIT; return true; }
        else if (value == "SIGNED_8BIT") { vd->format = SIGNED_8BIT; return true; }
        else if (value == "UNSIGNED_16BIT") { vd->format = UNSIGNED_16BIT; return true; }
        else if (value == "SIGNED_16BIT") { vd->format = SIGNED_16BIT; return true; }
        else if (value == "UNSIGNED_32BIT") { vd->format = UNSIGNED_32BIT; return true; }
        else if (value == "SIGNED_32BIT") { vd->format = SIGNED_32BIT; return true; }
        else if (value == "FLOATT") { vd->format = FLOATT; return true; }
        else if (value == "DOUBLEE") { vd->format = DOUBLEE; return true; }
        else return false;
    }
    else if (attr == "byteorder") {
        if (value == "BIGENDIAN") { vd->byteorder = BIGENDIAN; return true; }
        else if (value == "LITTEENDIAN") { vd->byteorder = LITTEENDIAN; return true; }
        else return false;
    }
    else if (attr == "max") {
        vd->max = value.toDouble();
        if (vd->max<1E+20 && vd->min > -1E+20) vd->ifRange = true;
        return true;
    }
    else if (attr == "min") {
        vd->min = value.toDouble();
        if (vd->max<1E+20 && vd->min > -1E+20) vd->ifRange = true;
        return true;
    }
    return false;
}
VolumeData & VolumeParser::operator[](const int & idx) {
    return *(m_volumedata[idx]);
}
VolumeData & VolumeParser::operator()(const QString & idx) {
    return *(m_volumedatahash[idx]);
}
