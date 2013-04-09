#include "CProperties.h"
extern bool verbose;
#include <cstring>
#include <cstdio>

void CProperties::addProperty(char *_name, float _val, float _inc, unsigned char key1, unsigned char key2) {
  CProperty p ;
  strncpy(p.name, _name, 80);
  p.value = _val;
  p.increment = _inc;
  p.incKey = key1;
  p.decKey = key2;
  properties.push_back(p);
}

int CProperties::numProperties() {
  return static_cast<int>(properties.size());
}

void CProperties::getProperty(int i, char *name, float *value) {
  strncpy(name, properties[i].name, 80);
  *value = properties[i].value;
}

void CProperties::setProperty(char *name, float value) {
  int i=findProperty(name);
  if(i<0) {
    addProperty(name, value, 0,0,0);
    return;
  }
  else {
    properties[i].value = value;
  }
}

int CProperties::findProperty(char *name) {
  for(unsigned int i=0;i<properties.size();i++) {
    if(!strncmp(name, properties[i].name, 80)) {
      return i;
    }
  }
  return -1;
}

bool CProperties::pressKey(unsigned char key) {
  for(unsigned int i=0;i<properties.size();i++) {
    if(properties[i].incKey==key) {
      properties[i].value+=properties[i].increment;
      printf("[PROPERTIES] %s = %f\n", properties[i].name, properties[i].value);
      return true;
    }
    if(properties[i].decKey==key) {
      properties[i].value-=properties[i].increment;
      printf("[PROPERTIES] %s = %f\n", properties[i].name, properties[i].value);
      return true;
    }
  }
  return false;
}
float CProperties::getPropertyValue(char *name, float defaultValue) {
  int i = findProperty(name);
  if(i<0) return defaultValue;
  float value;
  char pname[80];
  getProperty(i, pname, &value);
  return value;
}
