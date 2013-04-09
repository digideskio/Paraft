#ifndef _CPROPERTY_H
#define _CPROPERTY_H

#include <vector>
using namespace std;

typedef struct{
  char name[80];
  float value;
  float increment;
  unsigned char incKey, decKey;
} CProperty;

class CProperties {
  vector<CProperty> properties;
 public:
  CProperties() {}
  void addProperty(char *_name, float _val, float _inc, unsigned char key1, unsigned char key2);
  int numProperties();
  void getProperty(int i, char *name, float *value);
  void setProperty(char *name, float value);
  int findProperty(char *name);
  bool pressKey(unsigned char key);
  float getPropertyValue(char *name, float defaultValue);
};


#endif
