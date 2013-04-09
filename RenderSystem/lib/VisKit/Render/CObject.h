#ifndef _COBJECT_H_
#define _COBJECT_H_

class CProperties;

class CObject {
 protected:
  CProperties* m_pGlobalProperties;
  
 public:
  CObject() {}
  ~CObject() {} 
  inline void setGlobalProperties(CProperties *props) {
    m_pGlobalProperties = props;
  }
  inline CProperties* getGlobalProperties() {
    return m_pGlobalProperties;
  }
};


#endif

