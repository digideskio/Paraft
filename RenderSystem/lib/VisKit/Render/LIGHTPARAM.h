#ifndef _LIGHTPARAM_H_
#define _LIGHTPARAM_H_

class LIGHTPARAM {
public:
	LIGHTPARAM(){Kamb=0.4f;Kdif=0.7f;Kspe=1.0f;Kshi=20.0f;}
	LIGHTPARAM(const LIGHTPARAM & l) { (*this) = l; }
	LIGHTPARAM & operator=(const LIGHTPARAM & src) {
		this->Kamb = src.Kamb;
		this->Kdif = src.Kdif;
		this->Kspe = src.Kspe;
		this->Kshi = src.Kshi;
		return *this;
	}
	float Kamb,Kdif,Kspe,Kshi;
};

#endif
