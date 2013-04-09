#ifndef _FRUSTRUM_H_
#define _FRUSTRUM_H_



#include "vectors.h"

class Camera;
class Frustrum {
	Camera* cam;
	
	Vector3 n_normal, f_normal, l_normal, r_normal, t_normal, b_normal;
	Vector3 n_point, f_point, l_point, r_point, t_point, b_point;
	
	void setPlanes();
	void doTransform();
	public:
		Frustrum(Camera* cam):cam(cam) {};
		void update();
		bool inside(const Vector3 &point) const;
};




#endif
