#ifndef _BOX_H_
#define _BOX_H_

#include "vectors.h"
#include "camera.h"
#include "slicer.h"

class GLVertexbufferf;
class GLIndexbuffer;
struct Box {
	Vector3 bottomleft;
	Vector3 topright;
	Vector3 pitch;

	GLVertexbufferf* vbo;
	GLIndexbuffer* ibo;

	Box():bottomleft(0,0,0), topright(1,1,1), pitch(1,1,1), vbo(0), ibo(0) {}
	Box(const Vector3& bottomleft, const Vector3& pitch);
	void drawBox();
	void drawSlice(const Camera& cam, const double& sDepth) const;
	void drawSlice(const CameraOptions & co, const double& sDepth) const;
	void drawSlicedBox(const Vector3& sliceVec, const double& sDepth, bool sliceonly = false) const;
	void drawMultiSlicedBox(const QList<Slicer> & slicers, bool sliceonly = false) const;
	void rayBoxIntersection(double& t1, double& t2, const Vector3& start, 
		const Vector3& ray, unsigned char planes=7) const;
	Vector3 getCenterPoint() const;
	virtual void setPitch(const Vector3& p);
	virtual void setBottomleft(const Vector3& b);
	virtual void setTopright(const Vector3& p);
	Box& operator=(const Box& rhs);
	void constructVBO();
};


#endif
