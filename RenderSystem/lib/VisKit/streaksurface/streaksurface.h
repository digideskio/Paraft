#ifndef _STREAK_H_
#define _STREAK_H_

#include "vectors.h"
#include "streaksurface.cuh"
#include <vector>

struct cudaArray;
struct cudaGraphicsResource;
class GLVertexbufferf;
class GLIndexbuffer;
struct Edge;
struct Triangle;
struct Point;
class StreakSurfaceGen {
	int numPoints;
	int numTriangles;
	int numEdges;
	int numSeeds;

	std::shared_ptr<cudaArray> field;
	std::shared_ptr<cudaGraphicsResource> buffers[3];
	cudaGraphicsResource* bufptrs[3];

	std::shared_ptr<GLVertexbufferf> vbo, vbo2;
	std::shared_ptr<GLIndexbuffer> ibo, ibo2, eibo;

	std::shared_ptr<Edge> edges, seedEdges;
	std::shared_ptr<Triangle> triangles, seedTriangles;
	std::shared_ptr<Point> seedPoints;
	SurfaceInfo info, seedInfo, prevSeedInfo;
	float stepSize;
	float maxDistance;

	Vector3 bl;
	Vector3 pitch;

	bool primed;
	int m_state;

	void setBLRScale();
	
	enum StateFlags { GeometrySet = 0x1, FieldSet = 0x2, BottomLeftSet = 0x4, TopRightSet=0x8, StepSizeSet = 0x10, MaxDistanceSet = 0x20, AllSet = 0x3f };
public:


	StreakSurfaceGen(const std::vector<float>& vecs, int x, int y, int z, int channels = 3,
		float stepsize = -1, float maxdistance = -1, Vector3 bottomleft = Vector3(), Vector3 topright = Vector3(1,1,1));
	StreakSurfaceGen();
	~StreakSurfaceGen();

	void generateTestSeeds();
	void setSeeds(
		const std::vector<Point>& seedPointsVec,
		const std::vector<Edge>& seedEdgesVec,
		const std::vector<Triangle>& seedTrianglesVec,
		const std::vector<Edge>& initialEdges);
	void setGeometry(const std::vector<Point>& points, 
		const std::vector<Edge>& edges, 
		const std::vector<Triangle>& triangles,
		int defaultTriangles=0, int defaultEdges=0);
	void setSeeds(const std::vector<Point>& points, bool loops=false);
	void step();
	void draw(bool pointsOnly=false);
	void drawEdges();

	float getStepSize() const { return stepSize; }
	float getMaxDistance() const { return maxDistance; }
	Vector3 getBottomLeft() const { return bl; }
	Vector3 getTopRight() const { return bl + pitch; }
	Vector3 getPitch() const { return pitch; }
	std::shared_ptr<GLVertexbufferf> getVBO() const { return vbo; }
	std::shared_ptr<GLIndexbuffer> getIBO() const { return ibo; }
	void setBottomLeft(const Vector3& v);
	void setTopRight(const Vector3& v);
	void setStepSize(float stepsize);
	void setMaxDistance(float maxDistance);
	void setTexture(const std::vector<float>& data, int width, int height, int depth, int channels=3);
	void setTexture(std::shared_ptr<cudaArray> data);
	void generateSeedLine(const Vector3& begin, const Vector3& end);
	std::shared_ptr<cudaArray> getFieldArray() const { return field; }

	void save(QIODevice* ios);


};

struct CudaFreeFunctor {
	template <typename T>
	void operator()(T* t) const;
};
extern template void CudaFreeFunctor::operator()<cudaArray>(cudaArray* t) const;
extern template void CudaFreeFunctor::operator()<cudaGraphicsResource>(cudaGraphicsResource* t) const;

template<typename T>
void CudaFreeFunctor::operator()(T* t) const {
	if(t)
		cudaFree(t);
}

#endif
