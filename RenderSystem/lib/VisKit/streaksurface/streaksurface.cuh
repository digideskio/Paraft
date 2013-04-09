#ifndef _STREAK_CUH_
#define _STREAK_CUH_

#include <cuda_runtime.h>

struct SurfaceInfo {
	unsigned int numPoints;
	unsigned int numEdges;
	unsigned int numTriangles;
	__host__ __device__
	SurfaceInfo() {}
	__host__ __device__
	SurfaceInfo(const SurfaceInfo& other) {
		*this = other;
	}
	__host__ __device__
	SurfaceInfo& operator-=(const SurfaceInfo& rhs) {
		numPoints -= rhs.numPoints;
		numEdges -= rhs.numEdges;
		numTriangles -= rhs.numTriangles;
		return *this;
	}
	__host__ __device__
	SurfaceInfo& operator=(const SurfaceInfo& rhs) {
		numPoints = rhs.numPoints;
		numEdges = rhs.numEdges;
		numTriangles = rhs.numTriangles;
		return *this;
	}
	SurfaceInfo(int p, int e, int t):numPoints(p), numEdges(e), numTriangles(t) {}
};

struct Edge {
	int p1;
	int p2;
	int newEdge;
	int t1;
	int t2;
	int id;
	__host__ __device__
	Edge() {}
	__host__ __device__
	Edge(const Edge& other) {
		*this = other;
	}
	__host__ __device__
	Edge& operator=(const Edge& rhs) {
		p1 = rhs.p1;
		p2 = rhs.p2;
		newEdge = rhs.newEdge;
		id = rhs.id;
		t1 = rhs.t1;
		t2 = rhs.t2;
		return *this;
	}
	__host__ __device__
	bool isValid() const {
		return (id >= 0) && (p1 != p2);
	}
	bool operator==(const Edge& rhs) const {
		return ((p1 < p2 ? p1 : p2) == (rhs.p1 < rhs.p2 ? rhs.p1 : rhs.p2)) &&
			((p1 < p2 ? p2 : p1) == (rhs.p1 < rhs.p2 ? rhs.p2 : rhs.p1));
	}
};

struct Triangle {
	int e1;
	int e2;
	int e3;
	__host__ __device__
	Triangle() {}
	__host__ __device__
	Triangle(const Triangle& other) {
		*this = other;
	}
	__host__ __device__
	Triangle& operator=(const Triangle& rhs) {
		e1 = rhs.e1;
		e2 = rhs.e2;
		e3 = rhs.e3;
		return *this;
	}
	__host__ __device__
	bool isValid() const {
		return e1 != e2;
	}
};

struct Point {
	float locX;
	float locY;
	float locZ;
	float texposX;
	union {
		float texposY;
		int reassignId;
	};
	float normalX; 
	float normalY; 
	float normalZ;
	__host__ __device__
	Point() {}
	__host__ __device__
	Point(const Point& other) {
		*this = other;
	}
	__host__ __device__
	Point& operator=(const Point& rhs) {
		locX = rhs.locX;
		locY = rhs.locY;
		locZ = rhs.locZ;

		texposX = rhs.texposX;
		texposY = rhs.texposY;

		normalX = rhs.normalX;
		normalY = rhs.normalY;
		normalZ = rhs.normalZ;

		return *this;
	}
	__host__ __device__
	float2 texpos() const {
		return make_float2(texposX, texposY);
	}
	__host__ __device__
	float3 normal() const {
		return make_float3(normalX, normalY, normalZ);
	}
	__host__ __device__
	float3 location() const { 
		return make_float3(locX, locY, locZ);
	}
	__host__ __device__
	void setLocation(float3 loc, bool check=false) {
		if(check && locX == loc.x && locY == loc.y && loc.z == locZ) {
			texposX = -1.f;
			reassignId = -1;
		}
		locX = loc.x;
		locY = loc.y;
		locZ = loc.z;
	}
	__host__ __device__
	void setNormal(float3 n) {
		normalX = n.x;
		normalY = n.y;
		normalZ = n.z;
	}
	__host__ __device__
	void setTexpos(float2 t) {
		texposX = t.x;
		texposY = t.y;
	}

	__host__ __device__
	bool isValid() const {
		return (texposX >= 0.f);
	}
	
	__host__ __device__
	bool isDead() const { //is not valid and is not reassigned
		return (!isValid() && reassignId < 0);
	}
	__host__ __device__
	bool isReassigned() const { 
		return (!isValid() && reassignId >= 0);
	}
};
void fillEdges(int numEdges, Edge* edges, uint2* indices);
void primeSeeds(int numSeeds, Point* points);
SurfaceInfo removeDead(SurfaceInfo& info, Point* points, Edge* edges, Triangle* triangles);
void buildTriangles(SurfaceInfo& info, Point* points, uint3* indices, Edge* edges, Triangle* triangles);
void advectAndRefine(SurfaceInfo& info, Point* points, Edge* edges, Triangle* triangles);
void plantSeeds(const SurfaceInfo& seedInfo, SurfaceInfo& info, 
	Point* points, Edge* edges, Triangle* triangles, 
	Point* seedPoints, Edge* seedEdges, Triangle* seedTriangles, SurfaceInfo& prevSeedLoc);
void initTex();

#endif
