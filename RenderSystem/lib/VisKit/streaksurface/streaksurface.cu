#include <cuda_runtime.h>
#include <cutil_math.h>
#include <device_functions.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include "streaksurface.cuh"
#define CUDAERROR(x) x; for(cudaError_t error = cudaGetLastError(); error != cudaSuccess; error = cudaGetLastError()) printError(error, __LINE__, __FILE__)
void printError(cudaError_t error, int line, const char* file);
texture<float4, 3, cudaReadModeElementType> field;

__device__ __constant__ float3 rscale;
__device__ __constant__ float3 bottomleft;
__device__ __constant__ float stepsize;
__device__ __constant__ float maxdistance;

__device__ SurfaceInfo sinfo;
__device__ __constant__ SurfaceInfo seedInfo;


template<class T>
struct is_invalid {
	__host__ __device__
	bool operator()(const T& t) {
		return !t.isValid();
	}
};

#ifndef STREAK_CUSTOM_GETVECTOR_IMPLEMENTATION_INCLUDE
__device__
float3 getVector(float3 pos) {
	pos = (pos - bottomleft)*rscale;
	if(pos.x < 0.f || pos.y < 0.f || pos.z < 0.f || pos.x > 1.f || pos.y > 1.f || pos.z > 1.f)
		return make_float3(0.f);
	float4 vec = tex3D(field, pos.x, pos.y, pos.z);
	return make_float3(vec);
}
#else
STREAK_CUSTOM_GETVECTOR_IMPLEMENTATION_INCLUDE
#endif

__device__ float3 safeNormalize(float3 v) {
	float l = dot(v,v);
	if(l > 0.f) {
		return v*rsqrt(l);
	}
	return make_float3(0.f);
}
__device__ 
float3 step(float3 pos) {
	float3 next, k1, k2, k3, k4;
	k1 = getVector(pos);
	k1 = safeNormalize(k1)*stepsize;

	k2 = getVector(pos + k1*0.5f);
	k2 = safeNormalize(k2)*stepsize;

	k3 = getVector(pos + k2*0.5f);
	k3 = safeNormalize(k3)*stepsize;

	k4 = getVector(pos + k3);
	k4 = safeNormalize(k4)*stepsize;


	next = (k1 + 2.f*k2 + 2.f*k3 + k4)/6.f;
	//next = safeNormalize(next);
	next = pos + next;

	return next;
}

__global__
void advectPoints(int numPoints, Point* points) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= numPoints)
		return;
	
	Point point = points[loc];
	if(!point.isValid())
		return;

	point.setLocation(step(point.location()), true); //note: if the point is stuck in place it gets flagged in here
	point.setNormal(make_float3(0));
	if(point.isValid())
		point.texposY *= 0.5f;
	points[loc] = point;
}

void primeSeeds(int numSeeds, Point* points) {
	dim3 grid;
	dim3 threads;
	threads.x = 512;
	threads.y = 1;
	threads.z = 1;
	
	grid.x = (numSeeds + 511)/512;
	grid.y = 1;
	grid.z = 1;

	advectPoints<<<grid, threads>>>(numSeeds, points);
}

__global__
void plantPoints(int numPoints, Point* points, Point* seeds) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= seedInfo.numPoints)
		return;

	Point point = seeds[loc];
	points[loc + numPoints] = point;
}

__global__
void plantEdges(int numEdges, int prevPointsLoc, int newSeedLoc, Edge* edges, Edge* seeds) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= seedInfo.numEdges)
		return;

	Edge edge = seeds[loc];
	edge.p1 = edge.p1 < 0 ? prevPointsLoc - edge.p1 - 1 : edge.p1 + newSeedLoc;
	edge.p2 = edge.p2 < 0 ? prevPointsLoc - edge.p2 - 1 : edge.p2 + newSeedLoc;
	edge.id = numEdges + loc;
	edge.newEdge = -1;
	edges[numEdges + loc] = edge;
}

__global__
void plantTriangles(int numTriangles, int edgeOffset, int prevEdgesLoc, Triangle* triangles, Triangle* seeds) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= seedInfo.numTriangles)
		return;

	Triangle triangle = seeds[loc];

	triangle.e1 = triangle.e1 < 0 ? prevEdgesLoc - triangle.e1 - 1 : triangle.e1 + edgeOffset;
	triangle.e2 = triangle.e2 < 0 ? prevEdgesLoc - triangle.e2 - 1 : triangle.e2 + edgeOffset;
	triangle.e3 = triangle.e3 < 0 ? prevEdgesLoc - triangle.e3 - 1 : triangle.e3 + edgeOffset;

	triangles[numTriangles + loc] = triangle;
}

__global__
void preReassignPoints(int numPoints, Point* pointsIn, int* pointsOut) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);
	if(loc >= numPoints)
		return;

	Point point = pointsIn[loc];
	if(!point.isReassigned()) {
		pointsOut[loc] = loc;
		return;
	}

	int id;
	do { //not sure if theres a big possibility for a reassignment chain, probably not
		id = point.reassignId;
		point = pointsIn[id];
	} while(point.isReassigned());

	pointsOut[loc] = id;
}

__global__
void reassignPoints(int numPoints, int* reassignPointIds, int* newPointIds) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);
	if(loc >= numPoints)
		return;

	int point = reassignPointIds[loc];
	if(point != loc) { //all chains should have been resolved in preReassign
		int reassignId = newPointIds[point];
		newPointIds[loc] = reassignId;
	}
}

__global__
void preCompactPoints(int numPoints, Point* pointsIn, int* pointsOut) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);
	if(loc >= numPoints)
		return;

	Point point = pointsIn[loc];
	pointsOut[loc] = point.isValid() ? loc : -1;
}

__global__
void checkEdges(int numEdges, Edge* edges, Point* points) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);
	if(loc >= numEdges) {
		return;
	}

	Edge edge = edges[loc];
	if(!edge.isValid())
		return;

	Point p1, p2;
	p1 = points[edge.p1];
	p2 = points[edge.p2];

	if(p1.isReassigned() || p2.isReassigned()) { //act as if all is well
		edge.newEdge = -1;
		edge.id = loc;
		edges[loc] = edge;
		return;
	}

	if(!p1.isValid() || !p2.isValid()) { //one point has died, kill this edge
		edges[loc].id = -1;
		return;
	}

	float3 offset = p2.location() - p1.location();
	float l = dot(offset, offset);
	
	/*
	if(l < maxdistance*0.0625) { //kill this edge and reassign the point
		int livepoint = edge.p1 < edge.p2 ? edge.p1 : edge.p2;
		int deadpoint = edge.p1 < edge.p2 ? edge.p2 : edge.p1;
		edges[loc].id = -1;
		points[deadpoint].reassignId = livepoint;
		points[deadpoint].texposX = -1.f;
		return;
	}*/
	
	if(true) {
		edge.newEdge = -1;
		edge.id = loc;
		edges[loc] = edge;
		//printf("passed [%d:%d:%d:%d] (%d), %d-%d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, edge.id, edge.p1, edge.p2);
		return;
	}

	Point newPoint;
	newPoint.setLocation(0.5f*p1.location() + 0.5f*p2.location());
	newPoint.setTexpos(0.5f*p1.texpos() + 0.5f*p2.texpos());
	//newPoint.texposY = 1.f;
	newPoint.setNormal(make_float3(0));

	int newPointLoc = atomicAdd(&sinfo.numPoints, 1);
	int newEdgeLoc = atomicAdd(&sinfo.numEdges, 1);


	//cuprintf("%d: %f %f %f\n", newPointLoc, newPoint.location.x, newPoint.location.y, newPoint.location.z);

	Edge newEdge = edge;
	
	newEdge.p1 = newPointLoc;
	newEdge.p2 = edge.p2;

	newEdge.id = newEdgeLoc;
	newEdge.newEdge = -1;
	edges[newEdgeLoc] = newEdge;
	//printf("edge (%d):(%d) (%d) !%d! (%d)\n", loc, newEdge.id, edge.p1, newEdge.p1, newEdge.p2);

	newEdge.p1 = edge.p1;
	newEdge.p2 = newPointLoc;

	newEdge.newEdge = newEdgeLoc;
	newEdge.id = loc;
	edges[loc] = newEdge;

	points[newPointLoc] = newPoint;
}

__global__
void preCompactEdges(int numEdges, Edge* edges, int* edgesOut) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);
	if(loc >= numEdges) {
		return;
	}

	Edge edge = edges[loc];
	edgesOut[loc] = edge.isValid() ? loc : -1;
}

__global__
void reassignEdgePoints(int numEdges, Edge* edges, int* newPointIds) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);
	if(loc >= numEdges) {
		return;
	}
	Edge edge = edges[loc];
	edge.id = loc;
	edge.p1 = newPointIds[edge.p1];
	edge.p2 = newPointIds[edge.p2];
	edge.newEdge = -1;

	edges[loc] = edge;
}

__global__
void reassignTriangles(int numTriangles, Triangle* triangles, int* newEdgeIds) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);
	if(loc >= numTriangles) {
		return;
	}

	Triangle tri = triangles[loc];
	tri.e1 = newEdgeIds[tri.e1];
	tri.e2 = newEdgeIds[tri.e2];
	tri.e3 = newEdgeIds[tri.e3];

	triangles[loc] = tri;
}

__global__
void invertReassignmentArray(int count, int* oldId, int* newId) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);
	if(loc >= count) {
		return;
	}
	int old = oldId[loc];
	newId[old] = loc;
}

__device__
void swap(int& v1, int& v2) {
	v1 ^= v2;
	v2 ^= v1;
	v1 ^= v2;
}

__device__
void flipCheck(Edge& e1, Edge& e2, Edge& e3, Edge& e4) {
	if(e1.newEdge != -1) {
		if(e3.newEdge != -1) {
			if(e1.p1 == e3.p1 || e1.p1 == e4.p2) {
				swap(e1.p1, e2.p2);
				swap(e1.id, e2.id);
			}
		} else {
			if(e1.p1 == e3.p1 || e1.p1 == e3.p2) {
				swap(e1.p1, e2.p2);
				swap(e1.id, e2.id);
			}
		}
	} 
}

__device__
void directionCheck(Edge& e1, Edge& e2) {
	if(e1.p1 == e2.p1 || e1.p1 == e2.p2) {
		swap(e1.p1, e1.p2);
	}
}

__device__
int swizzleEdges(int edgeMask, Edge& e1, Edge& e2, Edge& e3, Edge& e4, Edge& e5, Edge& e6, Edge* edges) {
	if(edgeMask == 2 || edgeMask == 6) {
		e4 = e1;
		e1 = e2;
		e2 = e3;
		e3 = e4;
	} else if(edgeMask == 4 || edgeMask == 5) {
		e4 = e3;
		e3 = e2;
		e2 = e1;
		e1 = e4;
	}
	e6 = e5 = e3;
	e3 = e4 = e2;
	e2 = e1;
	
	//gather new edges
	e2 = edges[e1.newEdge];
	e2.newEdge = -1;
	int newEdges = 1;

	if(e3.newEdge != -1) {
		e4 = edges[e3.newEdge];
		e4.newEdge = -1;
		++newEdges;
	}
	if(e5.newEdge != -1) {
		e6 = edges[e5.newEdge];
		e6.newEdge = -1;
		++newEdges;
	}
	
	flipCheck(e1, e2, e3, e4);
	flipCheck(e3, e4, e5, e6);
	flipCheck(e5, e6, e1, e2);

	//collapse
	if(e3.newEdge == -1) {
		e4 = e5;
		e5 = e6;
	} 
	if(e1.newEdge == -1) {
		e2 = e3;
		e3 = e4;
		e4 = e5;
		e5 = e6;
	}

	directionCheck(e1, e2);
	directionCheck(e2, e3);
	directionCheck(e3, e4);
	
	if(newEdges > 1)
		directionCheck(e4, e5);
	else
		directionCheck(e4, e1);

	if(newEdges > 2) {
		directionCheck(e5, e6);
		directionCheck(e6, e1);
	}
	else
		directionCheck(e5, e1);

	/*
	printf("new: %d, %d-%d (%d), %d-%d (%d), %d-%d (%d), %d-%d (%d), %d-%d (%d), %d-%d (%d)\n", newEdges,
		e1.p1, e1.p2, e1.newEdge, 
		e2.p1, e2.p2, e2.newEdge, 
		e3.p1, e3.p2, e3.newEdge,
		e4.p1, e4.p2, e4.newEdge, 
		e5.p1, e5.p2, e5.newEdge, 
		e6.p1, e6.p2, e6.newEdge);
	*/
	return newEdges;
}

__global__
void buildEdges(int numEdges, Edge* edges, uint2* indices) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= numEdges)
		return;
	
	Edge edge = edges[loc];
	indices[loc] = make_uint2(edge.p1, edge.p2);
}

__global__
void buildTriangles(int numTriangles, Triangle* triangles, Edge* edges, uint3* indices) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= numTriangles)
		return;
	
	Triangle triangle = triangles[loc];
	if(!triangle.isValid())
		return;

	Edge e1, e2, e3;
	e1 = edges[triangle.e1];
	e2 = edges[triangle.e2];
	e3 = edges[triangle.e3];

	uint3 t1;
	
	t1.x = (e1.p1 == e2.p1 || e1.p1 == e2.p2) ? e1.p2 : e1.p1;
	t1.y = (e1.p1 == e2.p1 || e1.p1 == e2.p2) ? e1.p1 : e1.p2;
	t1.z = (e2.p1 == e1.p1 || e2.p1 == e1.p2) ? e2.p2 : e2.p1;
	indices[loc] = t1;
}

__global__
void checkTriangles(int numTriangles, Triangle* triangles, Edge* edges, Point* points) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= numTriangles)
		return;
	
	Triangle triangle = triangles[loc];
	if(!triangle.isValid())
		return;
	Edge e1, e2, e3, e4, e5, e6;
	Edge ne1, ne2, ne3;
	Triangle nt1, nt2, nt3;
	e1 = edges[triangle.e1];
	e2 = edges[triangle.e2];
	e3 = edges[triangle.e3];
	
	if(!e1.isValid() || !e2.isValid() || !e3.isValid()) { //kill this triangle
		triangle.e1 = 0;
		triangle.e2 = 0;
		triangle.e3 = 0;
		triangles[loc] = triangle;
		return;
	}


	int edgeMask = (e1.newEdge != -1 ? 1 : 0) | (e2.newEdge != -1 ? 2 : 0) | (e3.newEdge != -1 ? 4 : 0);
	
	if(!edgeMask) {
		return;
	}
	
	int newEdges = swizzleEdges(edgeMask, e1, e2, e3, e4, e5, e6, edges);
	int newTriLoc = atomicAdd(&sinfo.numTriangles, newEdges);
	int newEdgeLoc = atomicAdd(&sinfo.numEdges, newEdges);
	if(newEdges == 1) {
		ne1.p1 = e2.p1;
		ne1.p2 = e3.p2;
		ne1.id = newEdgeLoc;
		ne1.newEdge = -1;

		triangle.e1 = e1.id;
		triangle.e2 = ne1.id;
		triangle.e3 = e4.id;
		
		nt1.e1 = ne1.id;
		nt1.e2 = e2.id;
		nt1.e3 = e3.id;
	} else if(newEdges == 2) {
		ne1.p1 = e1.p2;
		ne1.p2 = e3.p2;
		ne1.id = newEdgeLoc;
		ne1.newEdge = -1;
		
		Point p1, p2, p3, p4;
		p1 = points[e1.p2];
		p2 = points[e3.p2];
		p3 = points[e1.p1];
		p4 = points[e5.p1];

		float3 r1, r2;
		r1 = p1.location() - p4.location();
		r2 = p2.location() - p3.location();

		float l1 = dot(r1, r1);
		float l2 = dot(r2, r2);

		ne2.p1 = l1 > l2 ? e3.p2 : e1.p2;
		ne2.p2 = l1 > l2 ? e1.p1 : e5.p1;
		ne2.id = newEdgeLoc + 1;
		ne2.newEdge = -1;

		triangle.e1 = e1.id;
		triangle.e2 = l1 > l2 ? ne1.id : ne2.id;
		triangle.e3 = l1 > l2 ? ne2.id : e5.id;

		nt1.e1 = e2.id;
		nt1.e2 = e3.id;
		nt1.e3 = ne1.id;

		nt2.e1 = e4.id;
		nt2.e2 = l1 > l2 ? e5.id : ne2.id;
		nt2.e3 = l1 > l2 ? ne2.id : ne1.id;
	} else {
		ne1.p1 = e1.p2;
		ne1.p2 = e3.p2;
		ne1.id = newEdgeLoc;
		ne1.newEdge = -1;

		ne2.p1 = e3.p2;
		ne2.p2 = e5.p2;
		ne2.id = newEdgeLoc + 1;
		ne2.newEdge = -1;

		ne3.p1 = e5.p2;
		ne3.p2 = e1.p2;
		ne3.id = newEdgeLoc + 2;
		ne3.newEdge = -1;
		
		triangle.e1 = e1.id;
		triangle.e2 = ne3.id;
		triangle.e3 = e6.id;

		nt1.e1 = e2.id;
		nt1.e2 = e3.id;
		nt1.e3 = ne1.id;

		nt2.e1 = e4.id;
		nt2.e2 = e5.id;
		nt2.e3 = ne2.id;

		nt3.e1 = ne1.id;
		nt3.e2 = ne2.id;
		nt3.e3 = ne3.id;
	}

	triangles[loc] = triangle;

	triangles[newTriLoc] = nt1;

	edges[newEdgeLoc] = ne1;
	
	if(newEdges > 1) {
		triangles[newTriLoc + 1] = nt2;
		edges[newEdgeLoc + 1] = ne2;
	}
	
	if(newEdges > 2) {
		triangles[newTriLoc + 2] = nt3;
		edges[newEdgeLoc + 2] = ne3;
	}
}

__device__
float3 getCross(float3 p1, float3 p2, float3 p3) {
	return cross(p1 - p2, p3 - p2);
}

__device__
float3 atomicAddNormal(Point* ptr, float3 val) {
	float3 ret;
	ret.x = atomicAdd(&ptr->normalX, val.x);
	ret.y = atomicAdd(&ptr->normalY, val.y);
	ret.z = atomicAdd(&ptr->normalZ, val.z);
	return ret;
}

__global__
void calculateNormals(int numTriangles, Point* points, uint3* indices) {
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= numTriangles)
		return;
	
	uint3 triangle = indices[loc];
	Point p1, p2, p3;
	p1 = points[triangle.x];
	p2 = points[triangle.y];
	p3 = points[triangle.z];

	float3 normal;
	normal = getCross(p1.location(), p2.location(), p3.location());
	//normal = safeNormalize(normal);
	atomicAddNormal(&points[triangle.x], normal);
	atomicAddNormal(&points[triangle.y], normal);
	atomicAddNormal(&points[triangle.z], normal);

}

__global__
void normalizeNormals(int numPoints, Point* points) {
	
	int loc = threadIdx.x + blockDim.x*(blockDim.y*(blockIdx.y*gridDim.x + blockIdx.x) + threadIdx.y);

	if(loc >= numPoints)
		return;

	float3 normal = points[loc].normal();
	normal = safeNormalize(normal);
	points[loc].setNormal(normal);
}

void fillEdges(int numEdges, Edge* edges, uint2* indices) {
	
	dim3 grid;
	dim3 threads;
	threads.x = 512;
	threads.y = 1;
	threads.z = 1;
	
	grid.x = (numEdges + 511)/512;
	grid.y = 1;
	grid.z = 1;

	buildEdges<<<grid, threads>>>(numEdges, edges, indices);
	
}

void advectAndRefine(SurfaceInfo& info, Point* points, Edge* edges, Triangle* triangles) {
	//SurfaceInfo pointsRemoved;// = removeDead(info, points, edges, triangles);

	dim3 grid;
	dim3 threads;
	threads.x = 512;
	threads.y = 1;
	threads.z = 1;
	
	grid.x = (info.numPoints + 511)/512;
	grid.y = 1;
	grid.z = 1;
	
	advectPoints<<<grid, threads>>>(info.numPoints, points);
	//int oldPoints = info.numPoints;
	//info.numPoints += numSeeds;
	cudaMemcpyToSymbol("sinfo", &info, sizeof(SurfaceInfo));
	grid.x = (info.numEdges + 511)/512;
	checkEdges<<<grid, threads>>>(info.numEdges, edges, points);
	grid.x = (info.numTriangles + 511)/512;
	checkTriangles<<<grid, threads>>>(info.numTriangles, triangles, edges, points);
	cudaMemcpyFromSymbol(&info, "sinfo", sizeof(SurfaceInfo));

	printf("%d %d %d\n", info.numEdges, info.numPoints, info.numTriangles);

	//return pointsRemoved;
	
}

void buildTriangles(SurfaceInfo& info, Point* points, uint3* indices, Edge* edges, Triangle* triangles) {
	dim3 grid;
	dim3 threads;
	threads.x = 512;
	threads.y = 1;
	threads.z = 1;
	
	grid.x = (info.numTriangles + 511)/512;
	grid.y = 1;
	grid.z = 1;

	buildTriangles<<<grid, threads>>>(info.numTriangles, triangles, edges, indices);
	calculateNormals<<<grid, threads>>>(info.numTriangles, points, indices);
	grid.x = (info.numPoints + 511)/512;
	normalizeNormals<<<grid, threads>>>(info.numPoints, points);
}

SurfaceInfo removeDead(SurfaceInfo& info, Point* points, Edge* edges, Triangle* triangles) {
	thrust::device_ptr<int> oldPointIds = thrust::device_malloc<int>(info.numPoints);
	thrust::device_ptr<int> newPointIds = thrust::device_malloc<int>(info.numPoints);
	thrust::device_ptr<int> reassignPointIds = thrust::device_malloc<int>(info.numPoints);
	thrust::device_ptr<int> oldEdgeIds = thrust::device_malloc<int>(info.numEdges);
	thrust::device_ptr<int> newEdgeIds = thrust::device_malloc<int>(info.numEdges);

	dim3 grid;
	dim3 threads;
	threads.x = 512;
	threads.y = 1;
	threads.z = 1;
	
	grid.x = (info.numPoints + 511)/512;
	grid.y = 1;
	grid.z = 1;

	thrust::device_ptr<Point> pointsdPtr(points);
	thrust::device_ptr<Edge> edgesdPtr(edges);
	thrust::device_ptr<Triangle> trianglesdPtr(triangles);

	preReassignPoints<<<grid, threads>>>(info.numPoints, points, reassignPointIds.get());
	preCompactPoints<<<grid, threads>>>(info.numPoints, points, oldPointIds.get());
	grid.x = (info.numEdges + 511)/512;
	preCompactEdges<<<grid, threads>>>(info.numEdges, edges, oldEdgeIds.get());

	
	//std::vector<int> test(info.numEdges);
	//cudaMemcpy(test.data(), oldPointIds.get(), info.numEdges*sizeof(int), cudaMemcpyDeviceToHost);

	auto newPointEnd = thrust::remove_if(pointsdPtr, pointsdPtr + info.numPoints, is_invalid<Point>());
	auto newEdgeEnd = thrust::remove_if(edgesdPtr, edgesdPtr + info.numEdges, is_invalid<Edge>());
	auto newTriangleEnd = thrust::remove_if(trianglesdPtr, trianglesdPtr + info.numTriangles, is_invalid<Triangle>());


	int pointsRemoved = (int)((oldPointIds + info.numPoints) - thrust::remove(oldPointIds, oldPointIds + info.numPoints, -1));
	int edgesRemoved = (int)((oldEdgeIds + info.numEdges) - thrust::remove(oldEdgeIds, oldEdgeIds + info.numEdges, -1));
	int trianglesRemoved = (int)((trianglesdPtr + info.numTriangles) - newTriangleEnd);

	printf("edges removed: %d, points removed: %d, triangles removed: %d\n", edgesRemoved, pointsRemoved, trianglesRemoved);
	
	int oldPoints = info.numPoints;
	info.numPoints = (int)(newPointEnd - pointsdPtr);
	info.numEdges = (int)(newEdgeEnd - edgesdPtr);
	info.numTriangles = (int)(newTriangleEnd - trianglesdPtr);


	grid.x = (info.numPoints + 511)/512;
	invertReassignmentArray<<<grid, threads>>>(info.numPoints, oldPointIds.get(), newPointIds.get());
	grid.x = (oldPoints + 511)/512;
	reassignPoints<<<grid, threads>>>(oldPoints, reassignPointIds.get(), newPointIds.get());

	grid.x = (info.numEdges + 511)/512;
	invertReassignmentArray<<<grid, threads>>>(info.numEdges, oldEdgeIds.get(), newEdgeIds.get());
	reassignEdgePoints<<<grid, threads>>>(info.numEdges, edges, newPointIds.get());

	grid.x = (info.numTriangles + 511)/512;
	reassignTriangles<<<grid, threads>>>(info.numTriangles, triangles, newEdgeIds.get());

	cudaMemcpyToSymbol("sinfo", &info, sizeof(SurfaceInfo));

	thrust::device_free(oldPointIds);
	thrust::device_free(newPointIds);
	thrust::device_free(reassignPointIds);
	thrust::device_free(oldEdgeIds);
	thrust::device_free(newEdgeIds);

	SurfaceInfo removed;
	removed.numPoints = pointsRemoved;
	removed.numEdges = edgesRemoved;
	removed.numTriangles = 0;

	return removed;
}

void plantSeeds(const SurfaceInfo& seedInfo, SurfaceInfo& info, 
	Point* points, Edge* edges, Triangle* triangles, 
	Point* seedPoints, Edge* seedEdges, Triangle* seedTriangles, SurfaceInfo& prevSeedLoc) {
	
	dim3 grid;
	dim3 threads;
	threads.x = 512;
	threads.y = 1;
	threads.z = 1;
	
	grid.x = (seedInfo.numPoints + 511)/512;
	grid.y = 1;
	grid.z = 1;

	plantPoints<<<grid, threads>>>(info.numPoints, points, seedPoints);
	
	grid.x = (seedInfo.numEdges + 511)/512;
	plantEdges<<<grid, threads>>>(info.numEdges, prevSeedLoc.numPoints, info.numPoints, edges, seedEdges);
	
	grid.x = (seedInfo.numTriangles + 511)/512;
	plantTriangles<<<grid, threads>>>(info.numTriangles, info.numEdges, prevSeedLoc.numEdges, triangles, seedTriangles);
	
	prevSeedLoc.numPoints = info.numPoints;
	prevSeedLoc.numEdges = info.numEdges;
	info.numPoints += seedInfo.numPoints;
	info.numEdges += seedInfo.numEdges;
	info.numTriangles += seedInfo.numTriangles;

	cudaMemcpyToSymbol("sinfo", &info, sizeof(SurfaceInfo));

}

void initTex() {
	field.addressMode[0] = cudaAddressModeClamp;
	field.addressMode[1] = cudaAddressModeClamp;
	field.addressMode[2] = cudaAddressModeClamp;

	field.normalized = true;
	field.filterMode = cudaFilterModeLinear;
}
