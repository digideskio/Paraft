#ifdef _CUDA
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_math.h>
#include "streaksurface.h"
#include "streaksurface.cuh"
#include "glbuffers.h"

#include <unordered_set>
#include <cuda_gl_interop.h>

#define CUDAERROR(x) x; for(cudaError_t error = cudaGetLastError(); error != cudaSuccess; error = cudaGetLastError()) printError(error, __LINE__, __FILE__)
void printError(cudaError_t error, int line, const char* file) {
	qDebug("cuda error in %s(%d): %s", file, line, cudaGetErrorString(error));
}

#define minf(a,b) ((a) < (b) ? (a) : (b))

template<>
void CudaFreeFunctor::operator()<cudaGraphicsResource>(cudaGraphicsResource* t) const {
	if(t)
		cudaGraphicsUnregisterResource(t);
}
template<>
void CudaFreeFunctor::operator()<cudaArray>(cudaArray* t) const {
	if(t)
		cudaFreeArray(t);
}

template<typename T>
inline static void resizeData(std::shared_ptr<T>& data, size_t oldSize, size_t newSize) {
	T* temp;
	cudaMalloc(&temp, newSize*sizeof(T));
	cudaMemcpy(temp, data.get(), oldSize*sizeof(T), cudaMemcpyDeviceToDevice);
	data.reset(temp, CudaFreeFunctor());
}

template<typename T>
inline static void resizeData(std::shared_ptr<T>& data, size_t newSize) {
	T* temp;
	cudaMalloc(&temp, newSize*sizeof(T));
	data.reset(temp, CudaFreeFunctor());
}

template<typename T>
inline static void copyFromVec(std::shared_ptr<T>& data, const std::vector<T>& vec, size_t factor=1) {
	T* temp;
	cudaMalloc(&temp, vec.size()*sizeof(T)*factor);
	cudaMemcpy(temp, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice);
	data.reset(temp, CudaFreeFunctor());
}

void StreakSurfaceGen::setGeometry(
		const std::vector<Point>& pointsVec, 
		const std::vector<Edge>& edgesVec, 
		const std::vector<Triangle>& trianglesVec,
		int defaultTriangles, int defaultEdges) {
	numEdges = defaultEdges > 0 ? defaultEdges : (int)edgesVec.size();
	numTriangles = defaultTriangles > 0 ? defaultTriangles : (int)trianglesVec.size();
	numPoints = (int)pointsVec.size();

	info.numEdges = numEdges;
	info.numTriangles = numTriangles;
	info.numPoints = numPoints;

	if(!numTriangles) { //setting initial buffer size
		numTriangles = defaultTriangles;
	}
	void* temp;

	copyFromVec(edges, edgesVec, 16);
	if(trianglesVec.size())
		copyFromVec(triangles, trianglesVec, 4);
	else
		resizeData(triangles, numTriangles*4);

	ibo.reset(new GLIndexbuffer(GL_TRIANGLES, GL_DYNAMIC_COPY));
	ibo->bind();
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numTriangles*12*4, 0, GL_DYNAMIC_COPY);
	ibo->release();
	ibo2.reset(new GLIndexbuffer(GL_TRIANGLES, GL_DYNAMIC_COPY));
	ibo2->bind();
	ibo2->release();

	vbo.reset(new GLVertexbufferf(GL_POINTS, GL_DYNAMIC_COPY));
	vbo->setHasVertices(false);
	vbo->addAttribute("vVertex", 3, 32);
	vbo->addAttribute("vTexpos", 2, 32, 12);
	vbo->addAttribute("vNormal", 3, 32, 20);
	vbo2.reset(new GLVertexbufferf(GL_POINTS, GL_DYNAMIC_COPY));
	vbo2->setHasVertices(false);
	vbo2->addAttribute("vVertex", 3, 32);
	vbo2->addAttribute("vTexpos", 2, 32, 12);
	vbo2->addAttribute("vNormal", 3, 32, 20);

	
	vbo->bind();

	glBufferData(GL_ARRAY_BUFFER, numPoints*4*sizeof(Point), 0, GL_DYNAMIC_COPY);
	glBufferSubData(GL_ARRAY_BUFFER, 0, numPoints*sizeof(Point), (const GLvoid*)pointsVec.data());

	vbo->release();
	vbo2->bind();
	vbo2->release();
	
	cudaMemcpyToSymbol("sinfo", &info, sizeof(SurfaceInfo));
	//cudaMemcpyToSymbol("numSeeds", &numSeeds, sizeof(int));
	CUDAERROR(cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&temp, vbo->handle(), cudaGraphicsRegisterFlagsNone));
	buffers[0].reset((cudaGraphicsResource*)temp, CudaFreeFunctor());
	CUDAERROR(cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&temp, ibo->handle(), cudaGraphicsRegisterFlagsNone));
	buffers[1].reset((cudaGraphicsResource*)temp, CudaFreeFunctor());
	bufptrs[0] = buffers[0].get();
	bufptrs[1] = buffers[1].get();

	m_state |= GeometrySet;
}

void StreakSurfaceGen::setSeeds(
		const std::vector<Point>& seedPointsVec,
		const std::vector<Edge>& seedEdgesVec,
		const std::vector<Triangle>& seedTrianglesVec,
		const std::vector<Edge>& initialEdges) {
	numSeeds = (int)seedPointsVec.size();

	seedInfo.numPoints = (unsigned int)numSeeds;
	seedInfo.numTriangles = (unsigned int)seedTrianglesVec.size();
	seedInfo.numEdges = (unsigned int)seedEdgesVec.size();

	cudaMemcpyToSymbol("seedInfo", &seedInfo, sizeof(SurfaceInfo));

	copyFromVec(seedPoints, seedPointsVec);
	copyFromVec(seedEdges, seedEdgesVec);
	copyFromVec(seedTriangles, seedTrianglesVec);

	setGeometry(seedPointsVec, initialEdges, std::vector<Triangle>(), (int)seedTrianglesVec.size(), (int)initialEdges.size()*4);
	primed = false;
}

void StreakSurfaceGen::step() {
	if(m_state ^ AllSet) {
		qDebug("Not properly setup! %x, (%x)", m_state, m_state ^ AllSet);
		return;
	}
	CUDAERROR(cudaGraphicsMapResources(2, bufptrs));
	Point* points;
	uint3* indices;
	size_t size;
	CUDAERROR(cudaGraphicsResourceGetMappedPointer((void**)&points, &size, buffers[0].get()));
	CUDAERROR(cudaGraphicsResourceGetMappedPointer((void**)&indices, &size, buffers[1].get()));

	if(!primed) {
		primeSeeds(numSeeds, points);
		primed = true;
		//prevSeedLoc = numSeeds;
	}
	if(numSeeds) {
		plantSeeds(seedInfo, info, points, edges.get(), triangles.get(), seedPoints.get(), 
			seedEdges.get(), seedTriangles.get(), prevSeedInfo);
	}
	SurfaceInfo removedInfo(0,0,0);
	advectAndRefine(info, points, edges.get(), triangles.get());
	removedInfo = removeDead(info, points, edges.get(), triangles.get());
	buildTriangles(info, points, indices, edges.get(), triangles.get());
	if(numSeeds) {
		prevSeedInfo -= removedInfo;
	}
	//CUDAERROR(cudaMemcpyFromSymbol(&info, "sinfo", sizeof(SurfaceInfo)));
	qDebug("points: %d, edges: %d, triangles: %d", info.numPoints, info.numEdges, info.numTriangles);
	CUDAERROR(cudaGraphicsUnmapResources(2, bufptrs));
	bool resized = false;
	if((int)info.numPoints > 2*numPoints) {
		buffers[0].reset();
		//CUDAERROR(cudaGraphicsUnregisterResource(buffers[2]));
		qDebug("resizing point buffer: %d (%d)", info.numPoints, 4*numPoints);
		glBindBuffer(GL_COPY_WRITE_BUFFER, vbo2->handle());
		glBindBuffer(GL_COPY_READ_BUFFER, vbo->handle());
		glBufferData(GL_COPY_WRITE_BUFFER, info.numPoints*4*sizeof(Point), 0, GL_DYNAMIC_COPY);
		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, info.numPoints*sizeof(Point));
		glBufferData(GL_COPY_READ_BUFFER, 0, 0, GL_DYNAMIC_COPY);
		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
		glBindBuffer(GL_COPY_READ_BUFFER, 0);
		numPoints = info.numPoints;
		std::swap(vbo, vbo2);
		cudaGraphicsResource* temp;
		CUDAERROR(cudaGraphicsGLRegisterBuffer(&temp, vbo->handle(), cudaGraphicsRegisterFlagsNone));
		buffers[0].reset(temp, CudaFreeFunctor());
		bufptrs[0] = buffers[0].get();
		//CUDAERROR(cudaGraphicsGLRegisterBuffer(&buffers[2], vbo2->handle(), cudaGLMapFlagsNone));
		resized = true;
	}

	if((int)info.numEdges > 2*numEdges) {
		qDebug("resizing edge buffer: %d (%d)", info.numEdges, 4*numEdges);
		resizeData(edges, info.numEdges, info.numEdges*4);
		numEdges = info.numEdges;
		if(eibo) {
			eibo.reset();
		}
		resized = true;
	}
	

	if((int)info.numTriangles > 1.5f*numTriangles) {
		buffers[1].reset();
		qDebug("resizing triangle buffer: %d (%d)", info.numTriangles, 4*numTriangles);
		resizeData(triangles, info.numTriangles, info.numTriangles*4);
		glBindBuffer(GL_COPY_READ_BUFFER, ibo->handle());
		glBindBuffer(GL_COPY_WRITE_BUFFER, ibo2->handle());
		glBufferData(GL_COPY_WRITE_BUFFER, 4*info.numTriangles*12, 0, GL_DYNAMIC_COPY);
		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, info.numTriangles*12);
		glBufferData(GL_COPY_READ_BUFFER, 0, 0, GL_DYNAMIC_COPY);
		glBindBuffer(GL_COPY_READ_BUFFER, 0);
		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
		std::swap(ibo, ibo2);
		glFinish();
		numTriangles = info.numTriangles;
		cudaGraphicsResource* temp;
		CUDAERROR(cudaGraphicsGLRegisterBuffer(&temp, ibo->handle(), cudaGraphicsRegisterFlagsNone));
		buffers[1].reset(temp, CudaFreeFunctor());
		bufptrs[1] = buffers[1].get();
		resized = true;
	}

	//printf("test");

}

void StreakSurfaceGen::draw(bool pointsOnly) {
	if(!vbo)
		return;
	vbo->bind();
	if(!pointsOnly) {
		ibo->bind();
		glDrawElements(GL_TRIANGLES, info.numTriangles*3, GL_UNSIGNED_INT, 0);
		ibo->release();
	} else {
		glDrawArrays(GL_POINTS, 0, info.numPoints);
	}

	vbo->release();
}

void StreakSurfaceGen::drawEdges() {
	if(m_state ^ AllSet)
		return;
	if(!eibo) {
		eibo.reset(new GLIndexbuffer(GL_LINES, GL_DYNAMIC_COPY));
		eibo->bind();
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numEdges*2*2*sizeof(int), 0, GL_DYNAMIC_COPY);
		eibo->release();
		cudaGraphicsResource* temp;
		CUDAERROR(cudaGraphicsGLRegisterBuffer(&temp, eibo->handle(), cudaGraphicsRegisterFlagsNone));
		buffers[2].reset(temp, CudaFreeFunctor());
		bufptrs[2] = buffers[2].get();
	}
	uint2* indices;
	size_t size;
	CUDAERROR(cudaGraphicsMapResources(1, &bufptrs[2]));
	CUDAERROR(cudaGraphicsResourceGetMappedPointer((void**)&indices, &size, buffers[2].get()));
	fillEdges(info.numEdges, edges.get(), indices);
	CUDAERROR(cudaGraphicsUnmapResources(1, &bufptrs[2]));

	vbo->bind();
	eibo->bind();
	glDrawElements(GL_LINES, info.numEdges*2, GL_UNSIGNED_INT, 0);
	eibo->release();
	vbo->release();
}


StreakSurfaceGen::~StreakSurfaceGen() {
	const textureReference* ref;
	cudaGetTextureReference(&ref, "field");
	cudaUnbindTexture(ref);
}

StreakSurfaceGen::StreakSurfaceGen(const std::vector<float>& vecs, int x, int y, int z, int channels,
		float stepsize, float maxdistance, Vector3 bottomleft, Vector3 topright):
	numPoints(0), primed(true), m_state(0)
{
	setBottomLeft(bottomleft);
	setTopRight(topright);

	if(stepsize <= 0) {
		stepsize = minf(pitch.x()/x, minf(pitch.y()/y, pitch.z()/z));
		stepsize *= 0.25f;
	} 
	if(maxdistance <= 0) {
		maxdistance = stepsize*2.0;
	}
	setStepSize(stepsize);
	setMaxDistance(maxdistance);
	setTexture(vecs, x, y, z, channels);

	prevSeedInfo.numEdges = 0;
	prevSeedInfo.numPoints = 0;
	prevSeedInfo.numTriangles = 0;

	CUDAERROR(initTex());
}

StreakSurfaceGen::StreakSurfaceGen():numPoints(0), prevSeedInfo(0, 0, 0), primed(true), m_state(0) {
	CUDAERROR(initTex());
}

void StreakSurfaceGen::generateTestSeeds() {
	/*
	const int rows = 2;
	const int cols = 200;
	numEdges = 3*(rows - 1)*(cols - 1) + rows + cols - 2;
	numTriangles = 2*(rows - 1)*(cols - 1);
	std::vector<Edge> tempedges(numEdges);
	std::vector<Triangle> temptris(numTriangles);
	int curEdge = 0;
	int curTriangle = 0;

	int edgesPerRow = 3*(cols - 1) + 1;
	for(int i = 0; i < rows - 1; ++i) {
		for(int j = 0; j < cols - 1; ++j) {
			int id = j + i*cols;
			tempedges[curEdge].p1 = id;
			tempedges[curEdge].p2 = id + 1;
			
			++curEdge;
				
			tempedges[curEdge].p1 = id;
			tempedges[curEdge].p2 = id + cols + 1;

			++curEdge;

			tempedges[curEdge].p1 = id;
			tempedges[curEdge].p2 = id + cols;

			++curEdge;

			if(j) {
				temptris[curTriangle].e1 = curEdge - 1;
				temptris[curTriangle].e2 = curEdge - 5;
				temptris[curTriangle].e3 = curEdge - 6;
				++curTriangle;
			}
			
			if(i) {
				temptris[curTriangle].e1 = curEdge - 3;
				temptris[curTriangle].e2 = curEdge - edgesPerRow - 1;
				temptris[curTriangle].e3 = curEdge - edgesPerRow - 2;
				++curTriangle;
			}
		}
		tempedges[curEdge].p1 = (i + 1)*cols - 1;
		tempedges[curEdge].p2 = (i + 2)*cols - 1;
		++curEdge;

		temptris[curTriangle].e1 = curEdge - 1;
		temptris[curTriangle].e2 = curEdge - 3;
		temptris[curTriangle].e3 = curEdge - 4;
		++curTriangle;
	}

	int lastRowIndex = curEdge - edgesPerRow;
	for(int j = 0; j < cols - 1; ++j) {
		tempedges[curEdge].p1 = (rows - 1)*cols + j;
		tempedges[curEdge].p2 = (rows - 1)*cols + j + 1;
		
		temptris[curTriangle].e1 = curEdge;
		temptris[curTriangle].e2 = lastRowIndex + 2;
		temptris[curTriangle].e3 = lastRowIndex + 1;

		lastRowIndex += 3;
		++curEdge;
		++curTriangle;
	}
	for(int i = 0; i < curEdge; ++i) {
		tempedges[i].id = i;
	}


	std::vector<Point> temppoints(rows*cols);

	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			temppoints[i*cols + j].setLocation(make_float3(bl.x() + pitch.x()*(j/(float)200.f),
				bl.y() + pitch.y()*(i/(float)200.f + 0.5f), bl.z() + pitch.z()*(j/(float)200.f)));
			temppoints[i*cols + j].setTexpos(make_float2(j/(float)cols, i/(float)rows));
			temppoints[i*cols + j].setNormal(make_float3(0.f));
		}
	}


	numPoints = rows*cols;

	info.numEdges = numEdges;
	info.numTriangles = numTriangles;
	info.numPoints = numPoints;

	setSeeds(200, temppoints, tempedges, temptris); */

	std::vector<Point> temppoints;
	for(int i = 0; i < 200; ++i) {
		Point p;
		Vector3 loc = (i/200.0)*pitch + bl;
		p.setLocation(make_float3(unpack3(loc)));
		p.setTexpos(make_float2(i/200.0, 1.f));
		p.setNormal(make_float3(0.f));
		temppoints.push_back(p);
	}
	setSeeds(temppoints);
}

inline static void addEdge(int& edgeId, Edge& edge, std::vector<Edge>& edgesVec, std::unordered_set<int>& edgeSet) {
	if(edgeId < 0) {
		edgeId += 3; //these edges refer to already existing edges that will not be added again
	}
	if(edgeSet.find(edgeId) == edgeSet.end()) {
		edgeSet.insert(edgeId);
		edge.id = (int)edgesVec.size();
		edgesVec.push_back(edge);
	}
	edgeId = edge.id;
}

inline static void addEdge(Edge& edge, std::vector<Edge>& edges) {
	edge.id = (int)edges.size();
	edge.newEdge = -1;
	edges.push_back(edge);
}

void StreakSurfaceGen::setSeeds(const std::vector<Point>& pointsVec, bool loops) {
	int seedCount = (int)pointsVec.size();
	std::vector<Edge> edgesVec;
	std::vector<Triangle> triVec;
	std::vector<Point> newPoints = pointsVec;
	for(int i = 0; i < (seedCount - (loops ? 0 : 1)); ++i) {
		Edge e;
		e.p1 = i;
		e.p2 = (i + 1) % seedCount;
		addEdge(e, edgesVec); // right
	} 
	std::vector<Edge> initialEdges = edgesVec;
	for(int i = 0; i < seedCount; ++i) {
		Edge e1, e2;
		e1.p1 = i;
		e1.p2 = -(i + 1);
		addEdge(e1, edgesVec); // up

		if(!loops && !((i + 1) < seedCount))
			continue;

		bool loop = (i + 1) == seedCount;

		e2.p1 = i;
		e2.p2 = -(i + 2);
		addEdge(e2, edgesVec); // diag

		Triangle t;
		t.e1 = i;
		t.e2 = loop ? seedCount : e2.id + 1;
		t.e3 = e2.id;
		triVec.push_back(t);

		t.e1 = e2.id;
		t.e2 = -(i + 1);
		t.e3 = e1.id;
		triVec.push_back(t);

	}
	for(int i = 0; i < initialEdges.size(); ++i) {
		initialEdges[i].id = i;
	}

	setSeeds(newPoints, edgesVec, triVec, initialEdges);

}

void StreakSurfaceGen::setBLRScale() {
	float3 rscale = make_float3(1.f/pitch.x(), 1.f/pitch.y(), 1.f/pitch.z());
	float3 bottomleft = make_float3(unpack3(bl));

	cudaMemcpyToSymbol("rscale", &rscale, sizeof(float3));
	cudaMemcpyToSymbol("bottomleft", &bottomleft, sizeof(float3));
}

void StreakSurfaceGen::setBottomLeft(const Vector3& v) {
	Vector3 topright = bl + pitch;
	bl = v;
	pitch = topright - bl;
	setBLRScale();
	m_state |= BottomLeftSet;
}

void StreakSurfaceGen::setTopRight(const Vector3& v) {
	pitch = v - bl;
	setBLRScale();
	m_state |= TopRightSet;
}

void StreakSurfaceGen::setStepSize(float v) {
	stepSize = v;
	cudaMemcpyToSymbol("stepsize", &stepSize, sizeof(float));
	m_state |= StepSizeSet;
}

void StreakSurfaceGen::setMaxDistance(float v) {
	maxDistance = v;
	v *= v;
	cudaMemcpyToSymbol("maxdistance", &v, sizeof(float)); //store maxDistance^2 so we dont' have to sqrt
	m_state |= MaxDistanceSet;
}

void StreakSurfaceGen::setTexture(const std::vector<float>& vecs, int width, int height, int depth, int channels) {
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
	cudaExtent extent = make_cudaExtent(width, height, depth);
	cudaArray* temp;
	CUDAERROR(cudaMalloc3DArray(&temp, &desc, extent));
	field.reset(temp, CudaFreeFunctor());
	float4* outvecs = (float4*)vecs.data();
	std::unique_ptr<float4[]> newvecs;
	if(channels == 3) {
		newvecs.reset(new float4[width*height*depth]); //vecs is packed by 3s
		for(int i = 0; i < width*height*depth; ++i) {
			newvecs[i] = make_float4(vecs[i*3], vecs[i*3 + 1], vecs[i*3 + 2], 0.f);
			float l = dot(newvecs[i],newvecs[i]);
			if(l > 0.f) {
				newvecs[i] *= rsqrtf(l);
			}
		}
		outvecs = newvecs.get();
	} else if(channels != 4) {
		qFatal("Only 3 channels and 4 channels supported!");
	}
	cudaMemcpy3DParms parms;
	memset(&parms, 0, sizeof(cudaMemcpy3DParms));
	parms.dstArray = field.get();
	parms.extent = extent;
	parms.kind = cudaMemcpyHostToDevice;
	parms.srcPtr = make_cudaPitchedPtr(outvecs, sizeof(float4)*width, width, height);

	CUDAERROR(cudaMemcpy3D(&parms));

	const textureReference* ref;
	CUDAERROR(cudaGetTextureReference(&ref, "field"));
	CUDAERROR(cudaBindTextureToArray(ref, field.get(), &desc));

	m_state |= FieldSet;
}

void StreakSurfaceGen::setTexture(std::shared_ptr<cudaArray> data) {
	field = data;
	
	cudaChannelFormatDesc desc;
	cudaGetChannelDesc(&desc, field.get());

	const textureReference* ref;
	CUDAERROR(cudaGetTextureReference(&ref, "field"));
	CUDAERROR(cudaBindTextureToArray(ref, field.get(), &desc));

	m_state |= FieldSet;
}

void StreakSurfaceGen::generateSeedLine(const Vector3& begin, const Vector3& end) {
	std::vector<Point> points;
	Vector3 dir = end - begin;
	float l = dir.length();
	dir.normalize();
	int steps = (int)floor(l/stepSize);
	points.resize(steps + 1);
	
	for(int i = 0; i <= steps; ++i) {
		Vector3 pos = begin + i*stepSize*dir;
		points[i].setLocation(make_float3(unpack3(pos)));
		points[i].setTexpos(make_float2(i/(float)steps, 0.f));
		points[i].setNormal(make_float3(0.f));
	}

	setSeeds(points);
}


void StreakSurfaceGen::save(QIODevice* ios) {
	if(!vbo)
		return;
	vbo->save(*ios, info.numPoints*sizeof(Point)/4);
	ibo->save(*ios, info.numTriangles*3);
}

#endif