#include <GL/glew.h>
#include "box.h"
#define ROOT3OVER2 0.8660254
#define EPSILON 0.0000001

#include "glbuffers.h"
#include <cmath>
#include <limits>

Vector3 rayPlaneIntersect(const Vector3& start, const Vector3& ray, const Vector3& pnormal, const double& d) {
	return start + ray*((-pnormal.dot(start) - d)/(pnormal.dot(ray)));
} 


Box::Box(const Vector3& bottomleft, const Vector3& pitch):bottomleft(bottomleft), pitch(pitch) {
	topright = bottomleft + pitch;
}

struct Line {
	Vector3 start;
	Vector3 end;
};
void Box::drawSlice(const Camera& camera, const double& sDepth) const {
	if(pitch.lengthSquared() < 0.000001)
		return;
	Vector3 x = camera.getRight();
	Vector3 y = camera.getUp();

	Vector3 center = camera.getCamPosition() - camera.getCam()*sDepth;
	Vector3 p;

	double d = camera.getCam().dot(center);
	Vector3 normal = -camera.getCam();
	Vector3 ray;
	double t1, t2;
	Line l;
	QList<Line> lines;
	QList<Vector3> points;
	Vector3 axis;
	if(fabs(fabs(normal.dot(Vector3::zAxis)) - 1.) > EPSILON) { //xy
		//first, find point along x where it intersects our slice plane

		if(fabs(normal.dot(Vector3::xAxis)) > EPSILON)
			axis = Vector3::xAxis;
		else
			axis = Vector3::yAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::zAxis;
		rayBoxIntersection(t1, t2, center, ray, 3); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center= rayPlaneIntersect(Vector3(bottomleft.x(), bottomleft.y(), topright.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 3); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	if(fabs(fabs(normal.dot(Vector3::yAxis)) - 1.) > EPSILON) { //xz

		if(fabs(normal.dot(Vector3::xAxis)) > EPSILON)
			axis = Vector3::xAxis;
		else
			axis = Vector3::zAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::yAxis;
		rayBoxIntersection(t1, t2, center, ray, 5); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center = rayPlaneIntersect(Vector3(bottomleft.x(), topright.y(), bottomleft.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 5);
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	if(fabs(fabs(normal.dot(Vector3::xAxis)) - 1.) > EPSILON) { //yz

		if(fabs(normal.dot(Vector3::yAxis)) > EPSILON)
			axis = Vector3::yAxis;
		else
			axis = Vector3::zAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::xAxis;
		rayBoxIntersection(t1, t2, center, ray, 6); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center = rayPlaneIntersect(Vector3(topright.x(), bottomleft.y(), bottomleft.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 6); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	if(lines.size() >= 3) { //not sure how it happens, but sometimes we only get two lines
		QList<Line>::iterator it = lines.begin();
		points.push_back((*it).start);
		points.push_back((*it).end);
		it = lines.erase(it);
		bool connected;
		while(lines.size()) {
			connected = false;
			for(it = lines.begin(); it != lines.end(); it++) {
				p = points.back() - (*it).start;
				if(p.length() < EPSILON) {
					points.push_back((*it).end);
					lines.erase(it);
					connected = true;
					break;
				}
				p = points.back() - (*it).end;
				if(p.length() < EPSILON) {
					points.push_back((*it).start);
					lines.erase(it);
					connected = true;
					break;
				}
			}
			if(!connected) //Not a closed polygon!
				return;
		}

		//check that slicing plane is outsize the box
		Vector3 center(0.0,0.0,0.0);
		for(QList<Vector3>::iterator it = points.begin(); it != points.end(); ++it) {
			center += (*it);
		}
		center /= points.size();
		if (center.x() < bottomleft.x() || center.y() < bottomleft.y() || center.z() < bottomleft.z() ||
		    center.x() > topright.x() || center.y() > topright.y() || center.z() > topright.z()) return;


		glBegin(GL_POLYGON);
		Vector3 tex;

		//check that it's front facing, if not reverse it
		l.start = points[0] - points[1];
		l.end = points[2] - points[1];
		l.start = l.end*l.start;

		int i = 0, j = points.size(), k = 1;
		if(!(l.start.dot(-normal) > EPSILON)) { //front facing
			i = j - 1;
			j = -1;
			k = -1;
		}

		for(; i != j; i += k) {
			tex = points[i] - bottomleft;
			glTexCoord3d(tex.x()/pitch.x(), tex.y()/pitch.y(), tex.z()/pitch.z());
			glVertex3d(unpack3(points[i]));
		}
		glEnd();
	}
}

void Box::drawSlice(const CameraOptions& co, const double& sDepth) const {
	if(pitch.lengthSquared() < 0.000001)
		return;
	Vector3 x = co.o;
	Vector3 y = co.u;

	Vector3 center = co.l + co.c * co.dist - co.c * sDepth;
	Vector3 p;

	double d = co.c.dot(center);
	Vector3 normal = -co.c;
	Vector3 ray;
	double t1, t2;
	Line l;
	QList<Line> lines;
	QList<Vector3> points;
	Vector3 axis;
	if(fabs(fabs(normal.dot(Vector3::zAxis)) - 1.) > EPSILON) { //xy
		//first, find point along x where it intersects our slice plane

		if(fabs(normal.dot(Vector3::xAxis)) > EPSILON)
			axis = Vector3::xAxis;
		else
			axis = Vector3::yAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::zAxis;
		rayBoxIntersection(t1, t2, center, ray, 3); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center= rayPlaneIntersect(Vector3(bottomleft.x(), bottomleft.y(), topright.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 3); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	if(fabs(fabs(normal.dot(Vector3::yAxis)) - 1.) > EPSILON) { //xz

		if(fabs(normal.dot(Vector3::xAxis)) > EPSILON)
			axis = Vector3::xAxis;
		else
			axis = Vector3::zAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::yAxis;
		rayBoxIntersection(t1, t2, center, ray, 5); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center = rayPlaneIntersect(Vector3(bottomleft.x(), topright.y(), bottomleft.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 5);
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	if(fabs(fabs(normal.dot(Vector3::xAxis)) - 1.) > EPSILON) { //yz

		if(fabs(normal.dot(Vector3::yAxis)) > EPSILON)
			axis = Vector3::yAxis;
		else
			axis = Vector3::zAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::xAxis;
		rayBoxIntersection(t1, t2, center, ray, 6); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center = rayPlaneIntersect(Vector3(topright.x(), bottomleft.y(), bottomleft.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 6); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	if(lines.size() >= 3) { //not sure how it happens, but sometimes we only get two lines
		QList<Line>::iterator it = lines.begin();
		points.push_back((*it).start);
		points.push_back((*it).end);
		it = lines.erase(it);
		bool connected;
		while(lines.size()) {
			connected = false;
			for(it = lines.begin(); it != lines.end(); it++) {
				p = points.back() - (*it).start;
				if(p.length() < EPSILON) {
					points.push_back((*it).end);
					lines.erase(it);
					connected = true;
					break;
				}
				p = points.back() - (*it).end;
				if(p.length() < EPSILON) {
					points.push_back((*it).start);
					lines.erase(it);
					connected = true;
					break;
				}
			}
			if(!connected) //Not a closed polygon!
				return;
		}

		//check that slicing plane is outsize the box
		Vector3 center(0.0,0.0,0.0);
		for(QList<Vector3>::iterator it = points.begin(); it != points.end(); ++it) {
			center += (*it);
		}
		center /= points.size();
		if (center.x() < bottomleft.x() || center.y() < bottomleft.y() || center.z() < bottomleft.z() ||
		    center.x() > topright.x() || center.y() > topright.y() || center.z() > topright.z()) return;


		glBegin(GL_POLYGON);
		Vector3 tex;

		//check that it's front facing, if not reverse it
		l.start = points[0] - points[1];
		l.end = points[2] - points[1];
		l.start = l.end*l.start;

		int i = 0, j = points.size(), k = 1;
		if(!(l.start.dot(-normal) > EPSILON)) { //front facing
			i = j - 1;
			j = -1;
			k = -1;
		}

		for(; i != j; i += k) {
			tex = points[i] - bottomleft;
			glTexCoord3d(tex.x()/pitch.x(), tex.y()/pitch.y(), tex.z()/pitch.z());
			glVertex3d(unpack3(points[i]));
		}
		glEnd();
	}
}

void Box::drawSlicedBox(const Vector3& sliceVec, const double& sDepth, bool sliceonly) const {
	if(pitch.lengthSquared() < 0.000001)
		return;
	Vector3 center = getCenterPoint() + sliceVec * sDepth;
	Vector3 p;

	double d = sliceVec.dot(center);
	Vector3 normal = -sliceVec;
	Vector3 ray;
	double t1, t2;
	Line l;
	QList<Line> lines;
	QList<Vector3> points;
	Vector3 axis;

	Vector3 vertex[6][4];
	vertex[0][0] = Vector3(topright.x(),  topright.y(),  topright.z());
	vertex[0][1] = Vector3(topright.x(),  bottomleft.y(),  topright.z());
	vertex[0][2] = Vector3(topright.x(),  bottomleft.y(), bottomleft.z());
	vertex[0][3] = Vector3(topright.x(),  topright.y(), bottomleft.z());
	vertex[1][0] = Vector3(bottomleft.x(),  topright.y(),  bottomleft.z());
	vertex[1][1] = Vector3(bottomleft.x(),  bottomleft.y(),  bottomleft.z());
	vertex[1][2] = Vector3(bottomleft.x(),  bottomleft.y(), topright.z());
	vertex[1][3] = Vector3(bottomleft.x(),  topright.y(), topright.z());
	vertex[2][0] = Vector3(bottomleft.x(), topright.y(), topright.z());
	vertex[2][1] = Vector3(topright.x(), topright.y(), topright.z());
	vertex[2][2] = Vector3(topright.x(), topright.y(), bottomleft.z());
	vertex[2][3] = Vector3(bottomleft.x(), topright.y(), bottomleft.z());
	vertex[3][0] = Vector3(bottomleft.x(), bottomleft.y(), bottomleft.z());
	vertex[3][1] = Vector3(topright.x(), bottomleft.y(), bottomleft.z());
	vertex[3][2] = Vector3(topright.x(), bottomleft.y(), topright.z());
	vertex[3][3] = Vector3(bottomleft.x(), bottomleft.y(), topright.z());
	vertex[4][0] = Vector3(bottomleft.x(), topright.y(), topright.z());
	vertex[4][1] = Vector3(bottomleft.x(), bottomleft.y(), topright.z());
	vertex[4][2] = Vector3(topright.x(), bottomleft.y(), topright.z());
	vertex[4][3] = Vector3(topright.x(), topright.y(), topright.z());
	vertex[5][0] = Vector3(bottomleft.x(), topright.y(), bottomleft.z());
	vertex[5][1] = Vector3(topright.x(), topright.y(), bottomleft.z());
	vertex[5][2] = Vector3(topright.x(), bottomleft.y(), bottomleft.z());
	vertex[5][3] = Vector3(bottomleft.x(), bottomleft.y(), bottomleft.z());
	bool surfaces[6];
	for (int i = 0; i < 6; ++i) surfaces[i] = false;


	if(fabs(fabs(normal.dot(Vector3::zAxis)) - 1.) > EPSILON) { //xy
		//first, find point along x where it intersects our slice plane

		if(fabs(normal.dot(Vector3::xAxis)) > EPSILON)
			axis = Vector3::xAxis;
		else
			axis = Vector3::yAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::zAxis;
		rayBoxIntersection(t1, t2, center, ray, 3); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center= rayPlaneIntersect(Vector3(bottomleft.x(), bottomleft.y(), topright.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 3); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	if(fabs(fabs(normal.dot(Vector3::yAxis)) - 1.) > EPSILON) { //xz

		if(fabs(normal.dot(Vector3::xAxis)) > EPSILON)
			axis = Vector3::xAxis;
		else
			axis = Vector3::zAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::yAxis;
		rayBoxIntersection(t1, t2, center, ray, 5); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center = rayPlaneIntersect(Vector3(bottomleft.x(), topright.y(), bottomleft.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 5);
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	if(fabs(fabs(normal.dot(Vector3::xAxis)) - 1.) > EPSILON) { //yz

		if(fabs(normal.dot(Vector3::yAxis)) > EPSILON)
			axis = Vector3::yAxis;
		else
			axis = Vector3::zAxis;

		center = rayPlaneIntersect(bottomleft, axis, normal, d);
		ray = normal*Vector3::xAxis;
		rayBoxIntersection(t1, t2, center, ray, 6); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
		center = rayPlaneIntersect(Vector3(topright.x(), bottomleft.y(), bottomleft.z()), axis, normal, d);
		rayBoxIntersection(t1, t2, center, ray, 6); 
		if(t1 < t2) {
			l.start = center + ray*t1;
			l.end = center + ray*t2;
			lines.push_back(l);
		}
	}

	while(lines.size() >= 3) { //not sure how it happens, but sometimes we only get two lines
		QList<Line>::iterator it = lines.begin();
		points.push_back((*it).start);
		points.push_back((*it).end);
		it = lines.erase(it);
		bool connected;
		while(lines.size()) {
			connected = false;
			for(it = lines.begin(); it != lines.end(); it++) {
				p = points.back() - (*it).start;
				if(p.length() < EPSILON) {
					points.push_back((*it).end);
					if ((points.first() - points.last()).length() < EPSILON) lines.clear();
					else lines.erase(it);
					connected = true;
					break;
				}
				p = points.back() - (*it).end;
				if(p.length() < EPSILON) {
					points.push_back((*it).start);
					if ((points.first() - points.last()).length() < EPSILON) lines.clear();
					else lines.erase(it);
					connected = true;
					break;
				}
			}
			if(!connected) { //Not a closed polygon!
				if (lines.size()) {
					points.clear();
					it = lines.begin();
					points.push_back((*it).start);
					points.push_back((*it).end);
					lines.erase(it);
					connected = true;
				}
				else
					break;
			}
		}
		if (!connected || points.size() < 3) break;

		//check that slicing plane is outsize the box
		Vector3 center(0.0,0.0,0.0);
		for(QList<Vector3>::iterator it = points.begin(); it != points.end(); ++it) {
			center += (*it);
		}
		center /= points.size();
		if (center.x() < bottomleft.x() || center.y() < bottomleft.y() || center.z() < bottomleft.z() ||
		    center.x() > topright.x() || center.y() > topright.y() || center.z() > topright.z()) break;

		//check that it's front facing, if not reverse it
		l.start = points[0] - points[1];
		l.end = points[2] - points[1];
		l.start = l.end*l.start;

		int i = 0, j = points.size(), k = 1;
		if(!(l.start.dot(-normal) > EPSILON)) { //front facing
			i = j - 1;
			j = -1;
			k = -1;
		}

		// draw slicing plain
		glBegin(GL_POLYGON);
		Vector3 tex;
		for(; i != j; i += k) {
			tex = points[i] - bottomleft;
			glTexCoord3d(tex.x()/pitch.x(), tex.y()/pitch.y(), tex.z()/pitch.z());
			glVertex3d(unpack3(points[i]));
		}
		glEnd();

		if ((points.first() - points.last()).length() < EPSILON) points.removeLast();
		i = 0; j = points.size(); k = 1;
		if(!(l.start.dot(normal) > EPSILON)) { //front facing
			i = j - 1;
			j = -1;
			k = -1;
		}

		if (sliceonly) return;

		// draw original but sliced plains
		int surface = -1;
		for(; i != j; i += k) {
			int nexti = i + k;
			surface = -1;
			if (nexti == points.size()) nexti = 0;
			else if (nexti == -1) nexti = points.size() - 1;

			if ((points[i].x() > points[nexti].x() - EPSILON && points[i].x() < points[nexti].x() + EPSILON) &&
			    (points[i].x() > topright.x() - EPSILON && points[i].x() < topright.x() + EPSILON)) { // x = topright.x() surface
				surface = 0;
			}
			if ((points[i].x() > points[nexti].x() - EPSILON && points[i].x() < points[nexti].x() + EPSILON) &&
			    (points[i].x() > bottomleft.x() - EPSILON && points[i].x() < bottomleft.x() + EPSILON)) { // x = bottomleft.x() surface
				surface = 1;
			}
			if ((points[i].y() > points[nexti].y() - EPSILON && points[i].y() < points[nexti].y() + EPSILON) &&
			    (points[i].y() > topright.y() - EPSILON && points[i].y() < topright.y() + EPSILON)) { // y = topright.y() surface
				surface = 2;
			}
			if ((points[i].y() > points[nexti].y() - EPSILON && points[i].y() < points[nexti].y() + EPSILON) &&
			    (points[i].y() > bottomleft.y() - EPSILON && points[i].y() < bottomleft.y() + EPSILON)) { // y = bottomleft.y() surface
				surface = 3;
			}
			if ((points[i].z() > points[nexti].z() - EPSILON && points[i].z() < points[nexti].z() + EPSILON) &&
			    (points[i].z() > topright.z() - EPSILON && points[i].z() < topright.z() + EPSILON)) { // z = topright.z() surface
				surface = 4;
			}
			if ((points[i].z() > points[nexti].z() - EPSILON && points[i].z() < points[nexti].z() + EPSILON) &&
			    (points[i].z() > bottomleft.z() - EPSILON && points[i].z() < bottomleft.z() + EPSILON)) { // z = bottomleft.z() surface
				surface = 5;
			}
			if (surface >= 0) {
				bool lined = false;
				glBegin(GL_POLYGON);
				for (int idx = 0; idx < 4; ++idx) {
					Vector3 v1 = vertex[surface][idx] - points[i];
					if (v1.dot(normal) > EPSILON) {
						tex = vertex[surface][idx] - bottomleft;
						glTexCoord3d(tex.x()/pitch.x(), tex.y()/pitch.y(), tex.z()/pitch.z());
						glVertex3d(unpack3(vertex[surface][idx]));
					}
					else if (!lined) {
						tex = points[i] - bottomleft;
						glTexCoord3d(tex.x()/pitch.x(), tex.y()/pitch.y(), tex.z()/pitch.z());
						glVertex3d(unpack3(points[i]));
						tex = points[nexti] - bottomleft;
						glTexCoord3d(tex.x()/pitch.x(), tex.y()/pitch.y(), tex.z()/pitch.z());
						glVertex3d(unpack3(points[nexti]));
						lined = true;
					}
				}
				glEnd();
				surfaces[surface] = true;
			}
		}
		break;
	}

	if (sliceonly) return;

	// draw non-sliced plains
	glBegin(GL_QUADS);
	for (int i = 0; i < 6; ++i) {
		if (!surfaces[i]) {
			Vector3 v1 = (vertex[i][0] + vertex[i][1] + vertex[i][2] + vertex[i][3])/4 - center;
			if (v1.dot(normal) > EPSILON) {
				for (int idx = 0; idx < 4; ++idx) {
					Vector3 tex = vertex[i][idx] - bottomleft;
					glTexCoord3d(tex.x()/pitch.x(), tex.y()/pitch.y(), tex.z()/pitch.z());
					glVertex3d(unpack3(vertex[i][idx]));
				}
			}
		}
	}
	glEnd();
} 

void Box::drawMultiSlicedBox(const QList<Slicer> & _slicers, bool sliceonly) const {
	if(pitch.lengthSquared() < 0.000001)
		return;

	QList<Slicer> slicers = _slicers;

	Vector3 vertex[6][4];
	vertex[0][0] = Vector3(topright.x(),  topright.y(),  topright.z());
	vertex[0][1] = Vector3(topright.x(),  bottomleft.y(),  topright.z());
	vertex[0][2] = Vector3(topright.x(),  bottomleft.y(), bottomleft.z());
	vertex[0][3] = Vector3(topright.x(),  topright.y(), bottomleft.z());
	vertex[1][0] = Vector3(bottomleft.x(),  topright.y(),  bottomleft.z());
	vertex[1][1] = Vector3(bottomleft.x(),  bottomleft.y(),  bottomleft.z());
	vertex[1][2] = Vector3(bottomleft.x(),  bottomleft.y(), topright.z());
	vertex[1][3] = Vector3(bottomleft.x(),  topright.y(), topright.z());
	vertex[2][0] = Vector3(bottomleft.x(), topright.y(), topright.z());
	vertex[2][1] = Vector3(topright.x(), topright.y(), topright.z());
	vertex[2][2] = Vector3(topright.x(), topright.y(), bottomleft.z());
	vertex[2][3] = Vector3(bottomleft.x(), topright.y(), bottomleft.z());
	vertex[3][0] = Vector3(bottomleft.x(), bottomleft.y(), bottomleft.z());
	vertex[3][1] = Vector3(topright.x(), bottomleft.y(), bottomleft.z());
	vertex[3][2] = Vector3(topright.x(), bottomleft.y(), topright.z());
	vertex[3][3] = Vector3(bottomleft.x(), bottomleft.y(), topright.z());
	vertex[4][0] = Vector3(bottomleft.x(), topright.y(), topright.z());
	vertex[4][1] = Vector3(bottomleft.x(), bottomleft.y(), topright.z());
	vertex[4][2] = Vector3(topright.x(), bottomleft.y(), topright.z());
	vertex[4][3] = Vector3(topright.x(), topright.y(), topright.z());
	vertex[5][0] = Vector3(bottomleft.x(), topright.y(), bottomleft.z());
	vertex[5][1] = Vector3(topright.x(), topright.y(), bottomleft.z());
	vertex[5][2] = Vector3(topright.x(), bottomleft.y(), bottomleft.z());
	vertex[5][3] = Vector3(bottomleft.x(), bottomleft.y(), bottomleft.z());
	bool surfaces[6];
	for (int i = 0; i < 6; ++i) surfaces[i] = false;

	QList<Vector3> centers;
	QList<Vector3> normals;
	QList<double> dists;
	QList< QList<Line> > liness;
	QList< QList<Vector3> > pointss;
	QList<bool> facings;

	for (int i = 0; i < slicers.size(); ++i) {
		Vector3 sliceVec = slicers[i].getVec();
		double sDepth = slicers[i].getDist();

		Vector3 center = getCenterPoint() + sliceVec * sDepth;
		Vector3 normal = -sliceVec;
		double d = sliceVec.dot(center);
		QList<Line> lines;
		QList<Vector3> points;

		Vector3 p;
		Line l;
		Vector3 ray;
		double t1, t2;
		Vector3 axis;

		if(fabs(fabs(normal.dot(Vector3::zAxis)) - 1.) > EPSILON) { //xy
			//first, find point along x where it intersects our slice plane

			if(fabs(normal.dot(Vector3::xAxis)) > EPSILON)
				axis = Vector3::xAxis;
			else
				axis = Vector3::yAxis;

			center = rayPlaneIntersect(bottomleft, axis, normal, d);
			ray = normal*Vector3::zAxis;
			rayBoxIntersection(t1, t2, center, ray, 3); 
			if(t1 < t2) {
				l.start = center + ray*t1;
				l.end = center + ray*t2;
				lines.push_back(l);
			}
			center= rayPlaneIntersect(Vector3(bottomleft.x(), bottomleft.y(), topright.z()), axis, normal, d);
			rayBoxIntersection(t1, t2, center, ray, 3); 
			if(t1 < t2) {
				l.start = center + ray*t1;
				l.end = center + ray*t2;
				lines.push_back(l);
			}
		}

		if(fabs(fabs(normal.dot(Vector3::yAxis)) - 1.) > EPSILON) { //xz

			if(fabs(normal.dot(Vector3::xAxis)) > EPSILON)
				axis = Vector3::xAxis;
			else
				axis = Vector3::zAxis;

			center = rayPlaneIntersect(bottomleft, axis, normal, d);
			ray = normal*Vector3::yAxis;
			rayBoxIntersection(t1, t2, center, ray, 5); 
			if(t1 < t2) {
				l.start = center + ray*t1;
				l.end = center + ray*t2;
				lines.push_back(l);
			}
			center = rayPlaneIntersect(Vector3(bottomleft.x(), topright.y(), bottomleft.z()), axis, normal, d);
			rayBoxIntersection(t1, t2, center, ray, 5);
			if(t1 < t2) {
				l.start = center + ray*t1;
				l.end = center + ray*t2;
				lines.push_back(l);
			}
		}

		if(fabs(fabs(normal.dot(Vector3::xAxis)) - 1.) > EPSILON) { //yz

			if(fabs(normal.dot(Vector3::yAxis)) > EPSILON)
				axis = Vector3::yAxis;
			else
				axis = Vector3::zAxis;

			center = rayPlaneIntersect(bottomleft, axis, normal, d);
			ray = normal*Vector3::xAxis;
			rayBoxIntersection(t1, t2, center, ray, 6); 
			if(t1 < t2) {
				l.start = center + ray*t1;
				l.end = center + ray*t2;
				lines.push_back(l);
			}
			center = rayPlaneIntersect(Vector3(topright.x(), bottomleft.y(), bottomleft.z()), axis, normal, d);
			rayBoxIntersection(t1, t2, center, ray, 6); 
			if(t1 < t2) {
				l.start = center + ray*t1;
				l.end = center + ray*t2;
				lines.push_back(l);
			}
		}

		centers.push_back(center);
		normals.push_back(normal);
		dists.push_back(d);
		liness.push_back(lines);
		pointss.push_back(points);
	}


//	while(lines.size() >= 3) { //not sure how it happens, but sometimes we only get two lines

	// build points according to lines
	for (int i = 0; i < slicers.size(); ++i) {
		QList<Line> & lines = liness[i];
		QList<Vector3> & points = pointss[i];
		Vector3 normal = normals[i];

		if (lines.size() < 3) continue;

		QList<Line>::iterator it = lines.begin();
		points.push_back((*it).start);
		points.push_back((*it).end);
		it = lines.erase(it);
		bool connected;
		while(lines.size()) {
			connected = false;
			for(it = lines.begin(); it != lines.end(); it++) {
				Vector3 p = points.back() - (*it).start;
				if(p.length() < EPSILON) {
					points.push_back((*it).end);
					if ((points.first() - points.last()).length() < EPSILON) lines.clear();
					else lines.erase(it);
					connected = true;
					break;
				}
				p = points.back() - (*it).end;
				if(p.length() < EPSILON) {
					points.push_back((*it).start);
					if ((points.first() - points.last()).length() < EPSILON) lines.clear();
					else lines.erase(it);
					connected = true;
					break;
				}
			}
			if(!connected) { //Not a closed polygon!
				if (lines.size()) {
					points.clear();
					it = lines.begin();
					points.push_back((*it).start);
					points.push_back((*it).end);
					lines.erase(it);
					connected = true;
				}
				else break;
			}
		}
		if (!connected || points.size() < 3) {
			slicers.removeAt(i);
			pointss.removeAt(i);
			liness.removeAt(i);
			i--;
		}
		else {
			// check that slicing plane is clipping the box
			Vector3 center(0.0,0.0,0.0);
			for(QList<Vector3>::iterator it = points.begin(); it != points.end(); ++it) {
				center += (*it);
			}
			center /= points.size();
			if (center.x() < bottomleft.x() || center.y() < bottomleft.y() || center.z() < bottomleft.z() ||
			    center.x() > topright.x() || center.y() > topright.y() || center.z() > topright.z()) {
				slicers.removeAt(i);
				pointss.removeAt(i);
				liness.removeAt(i);
				i--;
			}
			else {
				if ((points.first() - points.last()).length() < EPSILON) points.removeLast();

				// compute if front facing
				Line l;
				l.start = points[0] - points[1];
				l.end = points[2] - points[1];
				l.start = l.end*l.start;
				if(l.start.dot(-normal) > EPSILON) facings.push_back(true);
				else facings.push_back(false);
			}
		}
	}

	// slices are also cut by slices
	for (int s = 0; s < pointss.size(); ++s) {
		for (int t = 0; t < pointss.size(); ++t) {
			if (s == t) continue;

			QList<Vector3> & points = pointss[s];
			QList<Vector3> & points2 = pointss[t];

			int i = 0, j = points.size(), k = 1;
			int i2 = 0, j2 = points2.size(), k2 = 1;

			Vector3 & center = centers[s];
			Vector3 & center2 = centers[t];
			double & d = dists[s];
			double & d2 = dists[t];
			Vector3 & normal = normals[s];
			Vector3 & normal2 = normals[t];

			if (!(normal == normal2) && !(normal == -normal2)) { // not parallel: intersect somewhere
				for (; i != j; i += k) {
					int nexti = i + k;
					if (nexti == points.size()) nexti = 0;

					double dot1 = normal2.dot(points[i] - center2);
					double dot2 = normal2.dot(points[nexti] - center2);
					if (dot1 * dot2 < 0.0) {
						Vector3 ray = points[nexti] - points[i];
						ray.normalize();
						Vector3 intersect = rayPlaneIntersect(points[i], ray, normal2, d2);

						if (dot1 < 0.0) {
							if (i != 0) {
								points.removeAt(i);
								points.insert(i, intersect);
							}
							else {
								points.insert(nexti, intersect);
								i += k;
							}
						}
						else if (dot2 < 0.0) {
							if (nexti == 0) {
								points.removeAt(nexti);
								points.push_back(intersect);
							}
							else {
								points.insert(nexti, intersect);
								i += k;
							}
						}
					}
					else if (dot1 < 0.0 && dot2 < 0.0) {
						if (i != 0) {
							points.removeAt(i);
							i -= k;
						}
						if (nexti == 0) {
							points.removeAt(nexti);
							i -= k;
						}
					}
					j = points.size();
				}
			}
			else if ((normal == normal2 && d > d2) || (normal == -normal2 && d + d2 == 0)) {
				points.clear();
			}
		}
	}

	// draw slicing plain
	for (int s = 0; s < pointss.size(); ++s) {
		QList<Vector3> & points = pointss[s];
		int i = 0, j = points.size(), k = 1;
		if (!facings[s]) {
			i = j - 1;
			j = -1;
			k = -1;
		}

		if (sliceonly) {
			if (!slicers[s].isMasking()) continue;
			QColor mcolor = slicers[s].getMaskingColor();
			glColor4d(mcolor.redF(), mcolor.greenF(), mcolor.blueF(), mcolor.alphaF());
		}

		glBegin(GL_POLYGON);
		Vector3 tex;
		for(; i != j; i += k) {
			tex = points[i] - bottomleft;
			glTexCoord3d(tex.x()/pitch.x(), tex.y()/pitch.y(), tex.z()/pitch.z());
			glVertex3d(unpack3(points[i]));
		}
		glEnd();
	}

	if (sliceonly) return;

	for (int s = 0; s < 6; ++s) {

		QList< QList<Vector3> > pointlists;

		if (vertex[s][0].x() == vertex[s][1].x() && vertex[s][0].x() == vertex[s][2].x() && vertex[s][0].x() == vertex[s][3].x()) {
			if (vertex[s][0].x() > topright.x() - EPSILON && vertex[s][0].x() < topright.x() + EPSILON) {
				for (int t = 0; t < pointss.size(); ++t) {
					QList<Vector3> pointlist;
					QList<Vector3> & points = pointss[t];
					int i = 0, j = points.size(), k = 1;
					if (facings[t]) {
						i = j - 1;
						j = -1;
						k = -1;
					}
					for (;i != j; i += k) {
						if (points[i].x() > topright.x() - EPSILON && points[i].x() < topright.x() + EPSILON) {
							if (i == j - k && !(points[i-k].x() > topright.x() - EPSILON && points[i-k].x() < topright.x() + EPSILON)) pointlist.push_front(points[i]);
							else pointlist.push_back(points[i]);
						}
					}
					if (pointlist.size() > 0) pointlists.push_back(pointlist);
				}
			}
			else if (vertex[s][0].x() > bottomleft.x() - EPSILON && vertex[s][0].x() < bottomleft.x() + EPSILON) {
				for (int t = 0; t < pointss.size(); ++t) {
					QList<Vector3> pointlist;
					QList<Vector3> & points = pointss[t];
					int i = 0, j = points.size(), k = 1;
					if (facings[t]) {
						i = j - 1;
						j = -1;
						k = -1;
					}
					for (;i != j; i +=k ) {
						if (points[i].x() > bottomleft.x() - EPSILON && points[i].x() < bottomleft.x() + EPSILON) {
							if (i == j - k && !(points[i-k].x() > bottomleft.x() - EPSILON && points[i-k].x() < bottomleft.x() + EPSILON)) pointlist.push_front(points[i]);
							else pointlist.push_back(points[i]);
						}
					}
					if (pointlist.size() > 0) pointlists.push_back(pointlist);
				}
			}
		}
		else if (vertex[s][0].y() == vertex[s][1].y() && vertex[s][0].y() == vertex[s][2].y() && vertex[s][0].y() == vertex[s][3].y()) {
			if (vertex[s][0].y() > topright.y() - EPSILON && vertex[s][0].y() < topright.y() + EPSILON) {
				for (int t = 0; t < pointss.size(); ++t) {
					QList<Vector3> pointlist;
					QList<Vector3> & points = pointss[t];
					int i = 0, j = points.size(), k = 1;
					if (facings[t]) {
						i = j - 1;
						j = -1;
						k = -1;
					}
					for (;i != j; i +=k ) {
						if (points[i].y() > topright.y() - EPSILON && points[i].y() < topright.y() + EPSILON) {
							if (i == j - k && !(points[i-k].y() > topright.y() - EPSILON && points[i-k].y() < topright.y() + EPSILON)) pointlist.push_front(points[i]);
							else pointlist.push_back(points[i]);
						}
					}
					if (pointlist.size() > 0) pointlists.push_back(pointlist);
				}
			}
			else if (vertex[s][0].y() > bottomleft.y() - EPSILON && vertex[s][0].y() < bottomleft.y() + EPSILON) {
				for (int t = 0; t < pointss.size(); ++t) {
					QList<Vector3> pointlist;
					QList<Vector3> & points = pointss[t];
					int i = 0, j = points.size(), k = 1;
					if (facings[t]) {
						i = j - 1;
						j = -1;
						k = -1;
					}
					for (;i != j; i +=k ) {
						if (points[i].y() > bottomleft.y() - EPSILON && points[i].y() < bottomleft.y() + EPSILON) {
							if (i == j - k && !(points[i-k].y() > bottomleft.y() - EPSILON && points[i-k].y() < bottomleft.y() + EPSILON)) pointlist.push_front(points[i]);
							else pointlist.push_back(points[i]);
						}
					}
					if (pointlist.size() > 0) pointlists.push_back(pointlist);
				}
			}
		}
		else if (vertex[s][0].z() == vertex[s][1].z() && vertex[s][0].z() == vertex[s][2].z() && vertex[s][0].z() == vertex[s][3].z()) {
			if (vertex[s][0].z() > topright.z() - EPSILON && vertex[s][0].z() < topright.z() + EPSILON) {
				for (int t = 0; t < pointss.size(); ++t) {
					QList<Vector3> pointlist;
					QList<Vector3> & points = pointss[t];
					int i = 0, j = points.size(), k = 1;
					if (facings[t]) {
						i = j - 1;
						j = -1;
						k = -1;
					}
					for (;i != j; i +=k ) {
						if (points[i].z() > topright.z() - EPSILON && points[i].z() < topright.z() + EPSILON) {
							if (i == j - k && !(points[i-k].z() > topright.z() - EPSILON && points[i-k].z() < topright.z() + EPSILON)) pointlist.push_front(points[i]);
							else pointlist.push_back(points[i]);
						}
					}
					if (pointlist.size() > 0) pointlists.push_back(pointlist);
				}
			}
			else if (vertex[s][0].z() > bottomleft.z() - EPSILON && vertex[s][0].z() < bottomleft.z() + EPSILON) {
				for (int t = 0; t < pointss.size(); ++t) {
					QList<Vector3> pointlist;
					QList<Vector3> & points = pointss[t];
					int i = 0, j = points.size(), k = 1;
					if (facings[t]) {
						i = j - 1;
						j = -1;
						k = -1;
					}
					for (;i != j; i +=k ) {
						if (points[i].z() > bottomleft.z() - EPSILON && points[i].z() < bottomleft.z() + EPSILON) {
							if (i == j - k && !(points[i-k].z() > bottomleft.z() - EPSILON && points[i-k].z() < bottomleft.z() + EPSILON)) pointlist.push_front(points[i]);
							pointlist.push_back(points[i]);
						}
					}
					if (pointlist.size() > 0) pointlists.push_back(pointlist);
				}
			}
		}

		glBegin(GL_POLYGON);
		for (int p = 0; p < 4; ++p) {
			bool in = true;
			for (int t = 0; t < centers.size(); ++t) {
				if ((vertex[s][p]-centers[t]).dot(normals[t]) < -EPSILON) in = false;
			}
			if (in) glVertex3d(unpack3(vertex[s][p]));

			for (int t = 0; t < pointlists.size(); ++t) {
				QList<Vector3> & pointlist = pointlists[t];
				if (pointlist.size() > 0) {
					int nextp = p + 1;
					if (nextp == 4) nextp = 0;
					double u = ((pointlist[0].x() - vertex[s][p].x()) * (vertex[s][nextp].x() - vertex[s][p].x()) + 
							(pointlist[0].y() - vertex[s][p].y()) * (vertex[s][nextp].y() - vertex[s][p].y()) + 
							(pointlist[0].z() - vertex[s][p].z()) * (vertex[s][nextp].z() - vertex[s][p].z())) / 
							(vertex[s][nextp] - vertex[s][p]).length() / (vertex[s][nextp] - vertex[s][p]).length();
					Vector3 midp = vertex[s][p] + u * (vertex[s][nextp] - vertex[s][p]);
					if ((pointlist[0] - midp).length() < EPSILON && u >= 0 && u <= 1) {
						glVertex3d(unpack3(pointlist[0]));
						glVertex3d(unpack3(pointlist[1]));
						Vector3 endpoint = pointlist[1];
						bool end;
						do {
							end = false;
							for (int u = 0; u < pointlists.size(); ++u) {
								if (t == u) continue;
								if (pointlists[u].size() > 1) {
									if ((endpoint - pointlists[u][0]).length() < EPSILON) {
										glVertex3d(unpack3(pointlists[u][1]));
										endpoint = pointlists[u][1];
										pointlists[u].clear();
										pointlists.removeAt(u);
										if (u < t) t--;
										u--;
										end = true;
										break;
									}
								}
							}
						} while(end);
						pointlist.clear();
						pointlists.removeAt(t);
						t--;
					}
				}
			}
		}

		if (pointlists.size() > 0) {
			int startidx = 0;
			int roundnum = 0;
			bool end;

			do {
				end = false;
				for (int t = 0; t < pointlists.size(); ++t) {
					if (startidx == t) continue;
					if ((pointlists[startidx][0]-pointlists[t][1]).length() < EPSILON) {
						startidx = t;
						end = true;
					}
				}
				roundnum++;
			} while(roundnum < pointlists.size() && end);

			roundnum = 0;
			do {
				glVertex3d(unpack3(pointlists[startidx][0]));
				glVertex3d(unpack3(pointlists[startidx][1]));
				end = false;
				for (int t = 0; t < pointlists.size(); ++t) {
					if (startidx == t) continue;
					if ((pointlists[startidx][1] - pointlists[t][0]).length() < EPSILON) {
						startidx = t;
						end = true;
					}
				}
				roundnum++;
			} while(roundnum < pointlists.size() && end);
		}
		glEnd();
	}
} 

//dumb ray/box intersection, it's assume the box is axis aligned. Otherwise...QQ.
void Box::rayBoxIntersection(double &tnear, double& tfar, 
						const Vector3& start, const Vector3& ray, unsigned char planes) const {
	tnear = -std::numeric_limits<double>::max();
	tfar = std::numeric_limits<double>::max();
	double t1, t2, s;

	if((planes & 1) && fabs(ray.x()) > EPSILON) { //check x boundaries
		t1 = (bottomleft.x() - start.x())/ray.x();
		t2 = (topright.x() - start.x())/ray.x();

		if(t1 > t2) {
			s = t1;
			t1 = t2;
			t2 = s;
		}
		if(t1 > tnear) {
			tnear = t1;
		}
		if(t2 < tfar) {
			tfar = t2;
		}
	}
	if((planes & 2) && fabs(ray.y()) > EPSILON) { //check y boundaries		
		t1 = (bottomleft.y() - start.y())/ray.y();
		t2 = (topright.y() - start.y())/ray.y();

		if(t1 > t2) {
			s = t1;
			t1 = t2;
			t2 = s;
		}
		if(t1 > tnear) {
			tnear = t1;
		}
		if(t2 < tfar) {
			tfar = t2;
		}

	}
	if((planes & 4) && fabs(ray.z()) > EPSILON) { //check z boundaries
		t1 = (bottomleft.z() - start.z())/ray.z();
		t2 = (topright.z() - start.z())/ray.z();

		if(t1 > t2) {
			s = t1;
			t1 = t2;
			t2 = s;
		}
		if(t1 > tnear) {
			tnear = t1;
		}
		if(t2 < tfar) {
			tfar = t2;
		}
	}
}




void Box::drawBox() {
	if(pitch.lengthSquared() < 0.00001)
		return;
	constructVBO();
	vbo->bind();
	ibo->bind();
	ibo->draw();
	ibo->release();
	vbo->release();
}

void Box::constructVBO() {

	if(!vbo) {
		vbo = new GLVertexbufferf(GL_QUADS, GL_STATIC_DRAW);
		(*vbo) << bottomleft
				<< Vector3(topright.x(), bottomleft.y(), bottomleft.z())
				<< Vector3(bottomleft.x(), topright.y(), bottomleft.z())
				<< Vector3(topright.x(), topright.y(), bottomleft.z())
				<< Vector3(bottomleft.x(), bottomleft.y(), topright.z())
				<< Vector3(topright.x(), bottomleft.y(), topright.z())
				<< Vector3(bottomleft.x(), topright.y(), topright.z())
				<< topright;
	}
	if(!ibo) {
		ibo = new GLIndexbuffer(GL_QUADS, GL_STATIC_DRAW);
		(*ibo) << 0 << 1 << 5 << 4
				<< 6 << 2 << 0 << 4
				<< 2 << 3 << 1 << 0
				<< 7 << 6 << 4 << 5
				<< 3 << 7 << 5 << 1
				<< 3 << 2 << 6 << 7;
	}
}

Vector3 Box::getCenterPoint() const {
	return bottomleft + 0.5*pitch;
}


void Box::setPitch(const Vector3& p) {
	pitch = p;
	topright = bottomleft + pitch;
	if(vbo)
		delete vbo;
	vbo = 0;
}
void Box::setBottomleft(const Vector3& b) {
	bottomleft = b;
	topright = bottomleft + pitch;
	if(vbo)
		delete vbo;
	vbo = 0;
}
void Box::setTopright(const Vector3& p) {
	topright = p;
	pitch = topright - bottomleft;
	if(vbo)
		delete vbo;
	vbo = 0;
}
Box& Box::operator=(const Box& rhs) {
	bottomleft = rhs.bottomleft;
	topright = rhs.topright;
	pitch = rhs.pitch;
	if(vbo)
		delete vbo;
	vbo = 0;
	return *this;
}

