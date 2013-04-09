#include "QAniClickable.h"
#include "QAniEditor.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

QAniClickable::QAniClickable(QWidget *parent) : QWidget(parent), timelineGL((QAniEditor*)parent)
{
	dragging = false;
	selected = false;
	visible = true;
}

bool QAniClickable::isSelected() {
	return selected;
}
void QAniClickable::drag(float dx, float dy) {
	x=dx;
	y=dy;
}
bool QAniClickable::encloses(float px, float py) {
	return (px>x && px<x+w && py>y && py<y+h);
}
bool QAniClickable::isVisible() {
	return visible;
}
void QAniClickable::setVisible(bool value) {
	visible = value;
}
void QAniClickable::draw() {
	if(selected) {
		glColor4d(0.2,0.4,1.0, 0.9);
	} else {
		glColor4d(0.4,0.4,0.5,0.6);
	}
	glRectf(x, y, x+w, y+h);
}
void QAniClickable::updateGL() {
	if (timelineGL) timelineGL->updateGL();
}

void QAniClickable::setCursor(Qt::CursorShape q) {
	timelineGL->setCursor(q);
}
/* class QAniTimelineScaler **********************************************************************************************
 *************************************************************************************************************************
 *
 */

QAniTimelineScaler::QAniTimelineScaler(QWidget *parent) : QAniClickable(parent) {
	offset = -5.0;
	scale = 20.0;
	dragable = true;
	dragging = false;
	offsetTimer = new QTimer(this);
	offsetTimer->setSingleShot(false);
	connect(offsetTimer, SIGNAL(timeout()), this, SLOT(offsetTimeout()));
}

void QAniTimelineScaler::draw() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0,timelineGL->width(),0,timelineGL->height(),-1,1); // 1 unit of OpenGL = 1 pixel
	float xTranslate = -51.0; // Translate because the left side toolbox is 50 pixel wide.
	float xScale = scale; // Scale because of the scaling range of the timeline.
	float left = offset;
	float right = left + (timelineGL->width()+xTranslate)/xScale;
	glTranslatef(-xTranslate,0.0,0.0);
	glScalef(xScale,1.0,1.0);
	glTranslatef(-offset,0.0,0.0);

	float markerDistance = 100/xScale;
	float markerSubDistance = 0.1f;
	float markerPower = 1.0;
	markerPower = pow(10, floor(log10(markerDistance)));
	markerDistance /= markerPower;
	if (markerDistance < 1.5) markerDistance = 1.0; 
	else if (markerDistance < 3.5) { markerDistance = 2.0; markerSubDistance = 0.5; }
	else if (markerDistance < 7.5) { markerDistance = 5.0; markerSubDistance = 0.2f; }
	else { markerDistance = 1.0; markerPower *= 10; }
	markerDistance *= markerPower;

	float leftStart;
	leftStart = markerDistance * floor(left/markerDistance);

	glColor3d(0.6,0.6,0.6);
	glBegin(GL_LINES);
		glVertex2f(left-1,20); glVertex2f(right,20);
		for (float i = leftStart; i<= right; i += markerDistance) {
			if (i >= left) {glVertex2f(i,20); glVertex2f(i,30);}
			for (float j = i + markerDistance * markerSubDistance; j <= right && j < i + markerDistance; j += markerDistance * markerSubDistance) {
				if (j >= left) {glVertex2f(j,20); glVertex2f(j,25);}
			}
		}
	glEnd();
	for (float i = leftStart; i<= right; i+=markerDistance) if (i >= left) timelineGL->renderText((double)i,5.0,0.0,QString::number(i, 'f', (-1*log10(markerPower)>0?-1*log10(markerPower):0)));
}

void QAniTimelineScaler::wheel(float px, float, int delta) {
	float value = offset + (px - 51.0) / scale;
	scale *= delta>0?1.1:1/1.1;
	offset = value - (px - 51.0) / scale;
	updateGL();
}

void QAniTimelineScaler::press(float px, float py) {
	dragging = true;
	drag_ox = px;
	drag_oy = py;
}

void QAniTimelineScaler::drag(float dx, float dy) {
	if (!dragging) return;
	float delta_x = dx - drag_ox;
	offset -= delta_x/scale;
	drag_ox = dx;
	drag_oy = dy;
	updateGL();
}

void QAniTimelineScaler::release(float, float) {
	dragging = false;
}

void QAniTimelineScaler::resize() {
	setRect(51, timelineGL->height()-30, timelineGL->width()-51, 30);
}

void QAniTimelineScaler::offsetTimeout() {
	offset += scrollDx;
	updateGL();
}

/* class QAniToolBox *****************************************************************************************************
 *************************************************************************************************************************
 *
 */

QAniToolBox::QAniToolBox(QWidget * parent) : QAniClickable(parent) {
	m_playing = false;
	m_hoverPlaying = false;
	m_hoverRecord = false;
}

QAniToolBox::~QAniToolBox() {
}

void QAniToolBox::draw() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0,timelineGL->width(),0,timelineGL->height(),-1,1); // 1 unit of OpenGL = 1 pixel
	glScalef(1.0,-1.0,1.0);
	glTranslatef(0.0,-timelineGL->height(),0.0);

	drawBackground();

	drawButtons();
}

void QAniToolBox::drawBackground() {
	// background
	glBegin(GL_QUADS);
		glColor3d(1.0,1.0,1.0); glVertex2f(x,y);
		glColor3d(1.0,1.0,1.0); glVertex2f(x+w,y);
		glColor3d(1.0,1.0,1.0); glVertex2f(x+w,y+h);
		glColor3d(1.0,1.0,1.0); glVertex2f(x,y+h);
	glEnd();

	glBegin(GL_LINES);
		glColor3d(0.8,0.8,0.8);
		glVertex2f(x+w-1,y);
		glVertex2f(x+w-1,y+h);
		glColor3d(0.3,0.3,0.3);
		glVertex2f(x+w,y);
		glVertex2f(x+w,y+h);
	glEnd();
}

void QAniToolBox::drawButtons() {
	glPushMatrix();
	glTranslatef(x,y,0);
	glTranslatef(15,10,0);

	// capture button
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	float hovercolor = 6.0;
	if (m_hoverRecord) hovercolor = 10.0;
	glBegin(GL_TRIANGLE_FAN);
	for(int i = 0; i < 10; ++i) {
		glColor3d(1.0 - (float)abs(i-5) / hovercolor,0.0,0.0); glVertex2f(10+10*cos(M_PI/5*(i+1)),10+10*sin(M_PI/5*(i+1)));
	}
	glEnd();
	glColor3d(0.0,0.0,0.0);
	glBegin(GL_LINE_STRIP);
	for(int i = 0; i < 11; ++i) {
		glVertex2f(10+10*cos(M_PI/5*i),10+10*sin(M_PI/5*i));
	}
	glEnd();


	glTranslatef(0,30,0);


	// play button
	if (!m_playing) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBegin(GL_TRIANGLES);
			glColor3d(0.0,1.0,0.0); glVertex2f(0.0,0.0);
			if (m_hoverPlaying) glColor3d(0.0,0.6,0.0);
			else glColor3d(0.0,0.2,0.0);
			glVertex2f(0.0,20.0);
			glVertex2f(20.0,10.0);
		glEnd();
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glColor3d(0.0,0.0,0.0);
		glBegin(GL_TRIANGLES);
			glVertex2f(0.0,0.0);
			glVertex2f(0.0,20.0);
			glVertex2f(20.0,10.0);
		glEnd();
	}
	// pause button
	else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBegin(GL_QUADS);
			glColor3d(1.0,1.0,0.0); glVertex2f(0.0,0.0);
			glColor3d(1.0,1.0,0.0); glVertex2f(0.0,20.0);
			if (m_hoverPlaying) glColor3d(0.6,0.6,0.0);
			else glColor3d(0.2,0.2,0.0);
			glVertex2f(5.0,20.0);
			glVertex2f(5.0,0.0);
			glColor3d(1.0,1.0,0.0); glVertex2f(15.0,0.0);
			glColor3d(1.0,1.0,0.0); glVertex2f(15.0,20.0);
			if (m_hoverPlaying) glColor3d(0.6,0.6,0.0);
			else glColor3d(0.2,0.2,0.0);
			glVertex2f(20.0,20.0);
			glVertex2f(20.0,0.0);
		glEnd();
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glColor3d(0.0,0.0,0.0);
		glBegin(GL_QUADS);
			glVertex2f(0.0,0.0);
			glVertex2f(0.0,20.0);
			glVertex2f(5.0,20.0);
			glVertex2f(5.0,0.0);
			glVertex2f(15.0,0.0);
			glVertex2f(15.0,20.0);
			glVertex2f(20.0,20.0);
			glVertex2f(20.0,0.0);
		glEnd();

	}


	glTranslatef(0,30,0);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor3d(0.0,0.0,0.0);
	glBegin(GL_QUADS);
	glVertex2f(0.0,0.0); glVertex2f(-1.0,4.0); glVertex2f(4.0,4.0); glVertex2f(4.0,0.0);
	glVertex2f(8.0,0.0); glVertex2f(7.0,4.0); glVertex2f(12.0,4.0); glVertex2f(12.0,0.0);
	glVertex2f(16.0,0.0); glVertex2f(15.0,4.0); glVertex2f(20.0,4.0); glVertex2f(20.0,0.0);
	glVertex2f(8.0,8.0); glVertex2f(7.0,12.0); glVertex2f(12.0,12.0); glVertex2f(12.0,8.0);
	glVertex2f(16.0,8.0); glVertex2f(15.0,12.0); glVertex2f(20.0,12.0); glVertex2f(20.0,8.0);
	glVertex2f(16.0,16.0); glVertex2f(15.0,20.0); glVertex2f(20.0,20.0); glVertex2f(20.0,16.0);
	glEnd();

	glBegin(GL_LINES);
	glVertex2f(4.0,2.0); glVertex2f(8.0,2.0);
	glVertex2f(12.0,2.0); glVertex2f(16.0,2.0);
	glVertex2f(10.0,4.0); glVertex2f(10.0,8.0);
	glVertex2f(12.0,10.0); glVertex2f(16.0,10.0);
	glVertex2f(18.0,12.0); glVertex2f(18.0,16.0);
	glEnd();

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPopMatrix();
}

void QAniToolBox::hover(float px, float py) {
	m_hoverPlaying = false;
	m_hoverRecord = false;
	if (px >= 15 && px <= 35 && py >= 10 && py <=40) {
		m_hoverRecord = true;
	}
	else if (px >= 15 && px <= 35 && py >= 40 && py <=70) {
		m_hoverPlaying = true;
	}
	updateGL();
}

void QAniToolBox::press(float px, float py) {
	// Keyframing
	if (px >= 15 && px <= 35 && py >= 10 && py <=40) {
		emit recordButtonHit();
	}
	// Playing
	else if (px >= 15 && px <= 35 && py >= 40 && py <=70) {
		m_playing = !m_playing;
		emit playButtonHit();
		updateGL();
	}
	// Image Graph
	else if (px >= 15 && px <= 35 && py >= 70 && py <=100) {
		emit menuButtonHit();
	}
}

void QAniToolBox::release(float, float) {
}

void QAniToolBox::resize() {
	setRect(0,0,50,100);
}

/* class QAniTemplateBox *************************************************************************************************
 *************************************************************************************************************************
 *
 */
QAniTemplateBox::QAniTemplateBox(QWidget * parent) : QAniClickable(parent) {
	offsety = -5.0;
	scroll = 0;
	templateDragIdx = -1;
}

QAniTemplateBox::~QAniTemplateBox() {
}

void QAniTemplateBox::draw() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0,timelineGL->width(),0,timelineGL->height(),-1,1); // 1 unit of OpenGL = 1 pixel

	// draw QAniTemplate Dragging if already in QAniTimeline
	if (isDragging() && getTemplateDrag()->dragInTimeline) {
		glPushMatrix();
		glScalef(1.0,-1.0,1.0);
		glTranslatef(0.0,-timelineGL->height(),0.0);
		getTemplateDrag()->drawDragging();
		glPopMatrix();
	}

	glScalef(1.0,-1.0,1.0);
	glTranslatef(0.0,-timelineGL->height(),0.0);

	// draw QAniTemplate Box
	drawBackground();

	float cury = y;

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glTranslatef(x,y,0.0);
	glTranslatef(10.0,15.0,0.0);
	cury += 15.0;

	if (offsety < 0) {
		glTranslatef(0.0,-offsety,0.0);
		cury-=offsety;
	}

	// determine the first template icon that is to be shown
	int first = floor((offsety>=0?offsety:0) / 35.0);
	float firstoffset = (float)(first + 1) * 35.0 - (offsety>=0?offsety:0);
	first = first >= 0 ? first : 0;

	for (unsigned int i = (unsigned int)first; i < templates.size(); ++i) {
		if (i == (unsigned int)first) {
			templates[i]->draw(-firstoffset);
			glTranslatef(0.0,firstoffset,0.0);
			cury += firstoffset;
		}
		else if (cury + 35.0 > y+h - 15.0) {
			templates[i]->draw(y+h - 15.0 - cury);
			break;
		}
		else {
			templates[i]->draw();
			glTranslatef(0.0,35.0,0.0);
			cury += 35.0;
		}
	}

	glPopMatrix();


	// draw QAniTemplate Dragging if not yet in QAniTimeline
	if (isDragging() && !getTemplateDrag()->dragInTimeline) {
		getTemplateDrag()->drawDragging();
	}
}

void QAniTemplateBox::add(QAniTemplate* t) {
	templates.push_back(t);
	t->tbox = this;
}

void QAniTemplateBox::drawBackground() {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// background
	glBegin(GL_QUADS);
		glColor3d(0.7,0.95,1.0); glVertex2f(x,y);
		glColor3d(1.0,1.0,1.0); glVertex2f(x+w,y);
		glColor3d(1.0,1.0,1.0); glVertex2f(x+w,y+h);
		glColor3d(0.7,0.95,1.0); glVertex2f(x,y+h);
	glEnd();

	// arrows
	glBegin(GL_QUADS);
		glColor4d(0.5,0.5,0.5,0.0); glVertex2f(x,y+15.0);
		glColor4d(0.5,0.5,0.5,0.0); glVertex2f(x+w,y+15.0);
		glColor4d(0.5,0.5,0.5,1.0); glVertex2f(x+w,y);
		glColor4d(0.5,0.5,0.5,1.0); glVertex2f(x,y);
		glColor4d(0.5,0.5,0.5,0.0); glVertex2f(x,y+h-15.0);
		glColor4d(0.5,0.5,0.5,0.0); glVertex2f(x+w,y+h-15.0);
		glColor4d(0.5,0.5,0.5,1.0); glVertex2f(x+w,y+h);
		glColor4d(0.5,0.5,0.5,1.0); glVertex2f(x,y+h);
	glEnd();
	glBegin(GL_TRIANGLES);
		glColor3d(0.7,0.7,0.7); glVertex2f(x+21,y+h-10);
		glColor3d(0.1,0.1,0.1); glVertex2f(x+29,y+h-10);
		glColor3d(0.0,0.0,0.0); glVertex2f(x+25,y+h-3);
		glColor3d(0.7,0.7,0.7); glVertex2f(x+21,y+10);
		glColor3d(0.1,0.1,0.1); glVertex2f(x+29,y+10);
		glColor3d(0.0,0.0,0.0); glVertex2f(x+25,y+3);
	glEnd();

	// border
	glBegin(GL_LINES);
		glColor3d(0.8,0.8,0.8);
		glVertex2f(x+w-1,y);
		glVertex2f(x+w-1,y+h);
		glColor3d(0.3,0.3,0.3);
		glVertex2f(x+w,y);
		glVertex2f(x+w,y+h);
	glEnd();

}

void QAniTemplateBox::press(float, float py) {
	int idx = floor((py - y - 15 + offsety)/35);
	dragging = true;
	if (py <= y+15.0) { // top arrow
		scroll = 1;
		QTimer::singleShot(30, this, SLOT(scrollTimer()));
	}
	else if (py >= y+h-15.0) { //bottom arrow
		scroll = -1;
		QTimer::singleShot(30, this, SLOT(scrollTimer()));
	}
	else if (idx >=0 && idx < templates.size()) {
		selectedIdx = idx;
		dragging = false;
	}
}

void QAniTemplateBox::drag(float, float) {
}

void QAniTemplateBox::hover(float px, float py) {
	int idx = floor((py - y - 15 + offsety)/35);
	if (idx >=0 && idx < templates.size() && px>=10 && px<=40) setCursor(Qt::PointingHandCursor);
	else setCursor(Qt::ArrowCursor);
	updateGL();
}

void QAniTemplateBox::release(float px, float py) {
	int idx = floor((py - y - 15 + offsety)/35);
	scroll = 0;
	dragging = false;
	if (idx >=0 && idx < templates.size() && idx == selectedIdx) {
		setDragging(idx);
		getTemplateDrag()->resetDragging(px, py);
		timelineGL->grabMouse();
		updateGL();
	}
}

void QAniTemplateBox::wheel(float, float, int delta) {
	offsety += (delta>0?-10:10);
	checkOffsety();
	updateGL();
}

void QAniTemplateBox::scrollTimer() {
	offsety -= (float)scroll;
	if (scroll > 0) scroll++;
	else if (scroll < 0) scroll--;
	if (!checkOffsety()) scroll = 0;

	updateGL();
	if (scroll) QTimer::singleShot(30, this, SLOT(scrollTimer()));
}

bool QAniTemplateBox::checkOffsety() {
	if (offsety < -5 || templates.size() * 35 + 5 <= h - 30) {
		offsety = -5;
		return false;
	}
	else if (offsety > (templates.size()) * 35 + 5 - h + 30) {
		offsety = (templates.size()) * 35 + 5 - h + 30;
		return false;
	}
	return true;
}

void QAniTemplateBox::resize() {
	setRect(0,100,50,timelineGL->height()-100);
	checkOffsety();
}

/* class QAniTemplate ****************************************************************************************************
 *************************************************************************************************************************
 *
 */

QAniTemplate::QAniTemplate(Type t, QWidget *parent) : QAniClickable(parent), type(t), tbox(NULL) {
	QPixmap *q;
	switch(t) {
	case SpacialOverview:
		q = new QPixmap(":/icons/SpacialOverview.png");
		break;
	case TemporalOverview:
		q = new QPixmap(":/icons/TemporalOverview.png");
		break;
	case TransferFunctionOverview:
		q = new QPixmap(":/icons/TransferFunctionOverview.png");
		break;
	}
	tex = timelineGL->bindTexture(*q);
}
/* draw template icons in template box
   offset - when scrolling the templatebox, top and bottom template icons sometimes can only be 
            shown partly.
            - negative offset: top icon
            + positive offset: bottom icon */
void QAniTemplate::draw(float offset) {
	bool drawtop = offset > 0 ? false : true;
	float drawoffset = offset > 0 ? offset : -offset;

	if (drawtop) drawoffset -= 5.0;
	if (drawtop && drawoffset < 0.0) return;
	if (!drawtop && drawoffset > 30.0) drawoffset = 30.0;

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, tex);

	glBegin(GL_QUADS);
	if (drawtop) {
		glTexCoord2f(0.0,drawoffset/30.0); glVertex2f(0.0,0.0);
		glTexCoord2f(0.0,0.0); glVertex2f(0.0,drawoffset);
		glTexCoord2f(1.0,0.0); glVertex2f(30.0,drawoffset);
		glTexCoord2f(1.0,drawoffset/30.0); glVertex2f(30.0,0.0);
	}
	else {
		glTexCoord2f(0.0,1.0); glVertex2f(0.0,0.0);
		glTexCoord2f(0.0,1.0-drawoffset/30.0); glVertex2f(0.0,drawoffset);
		glTexCoord2f(1.0,1.0-drawoffset/30.0); glVertex2f(30.0,drawoffset);
		glTexCoord2f(1.0,1.0); glVertex2f(30.0,0.0);
	}
	glEnd();
	glDisable(GL_TEXTURE_2D);
}	

void QAniTemplate::drawDragging() {
	float wy;
	switch (type) {
	case SpacialOverview:
		wy = 120;
		break;
	case TemporalOverview:
		wy = 160;
		break;
	case TransferFunctionOverview:
		wy = 200;
		break;
	}

	if (isFirstSet()) {
		float tx = tbox->scaler->WtoT(x);
		float wx = tbox->scaler->TtoW(fx);
		float ax = tbox->scaler->TtoW((tx+fx)/2);


		glColor4d(0.0,0.0,0.0,0.3);
		glBegin(GL_LINES);
			glVertex2f(x,100);
			glVertex2f(x,timelineGL->height()-20.0);
			glVertex2f(wx,100);
			glVertex2f(wx,timelineGL->height()-20.0);
			if (wx - x > 40 || wx - x < -40) {
				glVertex2f(wx,wy);
				glVertex2f(ax + (wx-x>0?20.0:-20.0),wy);
				glVertex2f(x,wy);
				glVertex2f(ax - (wx-x>0?20.0:-20.0),wy);
			}
		glEnd();

		glColor4d(0.8,0.95,1.0,0.3);
		glBegin(GL_QUADS);
			glVertex2f(x,wy-20.0);
			glVertex2f(x,wy+20.0);
			glVertex2f(wx,wy+20.0);
			glVertex2f(wx,wy-20.0);
		glEnd();

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, tex);

		glBegin(GL_QUADS);
			glTexCoord2f(0.0,1.0); glVertex2f(ax-15.0,wy-15.0);
			glTexCoord2f(0.0,0.0); glVertex2f(ax-15.0,wy+15.0);
			glTexCoord2f(1.0,0.0); glVertex2f(ax+15.0,wy+15.0);
			glTexCoord2f(1.0,1.0); glVertex2f(ax+15.0,wy-15.0);
		glEnd();
		glDisable(GL_TEXTURE_2D);
	}
	else {
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

		glColor4d(1.0,1.0,1.0,0.5);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, tex);

		glBegin(GL_QUADS);
			glTexCoord2f(0.0,1.0); glVertex2f(x-15.0,wy-15.0);
			glTexCoord2f(0.0,0.0); glVertex2f(x-15.0,wy+15.0);
			glTexCoord2f(1.0,0.0); glVertex2f(x+15.0,wy+15.0);
			glTexCoord2f(1.0,1.0); glVertex2f(x+15.0,wy-15.0);
		glEnd();
		glDisable(GL_TEXTURE_2D);

		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		if (dragInTimeline) {
			glColor4d(0.0,0.0,0.0,0.3);
			glBegin(GL_LINES);
				glVertex2f(x,100.0);
				glVertex2f(x,timelineGL->height()-20.0);
			glEnd();
		}
	}
	timelineGL->renderText(x,10.0,0.0,QString::number(tbox->scaler->WtoT(x), 'f', 2));

}

void QAniTemplate::drag(float px, float py) {
	x = px;
	y = py;

	float dx = 0;
	if (x > timelineGL->width()-5) dx = x - timelineGL->width() + 5;
	else if (x < 55) dx = x - 55;
	dx /= tbox->scaler->scale * 10;
	tbox->scaler->scrollDx = dx;

	if (x >= 55 && x <= timelineGL->width()-5) dragInTimeline = true; // first into timeline

	if (dragInTimeline && (x > timelineGL->width()-5 || x < 55)) {
		if (!tbox->scaler->offsetTimer->isActive()) tbox->scaler->offsetTimer->start(30);
	}
	else {
		if (tbox->scaler->offsetTimer->isActive()) tbox->scaler->offsetTimer->stop();
	}
}

void QAniTemplate::release(float px, float) {
	if (tbox->scaler->offsetTimer->isActive()) tbox->scaler->offsetTimer->stop();
	if (fx > tbox->scaler->WtoT(px)) {
		emit setTemplate(type, tbox->scaler->WtoT(px), fx, tex);
	}
	else if (tbox->scaler->WtoT(px) > fx) {
		emit setTemplate(type, fx, tbox->scaler->WtoT(px), tex);
	}
}

void QAniTemplate::resetDragging(float px, float py) {
	x = px;
	y = py;
	first = false;
	dragInTimeline = false;
}

void QAniTemplate::setFirstPoint(float px) {
	fx = tbox->scaler->WtoT(px);
	first = true;
}

/* class QAniTimeline ****************************************************************************************************
 *************************************************************************************************************************
 *
 */
QAniTimeline::QAniTimeline(QWidget * parent) : QAniClickable(parent) {
	cursorx = 0.0;
	currentx = 0.0;
	draggingInstance = false;
	draggingCurrentTime = false;
	draggingKeyframe = false;

	hoverCameraKeyframeKey = hoverTemporalKeyframeKey = hoverTransferFunctionKeyframeKey = hoverSliceKeyframeKey = false;
	pressedCameraKeyframeKey = pressedTemporalKeyframeKey = pressedTransferFunctionKeyframeKey = pressedSliceKeyframeKey = false;

	playingTimer = new QTimer(this);
	playingTimer->setInterval(30);	
	connect(playingTimer, SIGNAL(timeout()), this, SLOT(playingTimeout()));
}

void QAniTimeline::draw() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0,timelineGL->width(),0,timelineGL->height(),-1,1); // 1 unit of OpenGL = 1 pixel
	glScalef(1.0,-1.0,1.0);
	glTranslatef(0.0,-timelineGL->height(),0.0);

	drawBackground();

	// draw cursor pointed time
	//if (!dragging) drawCursor();

	// draw current time
	drawCurrent();

	// draw instances
	for (int i = 0; i < timelineGL->itm[QAniInstance::ALL].size(); ++i) {
		drawInstance(timelineGL->itm[QAniInstance::ALL][i]);
	}

	// draw keyframes
	m_snapshotPos.clear();
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	for (int i = 0; i < timelineGL->kfm[QAniKeyframe::ALL].size(); ++i) {
		drawKeyframe(timelineGL->kfm[QAniKeyframe::ALL][i]);
	}

	// draw instance
	for (unsigned int i = 0; i < instances.size(); ++i) {
		instances[i]->draw();
	}

	// draw dragging time step
	if (tbox->isDragging()) {
	}

}

void QAniTimeline::drawBackground() {
	glColor3d(0.9,0.9,0.9);
	glBegin(GL_LINES);
	glVertex2f(50,100);
	glVertex2f(timelineGL->width(), 100);
	glEnd();

	// Camera Key
	if (hoverCameraKeyframeKey) glColor3d(0.2,0.2,0.2);
	timelineGL->renderText(50,116.0,0.0,QString("Camera"),QFont("Arial", 16));

	glColor3d(0.9,0.9,0.9);
	glBegin(GL_LINES);
	glVertex2f(50,140);
	glVertex2f(timelineGL->width(), 140);
	glEnd();

	// Temporal Key
	if (hoverTemporalKeyframeKey) glColor3d(0.2,0.2,0.2);
	timelineGL->renderText(50,156.0,0.0,QString("Temporal"),QFont("Arial", 16));

	glColor3d(0.9,0.9,0.9);
	glBegin(GL_LINES);
	glVertex2f(50,180);
	glVertex2f(timelineGL->width(), 180);
	glEnd();

	// Transfer Function Key
	if (hoverTransferFunctionKeyframeKey) glColor3d(0.2,0.2,0.2);
	timelineGL->renderText(50,196.0,0.0,QString("Transfer"),QFont("Arial", 16));
	timelineGL->renderText(50,216.0,0.0,QString("Function"),QFont("Arial", 16));

	glColor3d(0.9,0.9,0.9);
	glBegin(GL_LINES);
	glVertex2f(50,220);
	glVertex2f(timelineGL->width(), 220);
	glEnd();

	// Slice Key
	if (hoverSliceKeyframeKey) glColor3d(0.2,0.2,0.2);
	timelineGL->renderText(50,236.0,0.0,QString("Slice"),QFont("Arial", 16));

	glColor3d(0.9,0.9,0.9);
	glBegin(GL_LINES);
	glVertex2f(50,260);
	glVertex2f(timelineGL->width(), 260);
	glEnd();

}

void QAniTimeline::drawCursor() {
	glColor3d(0.9,0.9,0.9);
	glBegin(GL_LINES);
	glVertex2f(cursorx, 100.0);
	glVertex2f(cursorx, timelineGL->height() - 20.0);
	glEnd();
}

void QAniTimeline::drawCurrent() {
	glColor3d(0.0,0.0,0.0);
	glBegin(GL_LINES);
	glVertex2f(scaler->TtoW(currentx), 100.0);
	glVertex2f(scaler->TtoW(currentx), timelineGL->height() - 20.0);
	glEnd();
}

void QAniTimeline::drawInstance(QAniInstance * instance) {
	float y;
	Vector3 topleft, mid, bottomright;
	switch (instance->getType()) {
	case QAniInstance::Camera:
		topleft = Vector3(1.0,1.0,1.0);
		mid = Vector3(1.0,0.0,0.0);
		bottomright = Vector3(0.4,0.0,0.0);
		y = 100.0;
		break;
	case QAniInstance::Temporal:
		topleft = Vector3(1.0,1.0,1.0);
		mid = Vector3(0.0,0.0,1.0);
		bottomright = Vector3(0.0,0.0,0.4);
		y = 140.0;
		break;
	case QAniInstance::TransferFunction:
		topleft = Vector3(1.0,1.0,1.0);
		mid = Vector3(0.0,1.0,0.0);
		bottomright = Vector3(0.0,0.4,0.0);
		y = 180.0;
		break;
	case QAniInstance::Slice:
		topleft = Vector3(1.0,1.0,1.0);
		mid = Vector3(1.0,0.8,0.2);
		bottomright = Vector3(0.4,0.3,0.05);
		y = 220.0;
		break;
	}

	glBegin(GL_QUADS);
	glColor4d(unpack3(topleft), 0.1); glVertex2f(scaler->TtoW(instance->getStartTime()), y + 1.0);
	glColor4d(unpack3(mid), 0.1); glVertex2f(scaler->TtoW(instance->getStartTime()), y + 40.0);
	glColor4d(unpack3(bottomright), 0.1); glVertex2f(scaler->TtoW(instance->getEndTime()), y + 40.0);
	glColor4d(unpack3(mid), 0.1); glVertex2f(scaler->TtoW(instance->getEndTime()), y + 1.0);
	glEnd();

	switch (instance->getType()) {
	case QAniInstance::Camera:
		glColor4d(1.0,0.0,0.0,1.0);
		break;
	case QAniInstance::Temporal:
		glColor4d(0.0,0.0,1.0,1.0);
		break;
	case QAniInstance::TransferFunction:
		glColor4d(0.0,1.0,0.0,1.0);
		break;
	case QAniInstance::Slice:
		glColor4d(1.0,0.8,0.2,1.0);
		break;
	}
	glBegin(GL_LINE_STRIP);
	glColor4d(unpack3(mid), 1.0); glVertex2f(scaler->TtoW(instance->getStartTime()) + 1.0, y + 1.0);
	glColor4d(unpack3(mid), 1.0); glVertex2f(scaler->TtoW(instance->getEndTime()) - 1.0, y + 1.0);
	glColor4d(unpack3(bottomright), 1.0); glVertex2f(scaler->TtoW(instance->getEndTime()) - 1.0, y + 39.0);
	glColor4d(unpack3(bottomright), 1.0); glVertex2f(scaler->TtoW(instance->getStartTime()) + 1.0, y + 39.0);
	glColor4d(unpack3(mid), 1.0); glVertex2f(scaler->TtoW(instance->getStartTime()) + 1.0, y + 1.0);
	glEnd();

	if (instance->getTexture() >= 0) {
		float x = (scaler->TtoW(instance->getStartTime()) + scaler->TtoW(instance->getEndTime()))/2.0;
		float d = scaler->TtoW(instance->getEndTime()) - scaler->TtoW(instance->getStartTime());
		if (d > 30.0) d = 30.0;
		d /= 2.0;
		y += 20.0;
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, instance->getTexture());
		glBegin(GL_QUADS);
		glTexCoord2f(0.0,1.0); glVertex2f(x - d, y - d);
		glTexCoord2f(0.0,0.0); glVertex2f(x - d, y + d);
		glTexCoord2f(1.0,0.0); glVertex2f(x + d, y + d);
		glTexCoord2f(1.0,1.0); glVertex2f(x + d, y - d);
		glEnd();
		glDisable(GL_TEXTURE_2D);

		if (d == 15.0) {
			glColor4d(0.0,0.0,0.0,0.2);
			glBegin(GL_LINES);
			glVertex2f(scaler->TtoW(instance->getStartTime()) + 1.0, y);
			glVertex2f(x - d, y);
			glVertex2f(scaler->TtoW(instance->getEndTime()) - 1.0, y);
			glVertex2f(x + d, y);
			glEnd();
		}
	}
}

void QAniTimeline::drawKeyframe(QAniKeyframe * keyframe) {
	float x = scaler->TtoW(keyframe->getTime());
	float y;
	switch (keyframe->getType()) {
	case QAniKeyframe::Camera:
		glColor3d(1.0,0.0,0.0);
		y = 100.0;
		break;
	case QAniKeyframe::Temporal:
		glColor3d(0.0,0.0,1.0);
		y = 140.0;
		break;
	case QAniKeyframe::TransferFunction:
		glColor3d(0.0,1.0,0.0);
		y = 180.0;
		break;
	case QAniKeyframe::Slice:
		glColor3d(1.0,0.8,0.2);
		y = 220.0;
		break;
	}
	glLineWidth(1.0);
	glBegin(GL_LINES);
	glVertex2f(x, y + 1.0);
	glVertex2f(x, y + 40.0);
	glColor4d(0.0,0.0,0.0,0.1);
	for (float i = y; i > 80.0; i -= 10.0) {
		glVertex2f(x, i);
		glVertex2f(x, i - 5.0);
	}
	glEnd();
	glLineWidth(1.0);

	bool ovp = false;
	for (QList<float>::iterator it = m_snapshotPos.begin(); it != m_snapshotPos.end(); ++it) {
		if (x - *it < 60.0 && x - *it > -60.0) {
			ovp = true;
			break;
		}
	}
	if (!ovp) {
		float r = (float)keyframe->getSnapshot().width() / (float)keyframe->getSnapshot().height();
		float w, h;
		if (r > 1.0) { w = 1/r/2; h = 0.5; }
		else { h = 1/r/2; w = 0.5; }
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, keyframe->getTexture());
		glBegin(GL_QUADS);
		glTexCoord2f(0.5-w,0.5-h); glVertex2f(x-30.0,20.0);
		glTexCoord2f(0.5-w,0.5+h); glVertex2f(x-30.0,80.0);
		glTexCoord2f(0.5+w,0.5+h); glVertex2f(x+30.0,80.0);
		glTexCoord2f(0.5+w,0.5-h); glVertex2f(x+30.0,20.0);
		glEnd();
		glDisable(GL_TEXTURE_2D);

		m_snapshotPos.push_back(x);
	}
}

void QAniTimeline::resize() {
	setRect(51, 0, timelineGL->width()-51, timelineGL->height()-30);
}

void QAniTimeline::hover(float px, float py) {
	setCursor(Qt::ArrowCursor);

	hoverCameraKeyframeKey = hoverTemporalKeyframeKey = hoverTransferFunctionKeyframeKey = hoverSliceKeyframeKey = false;
	if (py > 100 && py <= 140 && px < 150) hoverCameraKeyframeKey = true;
	else if (py > 140 && py <= 180 && px < 150) hoverTemporalKeyframeKey = true;
	else if (py > 180 && py <= 220 && px < 150) hoverTransferFunctionKeyframeKey = true;
	else if (py > 220 && py <= 260 && px < 150) hoverSliceKeyframeKey = true;

	// hover on an instance
	for (int i = 0; i < timelineGL->itm[QAniInstance::ALL].size(); ++i) {
		float winstartx = scaler->TtoW(timelineGL->itm[QAniInstance::ALL][i]->getStartTime());
		float winendx = scaler->TtoW(timelineGL->itm[QAniInstance::ALL][i]->getEndTime());
		QAniInstance::Type type = timelineGL->itm[QAniInstance::ALL][i]->getType();
		if (px >= winstartx + 1 && px <= winendx - 1) {
			if ((type == QAniInstance::Camera && py > 100 && py <= 140) ||
			    (type == QAniInstance::Temporal && py > 140 && py < 180) ||
			    (type == QAniInstance::TransferFunction && py > 180 && py < 220) ||
			    (type == QAniInstance::Slice && py > 220 && py < 260)) {
				setCursor(Qt::OpenHandCursor);
				break;
			}

		}
	}

	// hover on a keyframe
	for (int i = 0; i < timelineGL->kfm[QAniKeyframe::ALL].size(); ++i) {
		float winx = scaler->TtoW(timelineGL->kfm[QAniKeyframe::ALL][i]->getTime());
		QAniKeyframe::Type type = timelineGL->kfm[QAniKeyframe::ALL][i]->getType();
		if (px >= winx-1 && px <= winx+1) {
			if ((type == QAniKeyframe::Camera && py > 100 && py <= 140) ||
			    (type == QAniKeyframe::Temporal && py > 140 && py < 180) ||
			    (type == QAniKeyframe::TransferFunction && py > 180 && py < 220) ||
			    (type == QAniKeyframe::Slice && py > 220 && py < 260)) {
				setCursor(Qt::SplitHCursor);
				break;
			}
		}
	}

	cursorx = px;
	updateGL();
}

void QAniTimeline::press(float px, float py) {
	if (tbox->isDragging()) {
		tbox->getTemplateDrag()->setFirstPoint(px);
		updateGL();
		return;
	}

	if (py > 100 && py <= 140 && px < 150) {
		pressedCameraKeyframeKey = true;
		return;
	}
	else if (py > 140 && py <= 180 && px < 150) {
		pressedTemporalKeyframeKey = true;
		return;
	}
	else if (py > 180 && py <= 220 && px < 150) {
		pressedTransferFunctionKeyframeKey = true;
		return;
	}
	else if (py > 220 && py <= 260 && px < 150) {
		pressedSliceKeyframeKey = true;
		return;
	}

	for (int i = 0; i < instances.size(); ++i) {
		instances[i]->setSelected(false);
		if (instances[i]->encloses(px, py)) {
			instances[i]->press(px, py);
			draggingInstance = true;
		}
	}
	for (int i = timelineGL->kfm[QAniKeyframe::ALL].size() - 1; i >= 0; --i) {
		float winx = scaler->TtoW(timelineGL->kfm[QAniKeyframe::ALL][i]->getTime());
		QAniKeyframe::Type type = timelineGL->kfm[QAniKeyframe::ALL][i]->getType();
		if (px >= winx-1 && px <= winx+1) {
			if ((type == QAniKeyframe::Camera && py > 100 && py <= 140) ||
			    (type == QAniKeyframe::Temporal && py > 140 && py < 180) ||
			    (type == QAniKeyframe::TransferFunction && py > 180 && py < 220) ||
			    (type == QAniKeyframe::Slice && py > 220 && py < 260) ) {
				draggingKeyframe = true;
				draggingStartTime = timelineGL->kfm[QAniKeyframe::ALL][i]->getTime();
				draggingKeyframeIdx = i;
				break;
			}
		}
	}
	for (int i = timelineGL->itm[QAniInstance::ALL].size() - 1; i >= 0; --i) {
		float winstartx = scaler->TtoW(timelineGL->itm[QAniInstance::ALL][i]->getStartTime());
		float winendx = scaler->TtoW(timelineGL->itm[QAniInstance::ALL][i]->getEndTime());
		QAniInstance::Type type = timelineGL->itm[QAniInstance::ALL][i]->getType();
		if (px > winstartx + 1 && px < winendx - 1) {
			if ((type == QAniInstance::Camera && py > 100 && py <= 140) ||
			    (type == QAniInstance::Temporal && py > 140 && py < 180) ||
			    (type == QAniInstance::TransferFunction && py > 180 && py < 220) ||
			    (type == QAniInstance::Slice && py > 220 && py < 260) ) {
				draggingInstance = true;
				draggingInstanceTime = scaler->WtoT(px) - timelineGL->itm[QAniInstance::ALL][i]->getStartTime();
				draggingStartTime = timelineGL->itm[QAniInstance::ALL][i]->getStartTime();
				draggingInstanceIdx = i;
				break;
			}
		}
	}
	if (!draggingInstance && !draggingKeyframe) {
		currentx = scaler->WtoT(px);
		emit currentTimeChange();
		draggingCurrentTime = true;
	}

	opx = px;
	opy = py;
	dragging = true;
}

void QAniTimeline::drag(float px, float py) {
	if (draggingKeyframe) {
		float t = scaler->WtoT(px);
		QAniKeyframe * pkf = timelineGL->kfm.findPrevKeyframe(timelineGL->kfm[QAniKeyframe::ALL][draggingKeyframeIdx]->getTime(), timelineGL->kfm[QAniKeyframe::ALL][draggingKeyframeIdx]->getType());
		QAniKeyframe * nkf = timelineGL->kfm.findNextKeyframe(timelineGL->kfm[QAniKeyframe::ALL][draggingKeyframeIdx]->getTime(), timelineGL->kfm[QAniKeyframe::ALL][draggingKeyframeIdx]->getType());
		if (pkf) if (t < pkf->getTime() + 0.03) t = pkf->getTime() + 0.03;
		if (nkf) if (t > nkf->getTime() - 0.03) t = nkf->getTime() - 0.03;
		timelineGL->kfm[QAniKeyframe::ALL][draggingKeyframeIdx]->setTime(t);
		emit currentTimeChange();
		updateGL();
	}
	else if (draggingInstance) {
		QAniInstance * instance = timelineGL->itm[QAniInstance::ALL][draggingInstanceIdx];
		float newstarttime = scaler->WtoT(px) - draggingInstanceTime;
		QAniKeyframe * pkf = timelineGL->kfm.findPrevKeyframe(instance->getStartTime(), instance->getStart()->getType());
		if (pkf) if (newstarttime < pkf->getTime() + 0.03) newstarttime = pkf->getTime() + 0.03;
		timelineGL->kfm.translateAfter(instance->getStart()->getType(), instance->getStartTime(), newstarttime - instance->getStartTime());
		updateGL();
	}
	else if (draggingCurrentTime) {
		currentx = scaler->WtoT(px);
		emit currentTimeChange();
		updateGL();
	}
	opx = px;
	opy = py;
}

void QAniTimeline::release(float px, float py) {
	if (tbox->isDragging()) {
		tbox->getTemplateDrag()->release(px, py);
		tbox->clearDragging();
		timelineGL->releaseMouse();
		updateGL();
		return;
	}

	if (pressedCameraKeyframeKey) {
		pressedCameraKeyframeKey = false;
		if (py > 100 && py <= 140 && px < 150) {
			emit setKeyframe(currentx, QAniKeyframe::Camera);
		}
	}
	else if (pressedTemporalKeyframeKey) {
		pressedTemporalKeyframeKey = false;
		if (py > 140 && py <= 180 && px < 150) {
			emit setKeyframe(currentx, QAniKeyframe::Temporal);
		}
	}
	else if (pressedTransferFunctionKeyframeKey) {
		pressedTransferFunctionKeyframeKey = false;
		if (py > 180 && py <= 220 && px < 150) {
			emit setKeyframe(currentx, QAniKeyframe::TransferFunction);
		}
	}
	else if (pressedSliceKeyframeKey) {
		pressedSliceKeyframeKey = false;
		if (py > 220 && py <= 260 && px < 150) {
			emit setKeyframe(currentx, QAniKeyframe::Slice);
		}
	}

	if (draggingInstance) {
		draggingInstance = false;
		timelineGL->updateKeyframeAfter(draggingStartTime < timelineGL->itm[QAniInstance::ALL][draggingInstanceIdx]->getStartTime()?
						draggingStartTime : timelineGL->itm[QAniInstance::ALL][draggingInstanceIdx]->getStartTime());
	}
	else if (draggingKeyframe) {
		draggingKeyframe = false;
		timelineGL->updateKeyframeAfter(draggingStartTime < timelineGL->kfm[QAniKeyframe::ALL][draggingKeyframeIdx]->getTime()?
						draggingStartTime : timelineGL->kfm[QAniKeyframe::ALL][draggingKeyframeIdx]->getTime());
	}
	else if (draggingCurrentTime) {
		currentx = scaler->WtoT(px);
		emit currentTimeChange();
		cursorx = px;
		updateGL();
		draggingCurrentTime = false;
	}
	else {
	}

	dragging = false;
}

void QAniTimeline::playButtonHit() {
	m_lastTime = QTime::currentTime().second() * 1000 + QTime::currentTime().msec();
	if (playingTimer->isActive()) playingTimer->stop();
	else playingTimer->start();
}

void QAniTimeline::recordButtonHit() {
	emit setKeyframe(currentx);
}

void QAniTimeline::playingTimeout() {
	int currentTime = QTime::currentTime().second() * 1000 + QTime::currentTime().msec();
	if (currentTime < m_lastTime) currentTime += 60000;
	currentx += (float)(currentTime - m_lastTime) / 1000.0;
	emit currentTimeChange();
	m_lastTime = QTime::currentTime().second() * 1000 + QTime::currentTime().msec();
	updateGL();
}

/* class QAniTimelineInstance ********************************************************************************************
 *************************************************************************************************************************
 *
 */

QAniTimelineInstance::QAniTimelineInstance(float tx, float ty, QAniTemplate::Type, QAniTimeline * tl, QWidget * parent) : QAniClickable(parent), timeline(tl) {
	x = tx;
	y = ty;
}

void QAniTimelineInstance::draw() {
	float wx = timeline->scaler->TtoW(x);
	float wy = y;

	glColor3d(0.8,0.8,0.8);
	glBegin(GL_LINES);
		glVertex2f(wx, wy+15.0);
		glVertex2f(wx, timelineGL->height() - 20.0);
	glEnd();

/*	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, tex);
	glBegin(GL_QUADS);
		glTexCoord2f(0.0,1.0); glVertex2f(wx-15.0,wy-15.0);
		glTexCoord2f(0.0,0.0); glVertex2f(wx-15.0,wy+15.0);
		glTexCoord2f(1.0,0.0); glVertex2f(wx+15.0,wy+15.0);
		glTexCoord2f(1.0,1.0); glVertex2f(wx+15.0,wy-15.0);
	glEnd();
	glDisable(GL_TEXTURE_2D);*/
}

void QAniTimelineInstance::press(float, float) {
	dragging = true;
}

void QAniTimelineInstance::drag(float dx, float dy) {
	x += dx/timeline->scaler->scale;
	y += dy;
	updateGL();
}

void QAniTimelineInstance::release(float, float) {
	dragging = false;
}

bool QAniTimelineInstance::encloses(float px, float py) {
	float wx = timeline->scaler->TtoW(x);
	float wy = y;
	return (px>=wx-15.0) && (px<=wx+15.0) && (py>=wy-15.0) && (py<=wy+15.0);
}
