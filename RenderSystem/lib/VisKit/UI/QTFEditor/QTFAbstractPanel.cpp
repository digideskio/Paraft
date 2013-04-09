#include "QTFAbstractPanel.h"
#include "QTFEditor.h"
#include <QResizeEvent>
#include <QSize>
#include <QToolButton>
#include <QIcon>
#include <QPainter>

QTFAbstractPanel::QTFAbstractPanel(QWidget* parent):QWidget(parent), m_isHidden(false), m_isCollapsable(true),
								   m_checked(QString(":/arrow-collapsed.png")), m_unchecked(QString(":/arrow-expanded.png")), m_widgets(NULL) {
	m_pTFEditor = (QTFEditor*)parent;
	m_showButton = new QToolButton(this);
	m_showButton->setIconSize(QSize(20,20));
	m_showButton->setIcon(m_unchecked);
	m_showButton->setAutoRaise(true);
	connect(m_showButton, SIGNAL(clicked()),
			this, SLOT(toggleHide()));
	if(!m_isCollapsable)
		m_showButton->hide();
}

QTFAbstractPanel::~QTFAbstractPanel() {
	delete m_panelImage;
	delete m_showButton;
	if(m_widgets)
		delete m_widgets;
}

void QTFAbstractPanel::resizeEvent(QResizeEvent *event)
{
	m_panelWidth	= width() - m_panelLMargin - m_panelRMargin;
	m_panelHeight	= height() - m_panelUMargin - m_panelDMargin;
	if(m_panelImage->size() != event->size()) {
		delete m_panelImage;
		m_panelImage = new QImage(event->size(), QImage::Format_RGB32);
		m_panelImage->fill(m_backgroundColor.rgb());
	}
	updatePanelImage();
	QWidget::resizeEvent(event);
}

void QTFAbstractPanel::initLayout(int left, int top, int right, int bottom) {
// 	minSize = minimumSize();
// 	maxSize = maximumSize();
	m_panelLMargin	= left; // left
	m_panelRMargin	= right; // right
	m_panelUMargin	= top;  // up
	m_panelDMargin	= bottom;  // down
	m_panelWidth	= width() - m_panelLMargin - m_panelRMargin;
	m_panelHeight	= height() - m_panelUMargin - m_panelDMargin;	
	m_panelImage = new QImage(width(), height(),QImage::Format_RGB32);
	m_showButton->setGeometry(m_panelLMargin-22, m_panelUMargin, 20, 20);
	m_backgroundColor = QColor(255,255,255);
}

void QTFAbstractPanel::paintEvent(QPaintEvent* e) {
	if(!m_isHidden) {
		realPaintEvent(e);
		return;
	}
	
	paintHidden(e);
}

void QTFAbstractPanel::paintHidden(QPaintEvent*) {
	QPainter painter(this);
	painter.setPen(QColor(142, 158, 189));
	painter.fillRect(rect(),m_backgroundColor);
	painter.fillRect(QRect(m_panelLMargin, 3, m_panelWidth, 20),QColor(142, 158, 189));
	painter.setPen(QColor(255,255,255));
	painter.drawRect(QRect(m_panelLMargin+1, 4, m_panelWidth - 3, 14));
	painter.drawText(QRect(m_panelLMargin + 5, 5, m_panelWidth, 14), Qt::AlignLeft | Qt::AlignVCenter, m_panelName);
}


void QTFAbstractPanel::toggleHide() {
	m_isHidden = !m_isHidden;
	if(m_isHidden) {
		m_showButton->setIcon(m_checked);
		m_origHeight = height();
		m_origMinimumHeight = minimumHeight();
		if(m_widgets && !m_widgets->empty()) {
			for(int i = 0; i < m_widgets->size();i++) {
				m_origHeight += (*m_widgets)[i]->height();
				m_origMinimumHeight += (*m_widgets)[i]->minimumHeight();
				(*m_widgets)[i]->hide();
			}
		}
		minSize = minimumSize();
		maxSize = maximumSize();
		//m_pTFEditor->setMinimumHeight(0);
		setFixedHeight(20);
		m_pTFEditor->setMinimumHeight(m_pTFEditor->minimumHeight() - (m_origMinimumHeight - 20));
		m_pTFEditor->resize(m_pTFEditor->width(), m_pTFEditor->height() + 20 - m_origHeight);
	} else {
		m_showButton->setIcon(m_unchecked);
		if(m_widgets && !m_widgets->empty()) {
			for(int i = 0; i < m_widgets->size(); i++) {
				(*m_widgets)[i]->show();
			}
		}
		setMinimumSize(minSize);
		setMaximumSize(maxSize);
		m_pTFEditor->resize(m_pTFEditor->width(), m_pTFEditor->height() + m_origHeight - 20);
		m_pTFEditor->setMinimumHeight(m_pTFEditor->minimumHeight() + (m_origMinimumHeight - 20));
		//m_pTFEditor->setMinimumHeight(m_pTFEditor->minimumHeight() + (m_origHeight - 20));
	}
	repaint();
}

void QTFAbstractPanel::setCollapsable(bool collapsable) {
	m_isCollapsable = collapsable;
	if(m_isCollapsable)
		m_showButton->show();
	else
		m_showButton->hide();
}

void QTFAbstractPanel::setHideWith(QTFAbstractPanel* p) {
	p->setHideAlso(this);
}

void QTFAbstractPanel::setHideAlso(QWidget* p) {
	if(!m_widgets) {
		m_widgets = new QVector<QWidget*>();
	}
	m_widgets->append(p);
}

