#ifndef _QTFABSTRACTPANEL_H_
#define _QTFABSTRACTPANEL_H_

#include <QWidget>
#include <QIcon>
#include <QVector>

class QTFEditor;
class QToolButton;

class QTFAbstractPanel : public QWidget
{
	Q_OBJECT
	public:
		QTFAbstractPanel(QWidget *parent=0);
		~QTFAbstractPanel();

		void setCollapsable(bool);
		void setHideWith(QTFAbstractPanel*);
		void setHideAlso(QWidget*);
	protected:
		virtual void	initLayout()=0;
		void	initLayout(int,int,int,int);
	
		int		m_panelLMargin; // left
		int		m_panelRMargin; // right
		int		m_panelUMargin; // up
		int		m_panelDMargin; // down
		int		m_panelWidth;
		int		m_panelHeight;

		QImage			*m_panelImage;
		virtual void	updatePanelImage()=0;
		QColor			m_backgroundColor;
	
		void			paintEvent(QPaintEvent *);
		virtual void	realPaintEvent(QPaintEvent *)=0;
		virtual void	resizeEvent(QResizeEvent *);
		QString			m_panelName;
		QTFEditor		*m_pTFEditor;
		
	private:
		void			paintHidden(QPaintEvent *);
		bool			m_isHidden;
		bool 			m_isCollapsable;
		QToolButton*	m_showButton;
		QIcon			m_checked;
		QIcon m_unchecked;
		QSize minSize;
		QSize maxSize;
		int   m_origHeight;
		int	  m_origMinimumHeight;
		QVector<QWidget*>* m_widgets;
		
	public slots:
		void toggleHide();
};
#endif

