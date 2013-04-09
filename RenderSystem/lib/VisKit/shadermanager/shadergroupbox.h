//
// C++ Interface: shadergroupbox
//
// Description: stupid group box to emit a show event, le sigh.
//
//
// Author: Chris Ho <csho@ucdavis.edu>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef _SHADERGROUPBOX_H_
#define _SHADERGROUPBOX_H_

#include <QGroupBox>
#include <QString>

class ShaderGroupBox : public QGroupBox {
	Q_OBJECT
			
	
	protected:
		void showEvent(QShowEvent*);
	public:
		ShaderGroupBox(const QString& name):QGroupBox(name) {}
	signals:
		void shown();
};

#endif


