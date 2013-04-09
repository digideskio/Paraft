#ifndef _OPTIONGROUPBOX_H_
#define _OPTIONGROUPBOX_H_

#include <QGroupBox>
#include <QString>

class OptionGroupBox : public QGroupBox {
	Q_OBJECT
			
	
	protected:
		void showEvent(QShowEvent*);
	public:
		OptionGroupBox(const QString& name):QGroupBox(name) {}
	signals:
		void shown();
};

#endif

