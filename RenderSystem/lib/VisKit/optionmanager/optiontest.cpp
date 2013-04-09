#include "optiontest.h"
#include <QDialog>
#include <QVBoxLayout>
#include <QDialogButtonBox>

OptionTest::OptionTest():options(0) {
	om.addOptionb("checkbox");
	om.addOptionf("test", 1);
	om.addOptioni("test2", 3);
	om.addOptiond("test3", 4);
	
	om.b("checkbox")->setDisplayName("Checkbox test");
	om.b("checkbox")->setModifiable(true);
	
	om.f("test")->setDisplayName("Float test");
	om.f("test")->setModifiable(true);
	om.f("test")->setFieldNames("ohai");
	om.f("test")->setRanges(0, 1.);
	om.f("test")->setSteps(100);
	om.f("test")->setValues(0.5f);
	
	om.i("test2")->setDisplayName("Int test");
	om.i("test2")->setModifiable(true);
	om.i("test2")->setFieldNames("ohai", "ohbye", "omgwtf");
	om.i("test2")->setRanges(0, 10, 0, 5, 0, 15);
	om.i("test2")->setValues(0, 1, 2);
	om.i("test2")->setSteps(100);
	
	om.d("test3")->setDisplayName("Int test");
	om.d("test3")->setModifiable(true);
	om.d("test3")->setFieldNames("uurrrg", "urn", "urf", "urd");
	om.d("test3")->setRanges(0, 1., 0, 5., 0, 15., 0.1, 0.5);
	om.d("test3")->setValues(0.5, 0.2, 0.1, 0.3);
	om.d("test3")->setSteps(100);
}

void OptionTest::showOptions() {
	if(!options) {
		options = new QDialog(this);
		options->setWindowTitle("More Options");
		QVBoxLayout* l = new QVBoxLayout;
		QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
		connect(buttonBox, SIGNAL(accepted()), options, SLOT(accept()));
		connect(buttonBox, SIGNAL(rejected()), options, SLOT(reject()));
		l->addWidget(om.getOptions());
		l->addWidget(buttonBox);
		options->setLayout(l);
		connect(options, SIGNAL(accepted()),
			&om, SLOT(accepted()));
	}
	options->show();
}

void OptionTest::keyReleaseEvent(QKeyEvent*) {
	showOptions();
}

OptionTest::~OptionTest() {
	if(options)
		delete options;
}
