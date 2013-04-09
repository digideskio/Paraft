//
// C++ Implementation: shadergroupbox
//
// Description: 
//
//
// Author: Chris Ho <csho@ucdavis.edu>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include "shadergroupbox.h"
#include <QShowEvent>

void ShaderGroupBox::showEvent(QShowEvent*) {
	emit shown();
}

