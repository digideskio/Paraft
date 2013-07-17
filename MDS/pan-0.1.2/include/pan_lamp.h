/*******************************************************************************
 *
 * Copyright (C) 2011 Paulo Joia Filho
 * University of Sao Paulo - Sao Carlos/SP, Brazil.
 * All Rights Reserved.
 *
 * This file is part of Projection Analyzer (PAn).
 *
 * PAn is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * PAn is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public 
 * License for more details.
 *
 * This code was developed by Paulo Joia Filho <pjoia@icmc.usp.br>
 * at Institute of Mathematics and Computational Sciences - ICMC
 * University of Sao Paulo, Sao Carlos/SP, Brazil (http://www.icmc.usp.br)
 *
 * Contributor(s):  Luis Gustavo Nonato <gnonato@icmc.usp.br>
 *
 * You should have received a copy of the GNU Lesser General Public License 
 * along with PAn. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#ifndef PAN_LAMP_H_INCLUDED
#define PAN_LAMP_H_INCLUDED

#include "pan_useful.h"
#include "pan_math.h"
#include "pan_metric.h"

#ifdef DEBUG_LAMP
    #include "pan_dconv.h"
#endif

#include <time.h>

/******************************************************************************/
/*                         ENUMERATIONS / STRUCTURES                          */
/******************************************************************************/
struct lamp_struct {
    int numpoints;          // obligatory
    int numsamples;         // obligatory
    int highdim;            // obligatory
    int projdim;            // optional: DEFAULT_VALUE = 2
};

/******************************************************************************/
/*                                 FUNCTIONS                                  */
/******************************************************************************/
decimal lamp_elapsedtime();

int lamp_execute(decimal *inputdata, decimal *inputsampdata, decimal *inputsampproj,
                 struct lamp_struct *inputinfo, decimal* outputproj);


#endif // PAN_LAMP_H_INCLUDED
