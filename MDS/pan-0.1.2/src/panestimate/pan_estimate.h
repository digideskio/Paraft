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
#ifndef PAN_ESTIMATE_H_INCLUDED
#define PAN_ESTIMATE_H_INCLUDED

/******************************************************************************/
/*                                   INCLUDES                                 */
/******************************************************************************/
#include "pan_useful.h"
#include "pan_metric.h"
#include <stdlib.h>

/******************************************************************************/
/*                         ENUMERATIONS / STRUCTURES                          */
/******************************************************************************/
enum stress_enum {
    STRESS_KRUSKAL = 1,
    STRESS_NORMALIZED_KRUSKAL,        // default
    STRESS_PARTIAL_NORMALIZED_KRUSKAL,
    STRESS_SAMMON,
    STRESS_QUADRATIC
};
#define stress_enum_descr(index) ({                         \
    char *name[] = { "Kruskal's stress",                    \
                     "Normalized Kruskal's stress",         \
                     "Sampled normalized Kruskal's stress", \
                     "Sammnon's stress",                    \
                     "Quadratic stress"};                   \
    name[index-1];                                          \
})

struct stress_struct {
    int numpoints;                    // obligatory
    int highdim;                      // obligatory
    int projdim;                      // optional: DEFAULT_VALUE = 2
    enum dissimilarity_enum disstype; // optional: DEFAULT_VALUE = DISS_EUCLIDEAN
    enum stress_enum stresstype;      // optional: DEFAULT_VALUE = STRESS_NORMALIZED_KRUSKAL
};

/******************************************************************************/
/*                                   FUNCTIONS                                */
/******************************************************************************/
decimal stress_elapsedtime();

decimal* precision_curve_execute(decimal *highdata, int highdim, decimal* projdata,
                                 int projdim, int numpoints);

decimal stress_calc(decimal *highdata, decimal *projdata, struct stress_struct *stressinfo);


#endif // PAN_ESTIMATE_H_INCLUDED
