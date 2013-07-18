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
/*
 * Based on code by Fernando V. Paulovich: PEx
 * available on http://infoserver.lcad.icmc.usp.br/infovis2/PEx
 * University of Sao Paulo, Sao Carlos/SP, Brazil
 */
#include "pan_estimate.h"

static decimal dissRn_max = - DEC_MAX;
static decimal dissRp_max = - DEC_MAX;
static decimal *highptI, *highptJ;
static decimal *projptI, *projptJ;

void dissmax_calc(decimal *highdata, decimal* projdata, int numpoints,
                  int highdim, int projdim, enum dissimilarity_enum disstype);


decimal kruskal_stress(decimal *highdata, decimal* projdata, int numpoints, int highdim,
            int projdim, enum dissimilarity_enum disstype, enum stress_enum stresstype)
{
    int i, j;
    decimal dissRn_norm, dissRp_norm;
    decimal num=0.0, den=0.0;

    highptI = (decimal*) malloc (highdim*sizeof(decimal));
    highptJ = (decimal*) malloc (highdim*sizeof(decimal));
    projptI = (decimal*) malloc (projdim*sizeof(decimal));
    projptJ = (decimal*) malloc (projdim*sizeof(decimal));

    // dissimilarity max
    if (stresstype == STRESS_NORMALIZED_KRUSKAL) {
        dissmax_calc(highdata, projdata, numpoints, highdim, projdim, disstype);
    }
    else {
        dissRn_max = 1.0;
        dissRp_max = 1.0;
    }

    // stress computation
    for (i=0; i < numpoints; i++)
    {
        decptrcpy(highptI, NULL_INT, NULL_INT, highdata, i*highdim, (i+1)*highdim-1);
        decptrcpy(projptI, NULL_INT, NULL_INT, projdata, i*projdim, (i+1)*projdim-1);

        for (j=i+1; j < numpoints; j++)
        {
            decptrcpy(highptJ, NULL_INT, NULL_INT, highdata, j*highdim, (j+1)*highdim-1);
            decptrcpy(projptJ, NULL_INT, NULL_INT, projdata, j*projdim, (j+1)*projdim-1);

            dissRn_norm = dissim_get3(highptI, highptJ, highdim, disstype) / dissRn_max;
            dissRp_norm = dissim_get3(projptI, projptJ, projdim, DISS_EUCLIDEAN) / dissRp_max;

            num += (dissRn_norm - dissRp_norm) * (dissRn_norm - dissRp_norm);
            den += dissRn_norm * dissRn_norm;
        }
    }

    // release memory
    free(highptI);
    free(highptJ);
    free(projptI);
    free(projptJ);

    return (num/den);
}

void dissmax_calc(decimal *highdata, decimal* projdata, int numpoints,
                  int highdim, int projdim, enum dissimilarity_enum disstype)
{
    dissRn_max = - DEC_MAX;
    dissRp_max = - DEC_MAX;

    int i, j;
    decimal dissRn, dissRp;

    for (i=0; i < numpoints; i++)
    {
        decptrcpy(highptI, NULL_INT, NULL_INT, highdata, i*highdim, (i+1)*highdim-1);
        decptrcpy(projptI, NULL_INT, NULL_INT, projdata, i*projdim, (i+1)*projdim-1);

        for (j=i+1; j < numpoints; j++)
        {
            decptrcpy(highptJ, NULL_INT, NULL_INT, highdata, j*highdim, (j+1)*highdim-1);
            decptrcpy(projptJ, NULL_INT, NULL_INT, projdata, j*projdim, (j+1)*projdim-1);

            dissRn = dissim_get3(highptI, highptJ, highdim, disstype);
            dissRp = dissim_get3(projptI, projptJ, projdim, DISS_EUCLIDEAN);

            if (dissRn > dissRn_max) {
                dissRn_max = dissRn;
            }

            if (dissRp > dissRp_max) {
                dissRp_max = dissRp;
            }
        }
    }
}
