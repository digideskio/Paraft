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
#include "pan_force.h"

/******************************************************************************/
/*                        Internal Functions Prototypes                       */
/******************************************************************************/
int fastmap_execute(decimal *inputdata, char **inputid, struct idmap_struct *info,
                    decimal *dissims, decimal* outputdata);
int force_execute(struct idmap_struct *info, decimal *dissims,
                  decimal* outputdata);


/******************************************************************************/
/*                               Functions Code                               */
/******************************************************************************/
int idmap_execute(decimal *inputdata, char **inputid, struct idmap_struct *info,
                  decimal* outputdata)
{
    if (info->projdim == DEFAULT_VALUE) info->projdim = 2;
    if (info->numiterations == DEFAULT_VALUE) info->numiterations = 50;
    if (info->fractiondelta == DEFAULT_VALUE) info->fractiondelta = 8.0;
    if (info->disstype == DEFAULT_VALUE) info->disstype = DISS_EUCLIDEAN;
    if (info->inittype == DEFAULT_VALUE) info->inittype = IDMAP_FASTMAP;

    // calc dissimilarities
    int numdiss = (info->numpoints*(info->numpoints-1))/2;
    decimal *dissims = (decimal*) malloc (numdiss*sizeof(decimal));
    dissims_calc(inputdata, info->numpoints, info->highdim, info->disstype, dissims);

    if (info->inittype == IDMAP_FASTMAP)
    {
        fastmap_execute(inputdata, inputid, info, dissims, outputdata);
    }
    else if (info->inittype == IDMAP_RANDOM)
    {/*
        todo! */
    }
    else
    {/*
        todo! */
    }

    force_execute(info, dissims, outputdata);

    // release memory
    free(dissims);

    return EXIT_SUCCESS;
}
