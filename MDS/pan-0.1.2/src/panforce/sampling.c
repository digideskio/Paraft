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
#include "pan_force.h"

int sampling_execute(decimal *inputdata, char **inputid, struct sampling_struct *info,
                     decimal *outputdata, char **outputid)
{
    if (info->disstype == DEFAULT_VALUE) info->disstype = DISS_EUCLIDEAN;
    if (info->sampletype == DEFAULT_VALUE) info->sampletype = SAMP_RANDOM;

    const gsl_rng_type * T;
    gsl_rng * r;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    long int ltime = time(NULL);
    gsl_rng_set(r, ltime);

    int i;
    unsigned long int u;
    for (i=0; i<info->numsamples; i++)
    {
        u = gsl_rng_uniform_int(r, info->numpoints);

        *(outputid+i) = (char*) malloc((strlen(*(inputid+u))+1)*sizeof(char));
        strcpy(*(outputid+i), *(inputid+u));

        decptrcpy(outputdata, i*info->highdim, NULL_INT, inputdata, u*info->highdim, (u+1)*info->highdim-1);
    }
    gsl_rng_free (r);

    #ifdef DEBUG_FORCE
        charptr_write("out/sampling_id.out", outputid, ",", info->numsamples, 1, 1,
                      DEBUG_INFO("Sample Id Random"));
        decptr_write("out/sampling_data.out", outputdata, ",", info->numsamples, info->highdim, 1,
                     DEBUG_INFO("Sample Data Id-based"));
    #endif

    return 0;
}




