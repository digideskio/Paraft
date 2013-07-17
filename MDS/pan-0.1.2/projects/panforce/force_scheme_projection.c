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
const boolean str_equal(const Object object_1, const Object object_2);
void iteration_calc(decimal *outputdata, int numpoints, int projdim, int *index,
     decimal *dissims, decimal fractiondelta, decimal *P, decimal *Q, decimal *R);


/******************************************************************************/
/*                               Functions Code                               */
/******************************************************************************/
const boolean str_equal(const Object object_1, const Object object_2)
{
    char *s1;
    char *s2;

    s1 = (char*) object_1;
    s2 = (char*) object_2;

    boolean result = !(strcmp(s1,s2));
    return result;
}

int force_execute(struct idmap_struct *info, decimal *dissims, decimal* outputdata)
{
    int numpoints = info->numpoints;
    int projdim = info->projdim;

    int i, j, ind;
    char *straux = (char*) malloc(10*sizeof(char));

    //Create the indexes and shuffle them
    arraylist index_aux = arraylist_create(str_equal);
    for (i=0; i < numpoints; i++)
    {
        sprintf(straux, "%d", i);
        arraylist_add(index_aux, strdup(straux));
    }

    int *index = (int*) malloc(numpoints*sizeof(int));
    for (ind=0, j=0; j < numpoints; ind += arraylist_size(index_aux) / 10, j++)
    {
        if (ind >= arraylist_size(index_aux))
            ind = 0;

        straux = (char*) arraylist_get(index_aux, ind);
        *(index+j) = atoi(straux);
        arraylist_remove(index_aux, straux);
    }

    free(straux);
    arraylist_free(index_aux);

    //if an entry is not provided, create one
    if (!outputdata)
    {
        int szinput = numpoints*projdim;
        outputdata = (decimal*) malloc (szinput*sizeof(decimal));

        // random function
        const gsl_rng_type * T;
        gsl_rng * r;

        gsl_rng_env_setup();

        T = gsl_rng_default;
        r = gsl_rng_alloc (T);

        long int ltime = time(NULL);
        gsl_rng_set(r, ltime);

        // fill the pointer
        for (i=0; i < numpoints; i++)
        {
            for (j=0; j < projdim; j++)
                *(outputdata+i*projdim+j) = gsl_rng_uniform_pos(r);
        }

        gsl_rng_free (r);

        #ifdef DEBUG_FORCE
            decptr_write("out/force_entry_random.out", outputdata, ",", numpoints, projdim, 1,
                         DEBUG_INFO("Random Numbers Resulting"));
        #endif
    }

    /* force-scheme */

    // auxiliary data structures
    decimal *P = (decimal*) malloc (projdim*sizeof(decimal));
    decimal *Q = (decimal*) malloc (projdim*sizeof(decimal));
    decimal *R = (decimal*) malloc (projdim*sizeof(decimal));

    for (i=0; i < info->numiterations; i++)
        iteration_calc(outputdata, numpoints, projdim, index, dissims,
                       info->fractiondelta, P, Q, R);

    // release memory
    free(P);
    free(Q);
    free(R);
    free(index);

    #ifdef DEBUG_FORCE
        decptr_write("out/force_output.out", outputdata, ",", numpoints, projdim, 1,
                     DEBUG_INFO("Force Resulting"));
    #endif

    return EXIT_SUCCESS;
}

void iteration_calc(decimal *outputdata, int numpoints, int projdim, int *index,
     decimal *dissims, decimal fractiondelta, decimal *P, decimal *Q, decimal *R)
{
    int i, j;
    int indexI, indexJ;
    decimal dissRn, dissRp, delta;

    //for each point
    for (i=0; i < numpoints; i++)
    {
        indexI = *(index+i);

        //for each other point
        for (j=0; j < numpoints; j++)
        {
            indexJ = *(index+j);
            if (indexI == indexJ) continue;

            // dissimilarity between projected points
            decptrcpy(P, NULL_INT, NULL_INT, outputdata, indexI*projdim, indexI*projdim+projdim-1);
            decptrcpy(Q, NULL_INT, NULL_INT, outputdata, indexJ*projdim, indexJ*projdim+projdim-1);
            vectors_subtract(R, Q, P, projdim);

            dissRp = euclid_norm(P, Q, projdim);
            if (dissRp < EPSILON) dissRp = EPSILON;

            dissRn = dissim_get(dissims, numpoints, indexI, indexJ);

            // Calculating the (fraction of) delta
            delta = dissRn - dissRp;
            delta /= fractiondelta;

            // moving Q -> P
            vector_multscalar(P, delta/dissRp, R, projdim);
            vectors_add(R, Q, P, projdim);

            decptrcpy(outputdata, indexJ*projdim, indexJ*projdim+projdim-1, R, NULL_INT, NULL_INT);
        }
    }
}

