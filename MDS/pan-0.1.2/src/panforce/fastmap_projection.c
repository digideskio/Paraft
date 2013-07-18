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
int fastmap_lessthan4pts(decimal *points, int numpoints, int projdim,
                         decimal *dissims);
int fastmap_computation(decimal *points, int numpoints, int projdim,
                        decimal *dissims);
int dissimilar_points_choose(decimal *dissims, int numpoints, int choosen[]);
int dissims_update(decimal *points, int numpoints, int projdim,
                   int currentdim, decimal *dissims);


/******************************************************************************/
/*                               Functions Code                               */
/******************************************************************************/
int fastmap_execute(decimal *inputdata, char **inputid, struct idmap_struct *info,
                    decimal *dissims, decimal* outputdata)
{
    int numpoints = info->numpoints;
    int projdim = info->projdim;

    // set all points values to zero
    decptrclr(outputdata, numpoints*projdim);

    // clone dissimilarities
    int numdiss = (numpoints*(numpoints-1))/2;
    decimal *dissims2 = (decimal*) malloc (numdiss*sizeof(decimal));
    decptrcpy(dissims2, NULL_INT, NULL_INT, dissims, 0, numdiss-1);

    if (numpoints < 4)
        fastmap_lessthan4pts(outputdata, numpoints, projdim, dissims2);
    else
        fastmap_computation(outputdata, numpoints, projdim, dissims2);

    #ifdef DEBUG_FORCE
        decptr_write("out/fastmap_output.out", outputdata, ",", numpoints, projdim, 1,
                     DEBUG_INFO("Fastmap Initialization Force"));
    #endif

    // release memory
    free(dissims2);

    return EXIT_SUCCESS;
}

int fastmap_lessthan4pts(decimal *points, int numpoints, int projdim, decimal *dissims)
{
    int pos=0;
    if (numpoints == 2)
    {
        pos = 1 * projdim + 0;
        *(points+pos) = dissim_get(dissims,numpoints,0,1);
    }
    else if (numpoints == 3)
    {
        pos = 1 * projdim + 0;
        *(points+pos) = dissim_get(dissims,numpoints,0,1);
        pos = 2 * projdim + 0;
        *(points+pos) = dissim_get(dissims,numpoints,0,1);
        pos = 2 * projdim + 1;
        *(points+pos) = dissim_get(dissims,numpoints,1,2);
    }
    return 0;
}

int fastmap_computation(decimal *points, int numpoints, int projdim, decimal *dissims)
{
    int currentdim=0;
    int lvchoosen[2];
    decimal dissval=0.0;
    decimal xi=0.0;
    int i, pos;

    while (currentdim < projdim)
    {
        // choosen pivots for this recursion
        dissimilar_points_choose(dissims, numpoints, lvchoosen);
        dissval = dissim_get(dissims, numpoints, lvchoosen[0], lvchoosen[1]);

        //if the dissimilarity between the pivots is 0, then set 0 for each point for this dimension
        if (dissval == 0.0)
        {
            //for each point in the table
            for (i=0; i < numpoints; i++)
            {
                pos = i*projdim+currentdim;
                *(points+pos) = 0.0;
            }
        }
        else
        {   //if the dissimilarity is not equal to 0, then points iterator
            for (i=0; i < numpoints; i++)
            {
                // current dimension xi = (dissimilarity between the point and the first pivot)^2+
                //                       +(dissimilarity between both pivots)^2-
                //						 -(dissimilarity between the point and the secod pivot)^2)
                // all divided by 2 times the (dissimilarity between both pivots)

                xi = ((pow(dissim_get(dissims,numpoints,lvchoosen[0],i), 2)
                     + pow(dissim_get(dissims,numpoints,lvchoosen[0],lvchoosen[1]), 2)
                     - pow(dissim_get(dissims,numpoints,i,lvchoosen[1]), 2))
                     / (2 * dissim_get(dissims,numpoints,lvchoosen[0],lvchoosen[1])));

                pos =i*projdim+currentdim;
                *(points+pos) = xi;
            }

            // updating the dissimilarities table with equation 4 of Faloutsos' paper (in detail below)
            if (currentdim < projdim-1)
                dissims_update(points, numpoints, projdim, currentdim, dissims);
        }

        // increase the current dimension
        currentdim++;
    }
    return 0;
}

int dissimilar_points_choose(decimal *dissims, int numpoints, int choosen[])
{
    int x=0, y=1;

    int i, j;
    for (i=0; i < numpoints-1; i++)
    {
        for (j=i+1; j < numpoints; j++)
        {
            if (dissim_get(dissims, numpoints, x, y) < dissim_get(dissims, numpoints, i, j))
            {
                x = i;
                y = j;
            }
        }
    }
    choosen[0] = x;
    choosen[1] = y;

    return 0;
}

int dissims_update(decimal *points, int numpoints, int projdim, int currentdim,
                           decimal *dissims)
{
    decimal value=0.0;
    int i, j;
    int posI, posJ;

    //for each point
    for (i=0; i < numpoints-1; i++)
    {
        posI = i*projdim+currentdim;

        //for each another point
        for (j=i+1; j < numpoints; j++)
        {
            posJ = j*projdim+currentdim;
            value = (sqrt(fabs(pow(dissim_get(dissims,numpoints,i,j), 2)
                     - pow((*(points+posI) - *(points+posJ)), 2))));
            dissim_set(dissims, numpoints, i, j, value);
        }
    }
    return 0;
}
