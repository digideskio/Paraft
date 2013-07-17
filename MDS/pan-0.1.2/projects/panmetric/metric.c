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
#include "pan_metric.h"

inline decimal euclid_norm(decimal *pointI, decimal *pointJ, int sz)
{
    decimal dist = euclid2_norm(pointI, pointJ, sz);
    return sqrt(dist);
}

inline decimal euclid2_norm(decimal *pointI, decimal *pointJ, int sz)
{
    int i;
    decimal diff=0.0;
    decimal dist=0.0;

    for(i=0; i<sz; i++)
    {
        diff = (pointJ[i]-pointI[i]);
        dist+= diff*diff;
    }

    return dist;
}

int dissims_calc(decimal *points, int numpoints, int dimension,
    enum dissimilarity_enum disstype, decimal *dissims)
{
    decimal *pointI = (decimal*) malloc (dimension*sizeof(decimal));
    decimal *pointJ = (decimal*) malloc (dimension*sizeof(decimal));

    int i, j, pos=0;
    for (i=0; i < numpoints; i++)
    {
        decptrcpy(pointI, NULL_INT, NULL_INT, points, i*dimension, (i+1)*dimension-1);

        for (j=0; j < numpoints; j++)
        {
            if (i>=j) continue;
            decptrcpy(pointJ, NULL_INT, NULL_INT, points, j*dimension, (j+1)*dimension-1);

            switch(disstype)
            {
                case DISS_EUCLIDEAN:
                    *(dissims+pos) = euclid_norm(pointI, pointJ, dimension);
                    break;
                case DISS_EUCLIDEAN2:
                    *(dissims+pos) = euclid2_norm(pointI, pointJ, dimension);
                    break;
                default:
                    *(dissims+pos) = NULL_DEC;
            }

            pos++;
        }
    }

    // release memory
    free(pointI);
    free(pointJ);

    return 0;
}

/*=======================================================*
 * Example:    numpoints = 5                             *
 * dissims =   [d01 d02 d03 d04 d12 d13 d14 d23 d24 d34] *
 * positions = [0   1   2   3   4   5   6   7   8   9  ] *
 *=======================================================*/

inline decimal dissim_get(decimal *dissims, int numpoints, int indexI, int indexJ)
{
    if ((indexI >= numpoints) || (indexJ >= numpoints))
        return -1.0;
    if (indexI == indexJ)
        return 0.0;
    if (indexI > indexJ) {
        int i_aux = indexI;
        int j_aux = indexJ;
        indexI = j_aux;
        indexJ = i_aux;
    }
    int pos = ((2*numpoints-indexI-1)*indexI)/2;
    pos += (indexJ-indexI-1);
    decimal dissIJ = *(dissims+pos);

    return dissIJ;
}

inline void dissim_set(decimal *dissims, int numpoints, int indexI, int indexJ, decimal value)
{
    if ((indexI >= numpoints) || (indexJ >= numpoints))
        return;
    if (indexI == indexJ)
        return;
    if (indexI > indexJ) {
        int i_aux = indexI;
        int j_aux = indexJ;
        indexI = j_aux;
        indexJ = i_aux;
    }
    int pos = ((2*numpoints-indexI-1)*indexI)/2;
    pos += (indexJ-indexI-1);
    *(dissims+pos) = value;
}

inline decimal dissim_get2(decimal *points, int dimension, int indexI,
                           int indexJ, enum dissimilarity_enum disstype)
{
    decimal *pointI = (decimal*) malloc (dimension*sizeof(decimal));
    decimal *pointJ = (decimal*) malloc (dimension*sizeof(decimal));
    decptrcpy(pointI, NULL_INT, NULL_INT, points, indexI*dimension, (indexI+1)*dimension-1);
    decptrcpy(pointJ, NULL_INT, NULL_INT, points, indexJ*dimension, (indexJ+1)*dimension-1);

    decimal dissIJ = dissim_get3(pointI, pointJ, dimension, disstype);

    // release memory
    free(pointI);
    free(pointJ);

    return dissIJ;
}

inline decimal dissim_get3(decimal *pointI, decimal *pointJ, int dimension,
                           enum dissimilarity_enum disstype)
{
    decimal dissIJ;

    switch(disstype)
    {
        case DISS_EUCLIDEAN:
            dissIJ = euclid_norm(pointI, pointJ, dimension);
            break;
        case DISS_EUCLIDEAN2:
            dissIJ = euclid2_norm(pointI, pointJ, dimension);
            break;
        default:
            dissIJ = NULL_DEC;
    }

    return dissIJ;
}
