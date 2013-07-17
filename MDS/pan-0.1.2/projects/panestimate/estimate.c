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
#include "pan_estimate.h"

// Internal Functions Prototypes
void diss_sort_add(decimal *precurve, int index, decimal hdiss, decimal pdiss);

decimal kruskal_stress(decimal *highdata, decimal* projdata, int numpoints, int highdim,
        int projdim, enum dissimilarity_enum disstype, enum stress_enum stresstype);


// Member variables
static decimal m_elapsedtime = 0.0;

decimal stress_elapsedtime()
{
    return m_elapsedtime;
}

// Functions
decimal* precision_curve_execute(decimal *highdata, int highdim, decimal* projdata,
                                 int projdim, int numpoints)
{
    decimal *precurve  = (decimal*) malloc(numpoints*3*sizeof(decimal));
    decimal *hvector_i = (decimal*) malloc(highdim*sizeof(decimal));
    decimal *hvector_j = (decimal*) malloc(highdim*sizeof(decimal));
    decimal *pvector_i = (decimal*) malloc(projdim*sizeof(decimal));
    decimal *pvector_j = (decimal*) malloc(projdim*sizeof(decimal));
    decimal hdiss, pdiss;

    int register i,j,p;
    for(i=0; i<numpoints; i++)
    {
        // vector i
        for(p=0; p<highdim; p++)
            hvector_i[p] = *(highdata+(i*highdim+p));
        for(p=0; p<projdim; p++)
            pvector_i[p] = *(projdata+(i*projdim+p));

        // dissimilarities
        hdiss=0.0;
        pdiss=0.0;

        for(j=0; j<numpoints; j++)
        {
            if(i==j) continue;

            // vector j
            for(p=0; p<highdim; p++)
                hvector_j[p] = *(highdata+(j*highdim+p));
            for(p=0; p<projdim; p++)
                pvector_j[p] = *(projdata+(j*projdim+p));

            // dissimilarities
            hdiss += euclid_norm(hvector_i, hvector_j, highdim);
            pdiss += euclid_norm(pvector_i, pvector_j, projdim);
        }

        hdiss /= (numpoints-1);
        pdiss /= (numpoints-1);

        diss_sort_add(precurve, i, hdiss, pdiss);
    }

    // release memory
    free(hvector_i);
    free(hvector_j);
    free(pvector_i);
    free(pvector_j);

    return precurve;
}

void diss_sort_add(decimal *precurve, int index, decimal hdiss, decimal pdiss)
{
    // first position
    if(index==0)
    {
        *precurve = index;
        *(precurve+1) = hdiss;
        *(precurve+2) = pdiss;
        return;
    }

    // hdiss greater than previous value
    int currpos = index*3+1;
    int prevpos = (index-1)*3+1;

    if (hdiss >= *(precurve+prevpos))
    {
        *(precurve+currpos-1) = index;
        *(precurve+currpos) = hdiss;
        *(precurve+currpos+1) = pdiss;
        return;
    }

    // search new position
    int register pos;
    for(pos=1; pos<currpos; pos+=3)
    {
        if(hdiss <= *(precurve+pos))
            break;
    }

    // move values forward
    int newpos = pos;
    for(pos=currpos; pos>newpos; pos-=3)
    {
        prevpos = pos-3;
        *(precurve+pos+1) = *(precurve+prevpos+1);
        *(precurve+pos) = *(precurve+prevpos);
        *(precurve+pos-1) = *(precurve+prevpos-1);
    }

    // add new value
    *(precurve+newpos-1) = index;
    *(precurve+newpos) = hdiss;
    *(precurve+newpos+1) = pdiss;
}

decimal stress_calc(decimal *highdata, decimal *projdata, struct stress_struct *stressinfo)
{
    if (stressinfo->projdim == DEFAULT_VALUE) stressinfo->projdim = 2;
    if (stressinfo->disstype == DEFAULT_VALUE) stressinfo->disstype = DISS_EUCLIDEAN;
    if (stressinfo->stresstype == DEFAULT_VALUE) stressinfo->stresstype = STRESS_NORMALIZED_KRUSKAL;

    decimal stressval;

    clock_t start, end;
    start = clock()/(CLOCKS_PER_SEC/1000);

    switch(stressinfo->stresstype)
    {
        case STRESS_KRUSKAL:
        case STRESS_NORMALIZED_KRUSKAL:
            stressval = kruskal_stress(highdata, projdata, stressinfo->numpoints, stressinfo->highdim,
                        stressinfo->projdim, stressinfo->disstype, stressinfo->stresstype);
            break;
        default:
            stressval = NULL_DEC;
    }

    end = clock()/(CLOCKS_PER_SEC/1000);
    m_elapsedtime = (end-start)/1000.0;

    return stressval;
}
