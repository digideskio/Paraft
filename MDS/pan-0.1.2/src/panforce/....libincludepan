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
#ifndef PAN_FORCE_H_INCLUDED
#define PAN_FORCE_H_INCLUDED

#include "pan_useful.h"
#include "pan_math.h"
#include "pan_metric.h"

#ifdef DEBUG_FORCE
    #include "pan_dconv.h"
#endif

/******************************************************************************/
/*                         ENUMERATIONS / STRUCTURES                          */
/******************************************************************************/
enum sampling_enum {
    SAMP_RANDOM = 1,  // default
    SAMP_CLUSTERING,
    SAMP_MAXMIN,
    SAMP_SPAM
};
#define sampling_enum_descr(index) ({      \
    char* name[] = {"Random sampling",     \
                    "Clustering sampling", \
                    "Max-min sampling",    \
                    "Spam"};               \
    name[index-1];                         \
})

enum idmap_init_enum {
    IDMAP_FASTMAP = 1,  // default
    IDMAP_NNP,
    IDMAP_RANDOM
};
#define idmap_init_enum_descr(index) ({                  \
    char *name[] = {"Fastmap",                           \
                    "Nearest Neighbor Projection (NNP)", \
                    "Random"};                           \
    name[index-1];                                       \
})

struct sampling_struct {
    int numpoints;                    // obligatory
    int highdim;                      // obligatory
    int numsamples;                   // obligatory
    enum dissimilarity_enum disstype; // optional: DEFAULT_VALUE = DISS_EUCLIDEAN
    enum sampling_enum sampletype;    // optional: DEFAULT_VALUE = SAMP_RANDOM
};

struct idmap_struct {
    int numpoints;                    // obligatory
    int highdim;                      // obligatory
    int projdim;                      // optional: DEFAULT_VALUE = 2
    int numiterations;                // optional: DEFAULT_VALUE = 50
    decimal fractiondelta;            // optional: DEFAULT_VALUE = 8.0
    enum dissimilarity_enum disstype; // optional: DEFAULT_VALUE = DISS_EUCLIDEAN
    enum idmap_init_enum inittype;    // optional: DEFAULT_VALUE = IDMAP_FASTMAP
};

/******************************************************************************/
/*                                 FUNCTIONS                                  */
/******************************************************************************/
int sampling_execute(decimal *inputdata, char **inputid, struct sampling_struct *inputinfo,
                     decimal *outputdata, char **outputid);

int idmap_execute(decimal *inputdata, char **inputid, struct idmap_struct *inputinfo,
                  decimal* outputdata);

#endif // PAN_FORCE_H_INCLUDED
