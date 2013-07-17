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
#ifndef PAN_METRIC_H_INCLUDED
#define PAN_METRIC_H_INCLUDED

/******************************************************************************/
/*                                   INCLUDES                                 */
/******************************************************************************/
#include "pan_useful.h"
#include <math.h>

/******************************************************************************/
/*                                 ENUMERATIONS                               */
/******************************************************************************/
enum dissimilarity_enum {
    DISS_EUCLIDEAN = 1,   // default
    DISS_EUCLIDEAN2,
    DISS_CITY_BLOCK,
    DISS_COSINE_BASED,
    DISS_INFINITY_NORM
};
#define dissimilarity_enum_descr(index) ({          \
    char *name[] = { "Euclidean",                   \
                     "Euclidean at the square",     \
                     "City-block",                  \
                     "Cosine-based dissimilarity",  \
                     "Infinity norm"};              \
    name[index-1];                                  \
})

/******************************************************************************/
/*                                   FUNCTIONS                                */
/******************************************************************************/
inline decimal euclid_norm(decimal *pointI, decimal *pointJ, int sz);
inline decimal euclid2_norm(decimal *pointI, decimal *pointJ, int sz);

int dissims_calc(decimal *points, int numpoints, int dimension,
                 enum dissimilarity_enum disstype, decimal *dissims);

inline decimal dissim_get(decimal *dissims, int numpoints, int indexI,
                          int indexJ);
inline void dissim_set(decimal *dissims, int numpoints, int indexI,
                       int indexJ, decimal value);

inline decimal dissim_get2(decimal *points, int dimension, int indexI,
                           int indexJ, enum dissimilarity_enum disstype);
inline decimal dissim_get3(decimal *pointI, decimal *pointJ, int dimension,
                           enum dissimilarity_enum disstype);

#endif // PAN_METRIC_H_INCLUDED
