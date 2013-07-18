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
#ifndef PAN_MATH_H_INCLUDED
#define PAN_MATH_H_INCLUDED

/******************************************************************************/
/*                                   INCLUDES                                 */
/******************************************************************************/
#include "pan_useful.h"

#include <gsl/config.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

/******************************************************************************/
/*                                   FUNCTIONS                                */
/******************************************************************************/
// Add two vectors decimal pointer
inline void vectors_add(decimal *result, decimal *vector1, decimal *vector2, int sz);

// Subtract two vectors decimal pointer
inline void vectors_subtract(decimal *result, decimal *vector1, decimal *vector2, int sz);

// Multiply a vector by scalar
inline void vector_multscalar(decimal *result, decimal scalar, decimal *vector, int sz);

// Get a decimal random number between 0 and 1
decimal randdec();

// Get a integer random number between 0 and maxvalue
unsigned long int randint(unsigned long int maxvalue);

// SVD decomposition as float
int gsl_linalg_float_SV_decomp_jacobi (gsl_matrix_float * A, gsl_matrix_float * Q, gsl_vector_float * S);


#endif // PAN_MATH_H_INCLUDED
