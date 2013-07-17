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
#include "pan_math.h"

static decimal m_randdec = 0.0;
static unsigned long int m_randint = 0;
void random_number(unsigned long int maxvalue);


inline void vectors_add(decimal *result, decimal *vector1, decimal *vector2, int sz)
{
    int i;
    for(i=0; i<sz; i++)
        *(result+i) = *(vector1+i) + *(vector2+i);
}

inline void vectors_subtract(decimal *result, decimal *vector1, decimal *vector2, int sz)
{
    int i;
    for(i=0; i<sz; i++)
        *(result+i) = *(vector1+i) - *(vector2+i);
}

inline void vector_multscalar(decimal *result, decimal scalar, decimal *vector, int sz)
{
    int i;
    for(i=0; i<sz; i++)
        *(result+i) = scalar * *(vector+i);
}

decimal randdec()
{
    m_randdec = 0.0;
    random_number(NULL_INT);
    return m_randdec;
}

unsigned long int randint(unsigned long int maxvalue)
{
    m_randint = 0;
    random_number(maxvalue);
    return m_randint;
}

void random_number(unsigned long int maxvalue)
{
    const gsl_rng_type * T;
    gsl_rng * r;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    long int ltime = time(NULL);
    gsl_rng_set(r, ltime);

    #ifdef DEBUG_MATH
        printf ("generator type: %s\n", gsl_rng_name(r));
        printf ("seed = %lu\n", gsl_rng_default_seed);
        printf ("first value = %lu\n", gsl_rng_get(r));
    #endif

    if (maxvalue == NULL_INT)
        m_randdec = gsl_rng_uniform_pos(r);
    else
        m_randint = gsl_rng_uniform_int(r, maxvalue);

    gsl_rng_free (r);
}


/* This is the jacobi version
 * Author:  G. Jungman
 * Modified by Paulo Joia Filho (single version), September 2011
 *
 * Algorithm due to J.C. Nash, Compact Numerical Methods for
 * Computers (New York: Wiley and Sons, 1979), chapter 3.
 * See also Algorithm 4.1 in
 * James Demmel, Kresimir Veselic, "Jacobi's Method is more
 * accurate than QR", Lapack Working Note 15 (LAWN15), October 1989.
 * Available from netlib.
 *
 * Based on code by Arthur Kosowsky, Rutgers University
 *                  kosowsky@physics.rutgers.edu
 *
 * Another relevant paper is, P.P.M. De Rijk, "A One-Sided Jacobi
 * Algorithm for computing the singular value decomposition on a
 * vector computer", SIAM Journal of Scientific and Statistical
 * Computing, Vol 10, No 2, pp 359-371, March 1989.
 *
 */

int gsl_linalg_float_SV_decomp_jacobi (gsl_matrix_float * A, gsl_matrix_float * Q, gsl_vector_float * S)
{
  if (A->size1 < A->size2)
    {
      /* FIXME: only implemented  M>=N case so far */

      GSL_ERROR ("svd of MxN matrix, M<N, is not implemented", GSL_EUNIMPL);
    }
  else if (Q->size1 != A->size2)
    {
      GSL_ERROR ("square matrix Q must match second dimension of matrix A",
                 GSL_EBADLEN);
    }
  else if (Q->size1 != Q->size2)
    {
      GSL_ERROR ("matrix Q must be square", GSL_ENOTSQR);
    }
  else if (S->size != A->size2)
    {
      GSL_ERROR ("length of vector S must match second dimension of matrix A",
                 GSL_EBADLEN);
    }
  else
    {
      const size_t M = A->size1;
      const size_t N = A->size2;
      size_t i, j, k;

      /* Initialize the rotation counter and the sweep counter. */
      int count = 1;
      int sweep = 0;
      int sweepmax = 5*N;

      float tolerance = 10 * M * GSL_FLT_EPSILON;

      /* Always do at least 12 sweeps. */
      sweepmax = GSL_MAX (sweepmax, 12);

      /* Set Q to the identity matrix. */
      gsl_matrix_float_set_identity (Q);

      /* Store the column error estimates in S, for use during the
         orthogonalization */

      for (j = 0; j < N; j++)
        {
          gsl_vector_float_view cj = gsl_matrix_float_column (A, j);
          float sj = gsl_blas_snrm2 (&cj.vector);
          gsl_vector_float_set(S, j, GSL_FLT_EPSILON * sj);
        }

      /* Orthogonalize A by plane rotations. */

      while (count > 0 && sweep <= sweepmax)
        {
          /* Initialize rotation counter. */
          count = N * (N - 1) / 2;

          for (j = 0; j < N - 1; j++)
            {
              for (k = j + 1; k < N; k++)
                {
                  float a = 0.0;
                  float b = 0.0;
                  float p = 0.0;
                  float q = 0.0;
                  float cosine, sine;
                  float v;
                  float abserr_a, abserr_b;
                  int sorted, orthog, noisya, noisyb;

                  gsl_vector_float_view cj = gsl_matrix_float_column (A, j);
                  gsl_vector_float_view ck = gsl_matrix_float_column (A, k);

                  gsl_blas_sdot (&cj.vector, &ck.vector, &p);
                  p *= 2.0 ;  /* equation 9a:  p = 2 x.y */

                  a = gsl_blas_snrm2 (&cj.vector);
                  b = gsl_blas_snrm2 (&ck.vector);

                  q = a * a - b * b;
                  v = hypot(p, q);

                  /* test for columns j,k orthogonal, or dominant errors */

                  abserr_a = gsl_vector_float_get(S,j);
                  abserr_b = gsl_vector_float_get(S,k);

                  sorted = (GSL_COERCE_FLT(a) >= GSL_COERCE_FLT(b));
                  orthog = (fabs (p) <= tolerance * GSL_COERCE_FLT(a * b));
                  noisya = (a < abserr_a);
                  noisyb = (b < abserr_b);

                  if (sorted && (orthog || noisya || noisyb))
                    {
                      count--;
                      continue;
                    }

                  /* calculate rotation angles */
                  if (v == 0 || !sorted)
                    {
                      cosine = 0.0;
                      sine = 1.0;
                    }
                  else
                    {
                      cosine = sqrt((v + q) / (2.0 * v));
                      sine = p / (2.0 * v * cosine);
                    }

                  /* apply rotation to A */
                  for (i = 0; i < M; i++)
                    {
                      const float Aik = gsl_matrix_float_get (A, i, k);
                      const float Aij = gsl_matrix_float_get (A, i, j);
                      gsl_matrix_float_set (A, i, j, Aij * cosine + Aik * sine);
                      gsl_matrix_float_set (A, i, k, -Aij * sine + Aik * cosine);
                    }

                  gsl_vector_float_set(S, j, fabs(cosine) * abserr_a + fabs(sine) * abserr_b);
                  gsl_vector_float_set(S, k, fabs(sine) * abserr_a + fabs(cosine) * abserr_b);

                  /* apply rotation to Q */
                  for (i = 0; i < N; i++)
                    {
                      const float Qij = gsl_matrix_float_get (Q, i, j);
                      const float Qik = gsl_matrix_float_get (Q, i, k);
                      gsl_matrix_float_set (Q, i, j, Qij * cosine + Qik * sine);
                      gsl_matrix_float_set (Q, i, k, -Qij * sine + Qik * cosine);
                    }
                }
            }

          /* Sweep completed. */
          sweep++;
        }

      /*
       * Orthogonalization complete. Compute singular values.
       */

      {
        double prev_norm = -1.0;

        for (j = 0; j < N; j++)
          {
            gsl_vector_float_view column = gsl_matrix_float_column (A, j);
            float norm = gsl_blas_snrm2 (&column.vector);

            /* Determine if singular value is zero, according to the
               criteria used in the main loop above (i.e. comparison
               with norm of previous column). */

            if (norm == 0.0 || prev_norm == 0.0
                || (j > 0 && norm <= tolerance * prev_norm))
              {
                gsl_vector_float_set (S, j, 0.0);     /* singular */
                gsl_vector_float_set_zero (&column.vector);   /* annihilate column */

                prev_norm = 0.0;
              }
            else
              {
                gsl_vector_float_set (S, j, norm);    /* non-singular */
                gsl_vector_float_scale (&column.vector, 1.0 / norm);  /* normalize column */

                prev_norm = norm;
              }
          }
      }

      if (count > 0)
        {
          /* reached sweep limit */
          GSL_ERROR ("Jacobi iterations did not reach desired tolerance",
                     GSL_ETOL);
        }

      return GSL_SUCCESS;
    }
}

